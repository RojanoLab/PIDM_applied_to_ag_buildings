import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.diffusion_new import ConditionalModel as CModel
from models.diffusion_new import Model
from functions.denoising_step import guided_ddpm_steps, guided_ddim_steps, ddpm_steps, ddim_steps

from einops import rearrange
import pickle
from torch.optim.sgd import SGD
################ Some definitions as part of the Physics Informed Condition ##################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KE_EPS_PATH = os.path.join(_REPO_ROOT, "data", "prediction_k_eps_mean.npz")
_KE_EPS_CACHE = {}

def _get_ke_init_tensors(target_device):
    cache_key = str(target_device)
    if cache_key not in _KE_EPS_CACHE:
        with np.load(_KE_EPS_PATH) as data_npz:
            k_load = data_npz["k"]
            eps_load = data_npz["epsilon"]

        k_init = torch.from_numpy(k_load).to(target_device, dtype=torch.float32)
        eps_init = torch.from_numpy(eps_load).to(target_device, dtype=torch.float32)
        k_init = torch.clamp(k_init, min=1e-8)
        eps_init = torch.clamp(eps_init, min=1e-8)
        _KE_EPS_CACHE[cache_key] = (k_init, eps_init)
    return _KE_EPS_CACHE[cache_key]


def _coerce_ke_tensors(ke_tensor, target_device):
    if ke_tensor is None:
        return None

    if isinstance(ke_tensor, dict):
        k_src = ke_tensor["k"]
        eps_src = ke_tensor["epsilon"]
    elif isinstance(ke_tensor, (tuple, list)) and len(ke_tensor) >= 2:
        k_src, eps_src = ke_tensor[0], ke_tensor[1]
    else:
        return None

    k = torch.as_tensor(k_src, dtype=torch.float32, device=target_device).detach()
    eps = torch.as_tensor(eps_src, dtype=torch.float32, device=target_device).detach()
    k = torch.clamp(k, min=1e-8)
    eps = torch.clamp(eps, min=1e-8)
    return k, eps

# (convert_to_physical, solve_poisson_iterative, precompute_static_derivatives, compute_residuals_jit)
@torch.jit.script
def convert_to_physical(img_float, physical_min: float, physical_max: float, pix_max: float, pix_min: float):
    img_clamped = torch.clamp(img_float, min=pix_min, max=pix_max)
    normalized_img = (img_clamped - pix_min) / (pix_max - pix_min)
    return normalized_img * (physical_max - physical_min) + physical_min

@torch.jit.script
def solve_poisson_iterative(source_term, p_init, dx: float, dy: float, iterations: int):
    p = p_init.clone()
    ax = dy / dx
    ay = dx / dy
    ap = -2 * (ax + ay)
    const = source_term * dx * dy
    
    for _ in range(iterations):
        p_E = torch.roll(p, -1, 1); p_W = torch.roll(p, 1, 1)
        p_N = torch.roll(p, -1, 0); p_S = torch.roll(p, 1, 0)
        p = (ax * (p_E + p_W) + ay * (p_N + p_S) - const) / -ap
        # BCs
        p[:, 0] = p[:, 1]; p[:, -1] = 0.0
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
    return p

@torch.jit.script
def precompute_static_derivatives(u, v, dx: float, dy: float):
    u_E = torch.roll(u, -1, 1); u_W = torch.roll(u, 1, 1)
    u_N = torch.roll(u, -1, 0); u_S = torch.roll(u, 1, 0)
    v_E = torch.roll(v, -1, 1); v_W = torch.roll(v, 1, 1)
    v_N = torch.roll(v, -1, 0); v_S = torch.roll(v, 1, 0)

    du_dx = (u_E - u_W) / (2 * dx); dv_dy = (v_N - v_S) / (2 * dy)
    du_dy = (u_N - u_S) / (2 * dy); dv_dx = (v_E - v_W) / (2 * dx)

    S2_static = 2*(du_dx**2) + 2*(dv_dy**2) + (du_dy + dv_dx)**2
    return S2_static, du_dx, du_dy, dv_dx, dv_dy, u_E, u_W, u_N, u_S

@torch.jit.script
def compute_residuals_jit( k_val, eps_val, alpha_val, u, v, S2_static, dp_dx_static, adv_term_u_static, dx: float, dy: float, rho: float, nu: float):
    k = k_val
    eps = eps_val
    alpha = 4.0 * torch.sigmoid(alpha_val) - 2.0

    C_mu = 0.09
    nu_t = C_mu * (k**2) / (eps + 1e-10)
    effective_nu = nu + nu_t 

    # Momentum Residual
    u_E = torch.roll(u, -1, 1); u_W = torch.roll(u, 1, 1)
    u_N = torch.roll(u, -1, 0); u_S = torch.roll(u, 1, 0)
    
    eff_nu_e = (effective_nu + torch.roll(effective_nu, -1, 1)) / 2
    eff_nu_w = (effective_nu + torch.roll(effective_nu, 1, 1)) / 2
    eff_nu_n = (effective_nu + torch.roll(effective_nu, -1, 0)) / 2
    eff_nu_s = (effective_nu + torch.roll(effective_nu, 1, 0)) / 2
    
    diff_term_u = ((rho * eff_nu_e * (u_E - u) / dx) - (rho * eff_nu_w * (u - u_W) / dx)) / dx + \
                  ((rho * eff_nu_n * (u_N - u) / dy) - (rho * eff_nu_s * (u - u_S) / dy)) / dy
    
    res_mom = ( adv_term_u_static) + ((1 - alpha)* dp_dx_static) - diff_term_u
    
    res = res_mom / rho
    return torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
################ End of Some definitions as part of the Physics Informed Condition ##################################

class MetricLogger(object):
    def __init__(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        self.metric_dict = {}
        self.reset()

    def reset(self):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key] = []

    @torch.no_grad()
    def update(self, **kwargs):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=''):
        with open(os.path.join(outdir, f'metric_log_{postfix}.pkl'), 'wb') as f:
            pickle.dump(self.metric_dict, f)

def normalize_to_01(image_array):#Scales a NumPy array to the range [0, 1]
    min_val = -7
    max_val = 17
    if max_val - min_val > 0:
        return (image_array - min_val) / 24
    else:
        return torch.zeros(image_array.shape)
    
################ Main code for the Physics Informed Condition #########################
def main_code(w, w2, ke_tensor=None):
    local_device = w.device
    w = w.clone()
    w.requires_grad_(True)
    w2 = w2.clone()
    w2.requires_grad_(True)
    wr = w.clone()
    wr.requires_grad_(True)

    # Physics Parameters
    params = {"dx": 0.071640625, "nu": 1.46e-5, "rho": 1.225, "mu": 1.78e-5}
    OPTIM_STEPS = 100  ###it is small because the value is close to the optimal one

    u_phys = convert_to_physical(w, -7.0, 17.0, 246.0, 0)
    v_phys = convert_to_physical(w2, -5.0, 9.0, 244.0, 0)

    # Use detached tensors inside the inner optimizer loop to avoid
    u_phys_opt = u_phys.detach()
    v_phys_opt = v_phys.detach()
    
    # 3. Precompute Static Fields
    dx, dy = params["dx"], params["dx"]
    rho, mu = params["rho"], params["mu"]
    
    S2_static, du_dx, du_dy, dv_dx, dv_dy, u_E, u_W, u_N, u_S = precompute_static_derivatives(u_phys, v_phys, dx, dy)
    
    d2u_dx2 = (u_E - 2*u_phys + u_W)/dx**2; d2u_dy2 = (u_N - 2*u_phys + u_S)/dy**2
    d2v_dx2 = (torch.roll(v_phys,-1,1) - 2*v_phys + torch.roll(v_phys,1,1))/dx**2 
    d2v_dy2 = (torch.roll(v_phys,-1,0) - 2*v_phys + torch.roll(v_phys,1,0))/dy**2
    
    Fx = rho*(u_phys*du_dx + v_phys*du_dy) - mu*(d2u_dx2 + d2u_dy2)
    Fy = rho*(u_phys*dv_dx + v_phys*dv_dy) - mu*(d2v_dx2 + d2v_dy2)
    
    div_F = (torch.roll(Fx,-1,1)-torch.roll(Fx,1,1))/(2*dx) + (torch.roll(Fy,-1,0)-torch.roll(Fy,1,0))/(2*dy)
    source_term = -div_F
    
    p_init = torch.zeros_like(source_term)
    p_field = solve_poisson_iterative(source_term, p_init, dx, dy, 50)
    dp_dx_static = (torch.roll(p_field,-1,1) - torch.roll(p_field,1,1)) / (2*dx)

    mask_u = (u_phys > 0).float()
    du_dx_up = mask_u * (u_phys - u_W)/dx + (1 - mask_u) * (u_E - u_phys)/dx
    mask_v = (v_phys > 0).float()
    du_dy_up = mask_v * (u_phys - u_S)/dy + (1 - mask_v) * (u_N - u_phys)/dy
    adv_term_u_static = (u_phys * du_dx_up + v_phys * du_dy_up)
    # Load once per process and reuse for all subsequent calls, unless external tensors are provided.
    ke_pair = _coerce_ke_tensors(ke_tensor, local_device)
    if ke_pair is None:
        k, eps = _get_ke_init_tensors(local_device)
    else:
        k, eps = ke_pair
    # Convert arrays to tensors for JIT function inputs.
    # Initialize Learnable Parameters based on the trained values from the surrogate model. We use a sigmoid transformation to ensure alpha stays within a reasonable range during optimization.
    alpha = torch.tensor([-0.7569], device=local_device, requires_grad=True)
    optimizer = SGD([alpha], lr=0.05)
    
    # Scaling factors for loss
    U_scale = torch.sqrt(u_phys_opt**2 + v_phys_opt**2) + 1e-10
    L_scale = params["dx"] * u_phys.shape[1]
    scale_mom = (U_scale**2/L_scale)
    res_acc_mom = torch.zeros_like(u_phys)
    S2_static_opt = S2_static.detach()
    dp_dx_static_opt = dp_dx_static.detach()
    adv_term_u_static_opt = adv_term_u_static.detach()

    # Run Optimization
    for epoch in range(OPTIM_STEPS):
        optimizer.zero_grad()
        res_acc_mom = compute_residuals_jit(
            k, eps, alpha, u_phys_opt, v_phys_opt, S2_static_opt,
            dp_dx_static_opt, adv_term_u_static_opt, dx, dy, rho, params["nu"]
        )
    
        # Loss Calculation
        total_loss = torch.mean(((res_acc_mom) / scale_mom)**2)
        loss_alpha_reg = 0.00001 * (alpha ** 2).mean()
        total_loss = total_loss + loss_alpha_reg
        total_loss.backward()
        optimizer.step()
        
    # Recompute residual on the original graph once, using optimized coefficients.
    res_acc_mom = compute_residuals_jit(
        k.detach(), eps.detach(), alpha.detach(), u_phys, v_phys, S2_static,
        dp_dx_static, adv_term_u_static, dx, dy, rho, params["nu"]
    )
    dt_char = params["dx"] / (torch.max(torch.sqrt(u_phys**2 + v_phys**2)) + 1e-10)
    vel_res = res_acc_mom * dt_char
    vel_res_u=normalize_to_01(vel_res)
    return vel_res_u

############################### Retrieving and Incorporating Physics Informed Conditions ##############################
def velocity_residual(w, w2, calc_grad=True, ke_tensor=None):
    if w.ndim == 3:  
        w = w.unsqueeze(0)
    if w2.ndim == 3:   
        w2 = w2.unsqueeze(0)
    # Detect the device of the generated image (w).
    device = w.device 
    w2 = w2.to(device)
    if ke_tensor is not None:
        ke_tensor = _coerce_ke_tensors(ke_tensor, device)
    w = w.clone().requires_grad_(True)
    w2 = w2.clone().requires_grad_(True)

    with torch.enable_grad():
        residual = main_code(w, w2, ke_tensor=ke_tensor)
    residual_loss = (residual**2).mean()

    if calc_grad:
        dw = torch.autograd.grad(residual_loss, w, retain_graph=True)[0]
        return dw, residual_loss
    else:
        return residual_loss
####################### End of the Physics Informed Conditions ########################################################

def load_recons_data(ref_path_gtx, ref_data_gux_path, ref_data_gty_path, ref_data_ke_path, smoothing, smoothing_scale):
    
    ref_data_gtx = np.load(ref_path_gtx).astype(np.float32)  # X VELOCITIES (GT), NPY. 
    data_mean = np.mean(ref_data_gtx[0:])
    data_scale = np.std(ref_data_gtx[0:])  #It is set to 0 because we only have one sample. [1,144,256,512]
    
    ref_data_gtx = ref_data_gtx[-4:, ...].copy().astype(np.float32)   
    ref_data_gtx = torch.as_tensor(ref_data_gtx, dtype=torch.float32)
    
    ref_data_gux = np.load(ref_data_gux_path).astype(np.float32)
    ref_data_gty = np.load(ref_data_gty_path).astype(np.float32)
    ref_data_ke_raw = np.load(ref_data_ke_path)
    if hasattr(ref_data_ke_raw, "files") and "k" in ref_data_ke_raw.files and "epsilon" in ref_data_ke_raw.files:
        ref_data_ke = {
            "k": ref_data_ke_raw["k"].astype(np.float32),
            "epsilon": ref_data_ke_raw["epsilon"].astype(np.float32),
        }
        ref_data_ke_raw.close()
    else:
        ref_data_ke = np.asarray(ref_data_ke_raw, dtype=np.float32)

    ref_data_gux = ref_data_gux[-4:, ...].copy().astype(np.float32)   
    ref_data_gux = torch.as_tensor(ref_data_gux, dtype=torch.float32)

    ref_data_gty = ref_data_gty[-4:, ...].copy().astype(np.float32)    
    ref_data_gty = torch.as_tensor(ref_data_gty, dtype=torch.float32)

    
    flattened_ref_data_gux = []
    flattened_ref_data_gtx = []
    flattened_ref_data_gty = []
    
    for i in range(ref_data_gtx.shape[0]):
        
        for j in range(ref_data_gtx.shape[1] - 2):
            
            flattened_ref_data_gtx.append(ref_data_gtx[i, j:j + 3, ...])
            flattened_ref_data_gux.append(ref_data_gux[i, j:j + 3, ...])
            flattened_ref_data_gty.append(ref_data_gty[i, j:j + 3, ...])    
                
    flattened_ref_data_gtx = torch.stack(flattened_ref_data_gtx, dim=0)
    flattened_ref_data_gux = torch.stack(flattened_ref_data_gux, dim=0)
    flattened_ref_data_gty = torch.stack(flattened_ref_data_gty, dim=0)
    return flattened_ref_data_gtx, flattened_ref_data_gux, flattened_ref_data_gty, ref_data_ke, data_mean.item(), data_scale.item() 

class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return (x - self.min) #/ (self.max - self.min)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def scale(self):
        return self.max - self.min

class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def slice2sequence(data):
    data = rearrange(data[:, 1:2], 't f h w -> (t f) h w')
    return data

def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))

def l2_loss(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt() 

def check_valid_image(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"{tensor_name} contains NaNs!")
    if torch.isinf(tensor).any():
        print(f"{tensor_name} contains infinite values!")
    if tensor.max() == tensor.min():
        print(f"{tensor_name} has constant values.")

################ Definition of Scheduler for Diffusion #########################################
def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion(object):
    def __init__(self, args, config, logger, log_dir, device=None):
        self.args = args
        self.config = config
        self.logger = logger
        self.image_sample_dir = log_dir

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def log(self, info):
        self.logger.info(info)

    def reconstruct(self):
        self.log('Doing sparse reconstruction task')
        self.log("Loading model")

        if self.config.model.type == 'conditional':
            print('Using conditional model')
            model = CModel(self.config)
        else:
            print('Using unconditional model')
            model = Model(self.config)

        model.load_state_dict(torch.load(self.config.model.ckpt_path)[-1])

        model.to(self.device)

        self.log("Model loaded")

        model.eval()
        self.log('Preparing data')
        ref_data, blur_data, ref_data3, ref_data_ke, data_mean, data_std = load_recons_data(self.config.data.data_dir_gtx, self.config.data.data_dir_gux, self.config.data.data_dir_gty, self.config.data.data_dir_ke, smoothing=self.config.data.smoothing,smoothing_scale=self.config.data.smoothing_scale)
        
        scaler = StdScaler(data_mean, data_std)

        self.log("Start sampling")

        testset = torch.utils.data.TensorDataset(ref_data,blur_data,ref_data3)
        
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.config.sampling.batch_size,
                                                  shuffle=False, num_workers=self.config.data.num_workers)
        
        for batch_index,(data, blur_data, ref_data3) in enumerate(test_loader):
            print(batch_index)
            self.log('Batch: {} / Total batch {}'.format(batch_index, len(test_loader)))
            
            x0 = blur_data.to(self.device)
            y0=ref_data3.to(self.device)
            gt = data.to(self.device)

            x0 = x0.squeeze(0)  # Removes the first dimension (size 1)
            gt = gt.squeeze(0)  # Removes the first dimension (size 1)
            y0 = y0.squeeze(0) 

            self.log('Preparing reference image')
            self.log('Dumping visualization...')

            sample_folder = 'sample_batch{}'.format(batch_index)
            ensure_dir(os.path.join(self.image_sample_dir, sample_folder))

            gt_residual = velocity_residual(gt, y0, calc_grad=True, ke_tensor=ref_data_ke)[1].detach()
            self.log('Residual reference: {}'.format(gt_residual.item()))
            init_residual = velocity_residual(x0, y0, calc_grad=True, ke_tensor=ref_data_ke)[1].detach()
            self.log('Residual init: {}'.format(init_residual.item()))
            
            x0 = scaler(x0)
            check_valid_image(x0, "Scaled x0")
        
            xinit = x0.clone()
            
            # prepare loss function
            if self.config.sampling.log_loss:
                l2_loss_fn = lambda x: l2_loss(scaler.inverse(x).to(gt.device), gt)
               
                equation_loss_fn = lambda x: velocity_residual(scaler.inverse(x), y0, calc_grad=False, ke_tensor=ref_data_ke)

                logger = MetricLogger({
                    'l2 loss': l2_loss_fn,
                    'residual loss': equation_loss_fn
                })
                # we repeat the sampling for multiple times
                for repeat in range(self.args.repeat_run):
                    self.log(f'=== Run No.{repeat} ===')
                                    
                    x0 = xinit.clone()
                    for it in range(self.args.sample_step):  # we run the sampling for multiple steps
                        if it == 0:
                            
                            self.log(f'--- Iteration {it} of Run No.{repeat} ---')                           
                            e = torch.randn_like(x0)
                            total_noise_levels = int(self.args.t * (1** it))
                            a = (1 - self.betas).cumprod(dim=0)
                            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                            # Default no-op gradient for non-conditional paths.
                            physical_gradient_func = lambda x: torch.zeros_like(x)
                                                                
                            # Setting up the physical gradient function
                            if self.config.model.type == 'conditional':
                                self.log('Using conditional model with vorticity residual gradient guidance.')
                               
                                physical_gradient_func = lambda x: velocity_residual(scaler.inverse(x), y0, calc_grad=True, ke_tensor=ref_data_ke)[0] / scaler.scale()

                            num_of_reverse_steps = int(self.args.reverse_steps * (1 ** it))                
                            betas = self.betas.to(self.device)
                            skip = total_noise_levels // num_of_reverse_steps
                            seq = range(0, total_noise_levels, skip)      
                            # Performing guided diffusion sampling
                            if self.config.model.type == 'conditional':
                                self.log('Performing guided DDIM steps with conditional model...')
                                xs, _ = guided_ddim_steps(x, seq, model, betas,
                                                        w=self.config.sampling.guidance_weight,
                                                        dx_func=physical_gradient_func, cache=False, logger=logger)
                            elif self.config.sampling.lambda_ > 0:
                                self.log('Performing guided DDIM steps with lambda > 0...')
                                xs, _ = ddim_steps(x, seq, model, betas,
                                                dx_func=physical_gradient_func, cache=True, logger=logger)
                            else:
                                self.log('Performing standard DDIM steps...')
                                xs, _ = ddim_steps(x, seq, model, betas, cache=True, logger=logger)

                            self.log(f'Sequence of images (xs) generated for iteration {it}. Total steps: {len(xs)}')
                                                
                            x = xs[-1]  # Get the final image
                            x0 = xs[-1].to(self.device)

                            self.log(f'Imaged saved as comparison_run_{repeat}_it{it}.png.')
                            # Optionally dump arrays
                            if self.config.sampling.dump_arr:
                                np.save(os.path.join(self.image_sample_dir, sample_folder, f'sample_arr_run_{repeat}_it{it}.npy'),
                                        slice2sequence(scaler.inverse(x)).cpu().numpy())
                            
                            # Log losses if enabled
                            if self.config.sampling.log_loss:
                                logger.log(os.path.join(self.image_sample_dir, sample_folder), f'run_{repeat}_it{it}')
                                logger.reset()
                                print(f'Logged and reset the logger for iteration {it}.')
                        else:
                            print(f'TESTING THIS CODE !!!!')
                    print(f'=== Finished Run No.{repeat} ===')
            self.log('Finished batch {}'.format(batch_index))
            self.log('========================================================')
        self.log('Finished sampling')
