import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from models.diffusion_new import ConditionalModel as CModel
from models.diffusion_new import Model
from functions.denoising_step import guided_ddpm_steps, guided_ddim_steps, ddpm_steps, ddim_steps

from einops import rearrange
import pickle
from torch.optim.sgd import SGD
import sys
import os
# Add train_ddpm to path for imports
train_ddpm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_ddpm'))
sys.path.insert(0, train_ddpm_path)
# Import directly from the losses.py file
import importlib.util
spec = importlib.util.spec_from_file_location("losses", os.path.join(train_ddpm_path, "functions/losses.py"))
losses_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses_module)
get_safe_fluid_mask = losses_module.get_safe_fluid_mask
compute_5point_divergence = losses_module.compute_5point_divergence

# Override velocity_residual with a version that handles detach correctly for inference
def velocity_residual(w: torch.Tensor, 
                      w2: torch.Tensor, 
                      safe_mask: torch.Tensor, 
                      dx_spacing: float = 0.0358203125, 
                      dy_spacing: float = 0.0358203125, 
                      calc_grad: bool = False):
    """
    Inference version of velocity_residual with proper detach handling.
    """
    # Detach first to break computation graph, then clone and set requires_grad
    w = w.detach().clone().requires_grad_(True)
    w2 = w2.detach().clone().requires_grad_(True)

    # 1. Compute raw 5-point spatial divergence
    vel_res = compute_5point_divergence(w, w2, dx=dx_spacing, dy=dy_spacing)
    masked_res = vel_res * safe_mask

    residual_loss = torch.nan_to_num((masked_res**2).mean(), nan=1e6, posinf=1e6, neginf=1e6)

    if not calc_grad:
        return residual_loss
   
    grads = torch.autograd.grad(residual_loss, [w, w2], create_graph=False, retain_graph=False)
    wr_u, wr_v = grads[0], grads[1]

    dx = torch.cat([wr_u, wr_v], dim=1)
    
    return residual_loss, dx
################ Some definitions as part of the Physics Informed Condition ##################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_num_workers(requested_workers):
    if os.name == "nt" and requested_workers > 0:
        return 0
    return requested_workers

def _model_in_channels(model) -> int:
    if hasattr(model, "module") and hasattr(model.module, "in_channels"):
        return int(model.module.in_channels)
    if hasattr(model, "in_channels"):
        return int(model.in_channels)
    raise AttributeError("Model does not expose in_channels")


def _converted_to_physical_linear_params(data_mean, data_scale,
                                         physical_min: float, physical_max: float,
                                         pix_max: float, pix_min: float):
    if pix_max <= pix_min:
        raise ValueError(f"Invalid pixel bounds: pix_max={pix_max}, pix_min={pix_min}")

    data_mean = float(data_mean)
    data_scale = float(data_scale)
    gain = (physical_max - physical_min) / (pix_max - pix_min)

    scale = data_scale * gain
    offset = (data_mean - pix_min) * gain + physical_min
    return offset, scale


def _pixel_to_physical(x, pix_min: float, pix_max: float, physical_min: float, physical_max: float):
    if pix_max <= pix_min:
        raise ValueError(f"Invalid pixel bounds: pix_max={pix_max}, pix_min={pix_min}")
    x = x.to(torch.float32)
    return (x - pix_min) / (pix_max - pix_min) * (physical_max - physical_min) + physical_min


def _is_state_dict_like(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    sample_value = next(iter(obj.values()))
    return torch.is_tensor(sample_value)


def _extract_model_state_dict(ckpt_obj, prefer_ema=False):
    if _is_state_dict_like(ckpt_obj):
        return ckpt_obj

    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt_obj and _is_state_dict_like(ckpt_obj[key]):
                return ckpt_obj[key]

    if isinstance(ckpt_obj, (list, tuple)):
        if prefer_ema and len(ckpt_obj) >= 5 and _is_state_dict_like(ckpt_obj[4]):
            return ckpt_obj[4]
        for item in ckpt_obj:
            if _is_state_dict_like(item):
                return item

    raise ValueError("Could not find a model state dict inside checkpoint")

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
            with torch.enable_grad():
                self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=''):
        with open(os.path.join(outdir, f'metric_log_{postfix}.pkl'), 'wb') as f:
            pickle.dump(self.metric_dict, f)

def load_recons_data(ref_path_gtx, ref_data_gux_path, ref_data_gty_path, ref_data_guy_path,
                     stat_path_x, stat_path_y, smoothing, smoothing_scale):
    
    ref_data_gtx = np.load(ref_path_gtx).astype(np.float32)  # X VELOCITIES (GT), NPY. 
    with np.load(stat_path_x) as stats_x:
        data_mean = float(stats_x['mean'])
        data_scale = float(stats_x['scale'])
    
    ref_data_gtx = ref_data_gtx[-4:, ...].copy().astype(np.float32)   
    ref_data_gtx = torch.as_tensor(ref_data_gtx, dtype=torch.float32)
    
    ref_data_gux = np.load(ref_data_gux_path).astype(np.float32)
    ref_data_gty = np.load(ref_data_gty_path).astype(np.float32)
    ref_data_guy = np.load(ref_data_guy_path).astype(np.float32)  # Y GUIDANCE
    
    with np.load(stat_path_y) as stats_y:
        data_mean_y = float(stats_y['mean'])
        data_scale_y = float(stats_y['scale'])

    ref_data_gux = ref_data_gux[-4:, ...].copy().astype(np.float32)   
    ref_data_gux = torch.as_tensor(ref_data_gux, dtype=torch.float32)

    ref_data_gty = ref_data_gty[-4:, ...].copy().astype(np.float32)    
    ref_data_gty = torch.as_tensor(ref_data_gty, dtype=torch.float32)
    
    ref_data_guy = ref_data_guy[-4:, ...].copy().astype(np.float32)
    ref_data_guy = torch.as_tensor(ref_data_guy, dtype=torch.float32)

    
    flattened_ref_data_gux = []
    flattened_ref_data_gtx = []
    flattened_ref_data_gty = []
    flattened_ref_data_guy = []
    
    for i in range(ref_data_gtx.shape[0]):
        
        for j in range(ref_data_gtx.shape[1] - 2):
            
            flattened_ref_data_gtx.append(ref_data_gtx[i, j:j + 3, ...])
            flattened_ref_data_gux.append(ref_data_gux[i, j:j + 3, ...])
            flattened_ref_data_gty.append(ref_data_gty[i, j:j + 3, ...])
            flattened_ref_data_guy.append(ref_data_guy[i, j:j + 3, ...])    
                
    flattened_ref_data_gtx = torch.stack(flattened_ref_data_gtx, dim=0)
    flattened_ref_data_gux = torch.stack(flattened_ref_data_gux, dim=0)
    flattened_ref_data_gty = torch.stack(flattened_ref_data_gty, dim=0)
    flattened_ref_data_guy = torch.stack(flattened_ref_data_guy, dim=0)
    
    return (
        flattened_ref_data_gtx,
        flattened_ref_data_gux,
        flattened_ref_data_gty,
        flattened_ref_data_guy,
        data_mean,
        data_scale,
        data_mean_y,
        data_scale_y,
    )

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

        ckpt = torch.load(self.config.model.ckpt_path, map_location=self.device)
        model_state_dict = _extract_model_state_dict(
            ckpt,
            prefer_ema=getattr(self.config.model, "ema", False),
        )
        model.load_state_dict(model_state_dict)

        model.to(self.device)

        self.log("Model loaded")

        model.eval()
        self.log('Preparing data')
        ref_data, blur_data, ref_data3, blur_data_y, data_mean, data_std, data_mean_y, data_std_y = load_recons_data(
            self.config.data.data_dir_gtx,
            self.config.data.data_dir_gux,
            self.config.data.data_dir_gty,
            self.config.data.data_dir_guy,
            self.config.data.stat_path,
            self.config.data.stat_pathy,
            smoothing=self.config.data.smoothing,
            smoothing_scale=self.config.data.smoothing_scale,
        )
        
        scaler = StdScaler(data_mean, data_std)
        y_scaler = StdScaler(data_mean_y, data_std_y)
        x_offset, x_scale = _converted_to_physical_linear_params(
            data_mean,
            data_std,
            physical_min=-7.0,
            physical_max=17.0,
            pix_max=246.0,
            pix_min=0.0,
        )
        y_offset, y_scale = _converted_to_physical_linear_params(
            data_mean_y,
            data_std_y,
            physical_min=-5.0,
            physical_max=9.0,
            pix_max=254.0,
            pix_min=0.0,
        )
        num_workers = _resolve_num_workers(self.config.data.num_workers)

        self.log("Start sampling")

        testset = TensorDataset(ref_data, blur_data, ref_data3, blur_data_y)
        
        test_loader = DataLoader(testset,
                     batch_size=self.config.sampling.batch_size,
                     shuffle=False, num_workers=num_workers)
        
        for batch_index,(data, blur_data, ref_data3, blur_data_y) in enumerate(test_loader):
            print(batch_index)
            self.log('Batch: {} / Total batch {}'.format(batch_index, len(test_loader)))
            
            x0 = blur_data.to(self.device)
            y0 = blur_data_y.to(self.device)  # Using guidance Y
            gt = data.to(self.device)

            x0_z = scaler(x0)
            y0_z = y_scaler(y0)
            x0_state = torch.cat([x0_z, y0_z], dim=1)


            self.log('Preparing reference image')
            self.log('Dumping visualization...')

            sample_folder = 'sample_batch{}'.format(batch_index)
            ensure_dir(os.path.join(self.image_sample_dir, sample_folder))

            model_in_ch = _model_in_channels(model)
            # Model expects concatenated [x, y] channels: 6 total (3 for X + 3 for Y)
            expected_channels = 6
            if model_in_ch != expected_channels:
                raise ValueError(
                    f"This runner expects model.in_channels == {expected_channels} (3 X + 3 Y), "
                    f"but got {model_in_ch}."
                )

            # Build the safety mask in the same physical space used by the
            # divergence residual. Using raw pixel values here makes the threshold
            # almost always active and washes out the real wall boundary.
            y0_phys = _pixel_to_physical(y0, 0.0, 254.0, -5.0, 9.0)
            gt_phys = _pixel_to_physical(gt, 0.0, 246.0, -7.0, 17.0)
            safe_mask = get_safe_fluid_mask(gt_phys, y0_phys, threshold=1e-3).to(self.device)

            gt_residual_loss, _ = velocity_residual(gt_phys, y0_phys, safe_mask, calc_grad=True)
            gt_residual_loss = gt_residual_loss.detach()
            self.log('Residual reference: {}'.format(gt_residual_loss.item()))
            init_phys = _pixel_to_physical(x0, 0.0, 246.0, -7.0, 17.0)
            init_residual_loss, _ = velocity_residual(init_phys, y0_phys, safe_mask, calc_grad=True)
            init_residual_loss = init_residual_loss.detach()
            self.log('Residual init: {}'.format(init_residual_loss.item()))
            
            x0 = x0_z.clone()
            check_valid_image(x0, "Scaled x0")

            xinit = x0_state.clone()
            
            # prepare optional loss logger
            logger = None
            if self.config.sampling.log_loss:
                l2_loss_fn = lambda x: l2_loss(scaler.inverse(x[:, :3]).to(gt.device), gt)

                equation_loss_fn = lambda x: velocity_residual(
                    x[:, :3] * x_scale + x_offset,
                    y0_phys,
                    safe_mask,
                    calc_grad=True,
                )[0]  

                logger = MetricLogger({
                    'l2 loss': l2_loss_fn,
                    'residual loss': equation_loss_fn
                })

            for repeat in range(self.args.repeat_run):
                self.log(f'=== Run No.{repeat} ===')

                if repeat == 0:
                    x0 = xinit.clone()
                else:
                    self.log(f'Refining the result from repeat {repeat - 1}.')

                for it in range(self.args.sample_step):  
                    if it == 0:

                        self.log(f'--- Iteration {it} of Run No.{repeat} ---')
                        e = torch.randn_like(x0)
                        noise_decay = 0.5 ** repeat
                        total_noise_levels = max(
                            1,
                            int(self.args.t * noise_decay * (1 ** it)),
                        )
                        self.log(
                            f'Repeat {repeat}: using {total_noise_levels} noise levels.'
                        )
                        a = (1 - self.betas).cumprod(dim=0)
                        x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                        def physical_gradient_func(x_state):
                            x_u = x_state[:, :3]
                            x_v = x_state[:, 3:6]
                            dx_raw = velocity_residual(
                                x_u * x_scale + x_offset,
                                x_v * y_scale + y_offset,
                                safe_mask,
                                calc_grad=True,
                            )[1]
                            dx_scaled = torch.cat([
                                dx_raw[:, 0:3] / x_scale,
                                dx_raw[:, 3:6] / y_scale,
                            ], dim=1)
                            return dx_raw, dx_scaled

                        use_physical_guidance = getattr(
                            self.config.sampling, 'use_physical_guidance', True
                        )
                        if self.config.model.type == 'conditional' and use_physical_guidance:
                            self.log('Using conditional model with vorticity residual gradient guidance.')

                        num_of_reverse_steps = int(self.args.reverse_steps * (1 ** it))
                        betas = self.betas.to(self.device)
                        skip = total_noise_levels // num_of_reverse_steps
                        seq = range(0, total_noise_levels, skip)
                  
                        if self.config.model.type == 'conditional' and use_physical_guidance:
                            self.log('Performing guided DDIM steps with conditional model...')
                            # Calculate dx (gradients) for this initial state
                            dx = velocity_residual(
                                x[:, :3] * x_scale + x_offset,
                                x[:, 3:6] * y_scale + y_offset,
                                safe_mask,
                                calc_grad=True,
                            )[1]
                            xs, _ = guided_ddim_steps(x, seq, model, betas,
                                                    w=self.config.sampling.guidance_weight,
                                                    w_min=getattr(self.config.sampling, 'guidance_weight_min', self.config.sampling.guidance_weight),
                                                    dx_scale=getattr(self.config.sampling, 'dx_scale', 1.0),
                                                    dx_func=physical_gradient_func,
                                                    dx=dx,
                                                    input_fn=lambda state: state,
                                                    cache=False, logger=logger)
                        elif self.config.model.type == 'conditional':
                            
                            self.log('Performing standard DDIM steps without physical guidance...')
                            xs, _ = ddim_steps(x, seq, model, betas, cache=False, logger=logger)
                        elif self.config.sampling.lambda_ > 0:
                            self.log('Performing guided DDIM steps with lambda > 0...')
                            xs, _ = ddim_steps(x, seq, model, betas,
                                            dx_func=physical_gradient_func, cache=True, logger=logger)
                        else:
                            self.log('Performing standard DDIM steps...')
                            xs, _ = ddim_steps(x, seq, model, betas, cache=True, logger=logger)

                        self.log(f'Sequence of images (xs) generated for iteration {it}. Total steps: {len(xs)}')

                        x = xs[-1]  # Get the final 6-channel image state
                        x0 = xs[-1].to(self.device)

                        self.log(f'Imaged saved as comparison_run_{repeat}_it{it}.png.')
                        # Optionally dump arrays
                        if self.config.sampling.dump_arr:
                            u_physical = x[:, :3] * x_scale + x_offset
                            v_physical = x[:, 3:6] * y_scale + y_offset
                            magnitude = torch.sqrt(u_physical.square() + v_physical.square())
                            output_dir = os.path.join(self.image_sample_dir, sample_folder)

                            np.save(
                                os.path.join(output_dir, f'sample_u_run_{repeat}_it{it}.npy'),
                                u_physical.cpu().numpy(),
                            )
                            np.save(
                                os.path.join(output_dir, f'sample_v_run_{repeat}_it{it}.npy'),
                                v_physical.cpu().numpy(),
                            )
                            np.save(
                                os.path.join(output_dir, f'sample_magnitude_run_{repeat}_it{it}.npy'),
                                magnitude.cpu().numpy(),
                            )

                        # Log losses if enabled
                        if self.config.sampling.log_loss and logger is not None:
                            logger.log(os.path.join(self.image_sample_dir, sample_folder), f'run_{repeat}_it{it}')
                            logger.reset()
                            print(f'Logged and reset the logger for iteration {it}.')
                    else:
                        print(f'TESTING THIS CODE !!!!')
                print(f'=== Finished Run No.{repeat} ===')
            self.log('Finished batch {}'.format(batch_index))
            self.log('========================================================')
        self.log('Finished sampling')
