
import torch
import numpy as np
import os
import pandas as pd
from torch.optim.sgd import SGD

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _KE_EPS_PATH = r"C:\Users\Rojano\Documents\diffusion_final\data\prediction_k_eps_mean.npz"
_KE_EPS_CACHE= {}


def _get_ke_init_tensors(target_device, ke_path=None):
    source_path = ke_path 
    cache_key = f"{str(target_device)}::{source_path}"
    if cache_key not in _KE_EPS_CACHE:
        with np.load(source_path) as data_npz:
            k_load = data_npz["k"]
            eps_load = data_npz["epsilon"]

        k_init = torch.from_numpy(k_load).to(target_device, dtype=torch.float32)
        eps_init = torch.from_numpy(eps_load).to(target_device, dtype=torch.float32)
        k_init = torch.clamp(k_init, min=1e-8)
        eps_init = torch.clamp(eps_init, min=1e-8)
        _KE_EPS_CACHE[cache_key] = (k_init, eps_init)
    return _KE_EPS_CACHE[cache_key]

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
    # Clamp exponent inputs to keep values in a numerically stable range.

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

def normalize_to_01(image_array):
    """Scales a NumPy array to the range [0, 1]."""
    min_val = -7
    max_val = 17
    
    if max_val - min_val > 0:
        return (image_array - min_val) / 24
    else:
        return torch.zeros(image_array.shape)


def main_code(w, w2, ke_path=None):
    w = w.clone()
    w.requires_grad_(True)
    w2 = w2.clone()
    w2.requires_grad_(True)

    # Physics Parameters
    params = {"dx": 0.071640625, "nu": 1.46e-5, "rho": 1.225, "mu": 1.78e-5}
    OPTIM_STEPS = 100  
    # EARLY_STOP_PATIENCE = 40
    # EARLY_STOP_MIN_DELTA = 1e-8
    OUTPUT_CSV = "sequence_statistics_optimized3.csv"
    if not os.path.exists(OUTPUT_CSV):
        df_init = pd.DataFrame(columns=[
            "Alpha", "RMSE_Momentum"
        ])
        df_init.to_csv(OUTPUT_CSV, index=False)

    u_phys = convert_to_physical(w, -7.0, 17.0, 246.0, 0)
    v_phys = convert_to_physical(w2, -5.0, 9.0, 244.0, 0)

    # Use detached tensors inside the inner optimizer loop to avoid
    # backpropagating through the outer training graph multiple times.
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

    # Load once per process and reuse for all subsequent calls.
    k, eps = _get_ke_init_tensors(device, ke_path=ke_path)
    # Convert arrays to tensors for JIT function inputs.
    # Initialize Learnable Parameters
    alpha = torch.tensor([-0.8569], device=device, requires_grad=True)

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

    # 6. Extract Final Results
    # Recompute residual on the original graph once, using optimized coefficients.
    res_acc_mom = compute_residuals_jit(
        k.detach(), eps.detach(), alpha.detach(), u_phys, v_phys, S2_static,
        dp_dx_static, adv_term_u_static, dx, dy, rho, params["nu"]
    )
    dt_char = params["dx"] / (torch.max(torch.sqrt(u_phys**2 + v_phys**2)) + 1e-10)
    vel_res = res_acc_mom * dt_char
    rmse = torch.sqrt(torch.mean(vel_res**2)).item()
    # 7. Save Stats
    alpha_value = (4.0 * torch.sigmoid(alpha.detach()) - 2.0)[0].item()
    stats = {
            "Alpha": alpha_value,
        "RMSE_Momentum": rmse,
    }
    
    df_new = pd.DataFrame([stats])
    df_new.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

    vel_res_u = normalize_to_01(torch.nan_to_num(vel_res, nan=0.0, posinf=1e6, neginf=-1e6)) 
    residual_loss = torch.nan_to_num((vel_res_u**2).mean(), nan=1e6, posinf=1e6, neginf=1e6)
    wr = torch.autograd.grad(residual_loss, w)[0]
    return wr


def velocity_residual(w, w2, ke_path=None):
    
    dw = main_code(w, w2, ke_path=ke_path)

    return dw


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                         keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def conditional_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          y0: torch.Tensor,                                                    
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          x_scale,
                          x_offset,
                          y_scale,
                          y_offset,
                          keepdim=False, p=0.1, ke_path=None):

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    y = y0 * a.sqrt() + e * (1.0 - a).sqrt()     
                            
        
    dx = velocity_residual(
        x * x_scale + x_offset / x_scale,
        y * y_scale + y_offset / y_scale,
        ke_path=ke_path,
    )

    # dx = velocity_residual(x, y
    #     ) 
    
                 #-----------------------
    # output = model(x, t.float(), dx)
    output = model(x, t.float(), dx)
    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

   

    # exit() 

    # print(f"output shape: {output.shape}")
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        r=(e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        # print(f"r shape: {r}")
        return r


loss_registry = {
    'simple': noise_estimation_loss,
    'conditional': conditional_noise_estimation_loss
}
