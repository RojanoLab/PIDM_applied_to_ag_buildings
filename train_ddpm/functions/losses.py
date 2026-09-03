import torch
import torch.nn.functional as F

def get_safe_fluid_mask(x0: torch.Tensor, y0: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """
    Dynamically generates a fluid mask from ground truth velocity fields (x0, y0)
    and erodes fluid boundaries by 2 pixels to eliminate stencil overlap artifacts.
    
    Returns:
        safe_fluid_mask: [B, 1, H, W] tensor (1.0 for valid fluid interior, 0.0 for walls/boundaries)
    """
    # 1. Identify fluid cells from clean ground truth (where velocity > threshold)
    raw_fluid_mask = ((x0.abs() > threshold) | (y0.abs() > threshold)).float()
    
    # 2. Invert to create a wall mask (1.0 for wall, 0.0 for fluid)
    wall_mask = 1.0 - raw_fluid_mask
    
    # 3. Expand wall boundaries by 2 pixels using max_pool2d (kernel=5, stride=1, padding=2)
    #    This covers the 2-pixel reach of the 5-point finite difference stencil
    expanded_wall = F.max_pool2d(wall_mask, kernel_size=5, stride=1, padding=2)
    
    # 4. Safe fluid interior: 1.0 only where stencil reads pure fluid data
    safe_fluid_mask = 1.0 - expanded_wall
    return safe_fluid_mask


def compute_5point_divergence(u: torch.Tensor, v: torch.Tensor, dx: float = 0.0358203125, dy: float = 0.0358203125) -> torch.Tensor:
    """
    Computes spatial divergence (du/dx + dv/dy) using a 4th-order 5-point stencil.
    Expects 1-channel u and v tensors: [B, 1, H, W].
    """
    in_channels = u.shape[1]

    # 4th-order finite difference kernel
    kernel_1d = torch.tensor([1/12, -8/12, 0.0, 8/12, -1/12], dtype=torch.float32, device=u.device)
    kernel_x = kernel_1d.view(1, 1, 1, 5).repeat(in_channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, 5, 1).repeat(in_channels, 1, 1, 1)

    u_padded_x = F.pad(u, (2, 2, 0, 0), mode='replicate')
    v_padded_y = F.pad(v, (0, 0, 2, 2), mode='replicate')

    du_dx = F.conv2d(u_padded_x, kernel_x, groups=in_channels) / dx
    dv_dy = F.conv2d(v_padded_y, kernel_y, groups=in_channels) / dy

    return du_dx + dv_dy

def velocity_residual(w: torch.Tensor, 
                      w2: torch.Tensor, 
                      safe_mask: torch.Tensor, 
                      dx_spacing: float = 0.0358203125, 
                      dy_spacing: float = 0.0358203125, 
                      calc_grad: bool = False):

    """
    Computes divergence loss restricted strictly to safe fluid regions
    and extracts 2-channel steering gradients [grad_u, grad_v].
    """
    # Detach first to break computation graph, then clone and set requires_grad
    w = w.detach().clone().requires_grad_(True)
    w2 = w2.detach().clone().requires_grad_(True)

    # 1. Compute raw 5-point spatial divergence
    vel_res = compute_5point_divergence(w, w2, dx=dx_spacing, dy=dy_spacing)

    # 2. ZERO OUT boundary stencil spikes and wall cells
    masked_res = vel_res * safe_mask

    # 3. Calculate physical mean loss on clean interior fluid pixels
    residual_loss = torch.nan_to_num((masked_res**2).mean(), nan=1e6, posinf=1e6, neginf=1e6)

    # 4. Compute steering sensitivity gradients for BOTH components
    grads = torch.autograd.grad(residual_loss, [w, w2], create_graph=False, retain_graph=False)
    wr_u, wr_v = grads[0], grads[1]

    # 5. Concatenate into a 2-channel steering guidance tensor [B, 2, H, W]
    dx = torch.cat([wr_u, wr_v], dim=1)
    
    if calc_grad:
        return residual_loss, dx
    
    return dx
    

def conditional_noise_estimation_loss(model,
                                       x0: torch.Tensor,        # [B, 3, H, W] Clean u
                                       y0: torch.Tensor,        # [B, 3, H, W] Clean v
                                       t: torch.LongTensor,     # [B] Timesteps
                                       e: torch.Tensor,         # [B, 6, H, W] Independent 2-channel noise
                                       b: torch.Tensor,         # Noise schedule beta
                                       x_scale, x_offset,
                                       y_scale, y_offset,
                                       keepdim=False):

    """
    Physically-guided 2-channel vector diffusion loss function with dynamic wall masking.
    """
    # 1. Compute dynamic fluid mask eroded by 2 pixels for current batch
    safe_mask = get_safe_fluid_mask(x0, y0)

    # 2. Diffusion forward setup with independent noise components
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    e_x = e[:, 0:3]  
    e_y = e[:, 3:6]  

    x = x0 * a.sqrt() + e_x * (1.0 - a).sqrt()
    y = y0 * a.sqrt() + e_y * (1.0 - a).sqrt()

    # 3. Unscale to physical velocity units (m/s)
    u_phys = (x * x_scale + x_offset) 
    v_phys = (y * y_scale + y_offset) 

    # 4. Compute 2-channel steering guidance dx using the safe fluid mask
    dx = velocity_residual(u_phys, v_phys, safe_mask)  # Shape: [B, 6, H, W]

    # 5. Concatenate noisy states into 2-channel input
    vel_input = torch.cat([x, y], dim=1)  # Shape: [B, 6, H, W]

    # 6. Forward pass through model predicting 2-channel noise
    output = model(vel_input, t.float(), dx)

    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

    # 7. Compute loss against target 2-channel noise tensor e
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          y0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          keepdim=False):
    

    """Unconditioned 2-channel noise estimation loss."""
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    e_x = e[:, 0:1]
    e_y = e[:, 1:2] 

    x = x0 * a.sqrt() + e_x * (1.0 - a).sqrt()
    y = y0 * a.sqrt() + e_y * (1.0 - a).sqrt()

    vel_input = torch.cat([x, y], dim=1)
    
    output = model(vel_input, t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'conditional': conditional_noise_estimation_loss
}
