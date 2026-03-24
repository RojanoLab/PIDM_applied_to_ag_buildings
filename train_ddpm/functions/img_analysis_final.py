import cv2
import torch
import matplotlib.pyplot as plt
import math
import csv
import os
from datetime import datetime

# --- NEW: Set up device for PyTorch (GPU or CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

### --- Helper functions converted to PyTorch --- ###
def scale_to_range(image_array, new_min, new_max):
    """Scales a NumPy array to a new specified min-max range."""
    old_min = torch.min(image_array)
    old_max = torch.max(image_array)
    
    # Avoid division by zero
    if old_max - old_min == 0:
        return torch.full(image_array.shape, new_min, dtype=torch.float32)
        
    # Standard normalization to [0, 1]
    normalized = (image_array - old_min) / (old_max - old_min)
    
    # Scale to new range
    scaled_array = normalized * (new_max - new_min) + new_min
    return scaled_array

### NEW: Function to normalize any array to a [0, 1] scale
def normalize_to_01(image_array):
    """Scales a NumPy array to the range [0, 1]."""
    min_val = torch.min(image_array)
    max_val = torch.max(image_array)
    
    if max_val - min_val > 0:
        return (image_array - min_val) / (max_val - min_val)
    else:
        return torch.zeros(image_array.shape)
    
def convert_to_physical(grayscale_img, physical_min, physical_max):
    """Maps a 0-255 grayscale image (as a tensor) to its physical value range."""
    img_float = grayscale_img.to(dtype=torch.float32, device=device)
    if torch.max(grayscale_img) - torch.min(grayscale_img) == 0:
        return torch.full_like(grayscale_img, physical_min, dtype=torch.float32, device=device)
    normalized_img = (img_float - torch.min(grayscale_img)) / (torch.max(grayscale_img) - torch.min(grayscale_img))
    return (normalized_img * (physical_max - physical_min) + physical_min)

def estimate_k_epsilon_proxies(u_physical, v_physical, turbulence_intensity, length_scale):
    """Estimates k and epsilon fields from mean velocity data using PyTorch."""
    C_mu = 0.09
    velocity_magnitude = torch.sqrt(u_physical**2 + v_physical**2) + 1e-10
    k_proxy = 1.5 * (velocity_magnitude * turbulence_intensity)**2
    epsilon_proxy = (C_mu**0.75) * (k_proxy**1.5) / length_scale
    return k_proxy, epsilon_proxy

def solve_pressure_poisson(u, v, dx, rho, dt, iterations=50):
    """Solves the Pressure Poisson Equation using PyTorch."""
    u_E = torch.roll(u, shifts=-1, dims=1); u_W = torch.roll(u, shifts=1, dims=1)
    v_N = torch.roll(v, shifts=-1, dims=0); v_S = torch.roll(v, shifts=1, dims=0)
    rhs = rho * ((u_E - u_W) / (2 * dx) + (v_N - v_S) / (2 * dx))
    p = torch.zeros_like(u, device=device)
    for _ in range(iterations):
        p_old = p.clone()
        p_E = torch.roll(p_old, shifts=-1, dims=1); p_W = torch.roll(p_old, shifts=1, dims=1)
        p_N = torch.roll(p_old, shifts=-1, dims=0); p_S = torch.roll(p_old, shifts=1, dims=0)
        p = 0.25 * (p_E + p_W + p_N + p_S - (dx**2) * rhs)
        p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
    return p * dt

def calculate_f_vm_spatial_terms(u, v, k_phys, epsilon_phys, dx, nu, dt):
    """Calculates advection and diffusion terms using FVM with PyTorch."""
    # Advective Flux
    u_E = torch.roll(u, shifts=-1, dims=1); u_W = torch.roll(u, shifts=1, dims=1)
    u_N = torch.roll(u, shifts=-1, dims=0); u_S = torch.roll(u, shifts=1, dims=0)
    v_N = torch.roll(v, shifts=-1, dims=0); v_S = torch.roll(v, shifts=1, dims=0)
    u_face_E = (u + u_E) / 2; u_face_W = (u + u_W) / 2
    v_face_N = (v + v_N) / 2; v_face_S = (v + v_S) / 2
    F_adv_E = 0.5 * ((u_face_E > 0) * u**2 + (u_face_E <= 0) * u_E**2)
    F_adv_W = 0.5 * ((u_face_W > 0) * u_W**2 + (u_face_W <= 0) * u**2)
    F_adv_N = ((v_face_N > 0) * u * v + (v_face_N <= 0) * u_N * v_N)
    F_adv_S = ((v_face_S > 0) * u_S * v_S + (v_face_S <= 0) * u * v)
    advection_term = (1/dx) * ((dt * F_adv_E - dt * F_adv_W) + (dt * F_adv_N - dt * F_adv_S))
    # Diffusive Flux
    C_mu = 0.09
    epsilon_safe = epsilon_phys + 1e-10
    turbulent_viscosity = C_mu * (k_phys**2) / epsilon_safe
    effective_viscosity = nu + turbulent_viscosity
    eff_visc_E = torch.roll(effective_viscosity, shifts=-1, dims=1)
    eff_visc_W = torch.roll(effective_viscosity, shifts=1, dims=1)
    eff_visc_N = torch.roll(effective_viscosity, shifts=-1, dims=0)
    eff_visc_S = torch.roll(effective_viscosity, shifts=1, dims=0)
    nu_face_E = (effective_viscosity + eff_visc_E) / 2
    nu_face_W = (effective_viscosity + eff_visc_W) / 2
    nu_face_N = (effective_viscosity + eff_visc_N) / 2
    nu_face_S = (effective_viscosity + eff_visc_S) / 2
    F_diff_E = -nu_face_E * (u_E - u) / dx
    F_diff_W = -nu_face_W * (u - u_W) / dx
    F_diff_N = -nu_face_N * (u_N - u) / dx
    F_diff_S = -nu_face_S * (u - u_S) / dx
    diffusion_term = (1/dx) * ((dt * F_diff_E - dt * F_diff_W) + (dt * F_diff_N - dt * F_diff_S))
    return advection_term, diffusion_term

def torch_gradient(tensor, dx):
    """Computes the gradient of a 2D tensor along axis 1 (dx) using central differences."""
    grad_center = (torch.roll(tensor, shifts=-1, dims=1) - torch.roll(tensor, shifts=1, dims=1)) / (2 * dx)
    grad_left = (torch.roll(tensor, shifts=-1, dims=1) - tensor) / dx
    grad_right = (tensor - torch.roll(tensor, shifts=1, dims=1)) / dx
    grad_center[:, 0] = grad_left[:, 0]
    grad_center[:, -1] = grad_right[:, -1]
    return grad_center

### Objective function ###
def calculate_prediction_error(coeffs, u, v, u_t1, I, L, dx, nu, rho, dt):
    """This function takes coefficients and returns a single error value (RMSE)."""
    alpha_k_coeff, beta_epsilon_coeff, gamma_pressure_coeff, delta_advection_coeff = coeffs
    
    k_phys, epsilon_phys = estimate_k_epsilon_proxies(u, v, I, L)
    k_phys_modified = alpha_k_coeff * k_phys
    epsilon_phys_modified = beta_epsilon_coeff * epsilon_phys
    
    advection_fvm, diffusion_fvm = calculate_f_vm_spatial_terms(u, v, k_phys_modified, epsilon_phys_modified, dx, nu, dt)
    advection_fvm = delta_advection_coeff * advection_fvm
    
    pressure_phys = solve_pressure_poisson(u, v, dx, rho, dt, iterations=50)
    dp_dx = torch_gradient(pressure_phys, dx)
    
    pressure_gradient_term = gamma_pressure_coeff * (-(1/rho) * dp_dx)
    
    net_acceleration = -advection_fvm + pressure_gradient_term + diffusion_fvm
    actual_velocity_change = (u_t1 - u)
    predicted_velocity_change = net_acceleration
    
    error_field = actual_velocity_change - predicted_velocity_change
    rmse = torch.sqrt(torch.mean(error_field**2))
    return rmse

### Plotting function ###
# def plot_final_results(actual_change, predicted_change, pressure_term, diffusion_term, final_error):
#     """Generates plots from tensors by first converting them to numpy arrays."""
#     # Convert all tensors to CPU-based numpy arrays for plotting
#     actual_change_np = actual_change.detach().cpu().numpy()
#     predicted_change_np = predicted_change.detach().cpu().numpy()
#     pressure_term_np = pressure_term.detach().cpu().numpy()
#     diffusion_term_np = diffusion_term.detach().cpu().numpy()
#     final_error_np = final_error.detach().cpu().numpy()

#     plt.figure(figsize=(30, 6)); cmap = 'viridis'
    
#     ax1 = plt.subplot(1, 5, 1); im1 = ax1.imshow(actual_change_np, cmap=cmap)
#     ax1.set_title('Actual Velocity Change (m/s)'); plt.colorbar(im1, ax=ax1)
    
#     ax2 = plt.subplot(1, 5, 2); im2 = ax2.imshow(predicted_change_np, cmap=cmap)
#     ax2.set_title('Predicted Velocity Change (m/s)'); plt.colorbar(im2, ax=ax2)
    
#     ax3 = plt.subplot(1, 5, 3); im3 = ax3.imshow(pressure_term_np, cmap=cmap)
#     ax3.set_title('Pressure Gradient Term (m/s²)'); plt.colorbar(im3, ax=ax3)
    
#     ax4 = plt.subplot(1, 5, 4); im4 = ax4.imshow(diffusion_term_np, cmap=cmap)
#     ax4.set_title('FVM Turbulent Diffusion (m/s²)'); plt.colorbar(im4, ax=ax4)
    
#     ax5 = plt.subplot(1, 5, 5); im5 = ax5.imshow(final_error_np, cmap=cmap)
#     ax5.set_title('Final Error (m/s)'); plt.colorbar(im5, ax=ax5)
    
#     plt.tight_layout(pad=2.0); plt.show()
    
#     error_values = final_error_np.flatten()
#     plt.figure(figsize=(12, 7))
#     plt.hist(error_values, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
#     plt.title('Distribution of Final Prediction Error')
#     plt.xlabel('Error (Actual - Predicted) [m/s]'); plt.ylabel('Frequency (Number of Pixels)')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     mean_val = error_values.mean(); std_val = error_values.std()
#     plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_val:.4f} m/s')
#     plt.axvline(mean_val + std_val, color='purple', linestyle=':', linewidth=2, label=f'Std Dev: {std_val:.4f} m/s')
#     plt.axvline(mean_val - std_val, color='purple', linestyle=':', linewidth=2)
#     plt.legend(); plt.show()


### Main execution block ###
if __name__ == '__main__':
    # --- Step 1: Load data and convert to PyTorch Tensors ---
    image_paths = [
        r'C:\Users\Rojano\Downloads\seq_sim_4x\seq4x\right_anim_s4-1_0071_0070.jpg',
        r'C:\Users\Rojano\Downloads\seq_sim_4x\seq4x\right_anim_s4-1_0072_0071.jpg',
        r'C:\Users\Rojano\Downloads\seq_sim_4y\seq4y\right_anim_s4-2_0071.jpg',
    ]
    try:
        img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(image_paths[2], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None or img3 is None:
            raise FileNotFoundError("One or more images not found.")
    except Exception as e:
        print(f"Error loading images: {e}. Please check file paths.")
        exit()

    # Convert images to tensors and move to the selected device
    img1_tensor = torch.from_numpy(img1).to(device)
    img2_tensor = torch.from_numpy(img2).to(device)
    img3_tensor = torch.from_numpy(img3).to(device)

    u_t0_phys = convert_to_physical(img1_tensor, -7.0, 17.0)
    v_t0_phys = convert_to_physical(img3_tensor, -5.0, 9.0)
    u_t1_phys = convert_to_physical(img2_tensor, -7.0, 17.0)
    
    # --- Step 2: Define fixed parameters ---
    params = {
        "u": u_t0_phys, "v": v_t0_phys, "u_t1": u_t1_phys,
        "I": 0.075, "L": 0.07 * 2.62 * 7, "dx": 0.1431796875,
        "nu": 0.00001460734, "rho": 1.225, "dt": 600
    }
    
    # --- Step 3: Define optimization settings ---
    # Define the initial guess for the coefficients
    initial_guess = [1e-10, 1.0, 1e-10, 1e-10]
    # Define bounds for clamping
    bounds_min = torch.tensor([1e-10, 1e-10, 1e-10, 1e-10], device=device)
    bounds_max = torch.tensor([10.0, 10.0, 10.0, 10.0], device=device)
    
    # Adam Hyperparameters
    LEARNING_RATE = 0.1
    ITERATIONS = 40

    # --- Step 4: Run the PyTorch Optimizer ---
    # Create a tensor for the coefficients that requires a gradient
    coeffs = torch.tensor(initial_guess, dtype=torch.float32, device=device, requires_grad=True)
    
    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([coeffs], lr=LEARNING_RATE)
    
    # print(f"\n--- Starting PyTorch Adam Optimization with initial guess: {initial_guess} ---")
    for t in range(1, ITERATIONS + 1):
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Calculate the loss (error)
        loss = calculate_prediction_error(coeffs, **params)
        
        # Automatically compute gradients
        loss.backward()
        
        # Update the coefficients
        optimizer.step()
        
        # Manually clamp the coefficients to stay within bounds
        with torch.no_grad():
            coeffs.clamp_(bounds_min, bounds_max)
            
        # if t % 10 == 0 or t == 1:
        #     print(f"Iteration {t}/{ITERATIONS}, Loss (RMSE): {loss.item():.6f}")

    # print("\n--- Optimization Complete ---")
    
    # --- Step 5: Display the best result ---
    best_coeffs = coeffs.detach() # Final coefficients are the best ones
    final_rmse = calculate_prediction_error(best_coeffs, **params).item()
    
    # print("\n--- Best Result Found ---")
    # print(f"Optimal Alpha (for k):        {best_coeffs[0]:.8f}")
    # print(f"Optimal Beta (for epsilon):   {best_coeffs[1]:.8f}")
    # print(f"Optimal Gamma (for pressure): {best_coeffs[2]:.8f}")
    # print(f"Optimal Delta (for advection):{best_coeffs[3]:.8f}")
    # print(f"Lowest RMSE achieved:         {final_rmse:.6f}")
     # Prepare the data for the new row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coeffs_list = best_coeffs.cpu().tolist()
    data_row = [timestamp] + coeffs_list + [final_rmse]
    
    file_path = "optimal_coeffs.txt"
    file_exists = os.path.exists(file_path)

    # Open the file in append mode ("a") to add a new line
    # newline='' is important to prevent blank rows when using the csv module
    with open(file_path, "a", newline='') as f:
        writer = csv.writer(f)
        
        # If the file is new, write the header first
        if not file_exists:
            header = ["datetime", "alpha_k", "beta_epsilon", "gamma_pressure", "delta_advection", "rmse"]
            writer.writerow(header)
        
        # Write the new data row
        writer.writerow(data_row)
            
    # print(f"\n✅ Results data row appended to {file_path}")
    # --- Re-run with optimal coefficients and plot final images ---
    # print("\n--- Generating final images with optimal coefficients ---")
    best_alpha, best_beta, best_gamma, best_delta = best_coeffs
    k_phys, epsilon_phys = estimate_k_epsilon_proxies(params['u'], params['v'], params['I'], params['L'])
    k_phys_mod = best_alpha * k_phys
    epsilon_phys_mod = best_beta * epsilon_phys
    
    advection, diffusion = calculate_f_vm_spatial_terms(
        params['u'], params['v'], k_phys_mod, epsilon_phys_mod, params['dx'], params['nu'], params['dt']
    )
    advection = best_delta * advection
    
    pressure = solve_pressure_poisson(params['u'], params['v'], params['dx'], params['rho'], params['dt'], iterations=50)
    dpdx = torch_gradient(pressure, params['dx'])
    pressure_grad = best_gamma * (-(1/params['rho']) * dpdx)
    
    predicted_accel = -advection + pressure_grad + diffusion
    actual_accel = (params['u_t1'] - params['u'])
    final_error_image = actual_accel - predicted_accel
    error_scaled_to_range = scale_to_range(final_error_image, -7.0, 17.0)
    error_final_normalized = normalize_to_01(error_scaled_to_range)
        
    # plot_final_results(actual_accel, predicted_accel, pressure_grad, diffusion, error_final_normalized)
    # plot_final_results(actual_accel, predicted_accel, pressure_grad, diffusion, final_error_image)
