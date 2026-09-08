#Code that print the generated images (read the files fot the U and V bias corrections and add upscaling)
import os
import re
import csv
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker


#global variables 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
COLORBAR_LABELSIZE = 80

# =====================================================================
# --- EDIT THESE VALUES ACCORDING TO YOUR RUN ---
# =====================================================================
LOG_DIR = "experiments/rectangular_images/guided_recons__t240_r30_w3.0"

REF_PATH_GTX = "data/test_Vel_X.npy"
REF_DATA_GUX_PATH = "data/test_Vel_X_guided.npy"
REF_DATA_GTY_PATH = "data/test_Vel_Y.npy"
REF_DATA_GUY_PATH = "data/test_Vel_Y_guided.npy"

BATCH_SIZE = 3
REPEAT = 0
IT = 0

FRAME_INDICES = []

HISTOGRAM_FRAMES = [10, 20, 30]

SAVE_IMAGES_FRAMES = [10, 20, 30]

APPLY_BIAS_CORRECTION_U = True
BIAS_CORRECTION_U_PATH = "runners/bias_U_quantile.npz"


APPLY_BIAS_CORRECTION_V = True
BIAS_CORRECTION_V_PATH = "runners/bias_V_quantile.npz"

OUTPUT_DIR = "./visualizacion_output"
AUTOSCALE_MARGIN = 0.30

PIXEL_MIN_U, PIXEL_MAX_U = 0.0, 246.0
PHYSICAL_MIN_U, PHYSICAL_MAX_U = -7.0, 17.0
PIXEL_MIN_V, PIXEL_MAX_V = 0.0, 254.0
PHYSICAL_MIN_V, PHYSICAL_MAX_V = -5.0, 9.0

POLYGON_COORDS = [(216, 508), (216, 462), (302, 442), (386, 462), (386, 508)]

# --- U_ref per frame (141 fixed values, from the original file) ---
UREF_DATA = np.array([
    1.0209, 1.0836, 1.1874, 1.3317, 1.5153, 1.7369, 1.9947, 2.2868,
    2.6109, 2.9647, 3.3453, 3.7500, 4.1756, 4.6189, 5.0765, 5.5449,
    6.0206, 6.5000, 6.9794, 7.4551, 7.9235, 8.3811, 8.8244, 9.2500,
    9.6547, 10.0353, 10.3891, 10.7132, 11.0053, 11.2631, 11.4847,
    11.6683, 11.8126, 11.9164, 11.9791, 12.0000, 11.9791, 11.9164,
    11.8126, 11.6683, 11.4847, 11.2631, 11.0053, 10.7132, 10.3891,
    10.0353, 9.6547, 9.2500, 8.8244, 8.3811, 7.9235, 7.4551, 6.9794,
    6.5000, 6.0206, 5.5449, 5.0765, 4.6189, 4.1756, 3.7500, 3.3453,
    2.9647, 2.6109, 2.2868, 1.9947, 1.7369, 1.5153, 1.3317, 1.1874,
    1.0836, 1.0209, 1.0000, 1.0209, 1.0836, 1.1874, 1.3317, 1.5153,
    1.7369, 1.9947, 2.2868, 2.6109, 2.9647, 3.3453, 3.7500, 4.1756,
    4.6189, 5.0765, 5.5449, 6.0206, 6.5000, 6.9794, 7.4551, 7.9235,
    8.3811, 8.8244, 9.2500, 9.6547, 10.0353, 10.3891, 10.7132, 11.0053,
    11.2631, 11.4847, 11.6683, 11.8126, 11.9164, 11.9791, 12.0000,
    11.9791, 11.9164, 11.8126, 11.6683, 11.4847, 11.2631, 11.0053,
    10.7132, 10.3891, 10.0353, 9.6547, 9.2500, 8.8244, 8.3811, 7.9235,
    7.4551, 6.9794, 6.5000, 6.0206, 5.5449, 5.0765, 4.6189, 4.1756,
    3.7500, 3.3453, 2.9647, 2.6109, 2.2868, 1.9947, 1.7369, 1.5153,
    1.3317, 1.1874, 1.0836, 1.0209
])
# =====================================================================


def _nice_tick_step(value_range, target_ticks=9):
    """Calculates a 'round' tick step (1, 2, 2.5, 5, 10, ...) for ~target_ticks marks."""
    if value_range <= 0:
        return 1.0
    raw_step = value_range / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 2.5, 5, 10):
        step = m * magnitude
        if step >= raw_step:
            return step
    return 10 * magnitude


def _format_tick(value, step):
    if step >= 1:
        return f'{value:.0f}'
    elif step >= 0.1:
        return f'{value:.1f}'
    else:
        return f'{value:.2f}'


def _align_colorbar_ticks(cbar, fig, labelsize=55):
    for label in cbar.ax.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_x(1.15)


def convert_pixel_to_physical_range(pixel_arr, pixel_min, pixel_max, physical_min, physical_max):
    normalized = (pixel_arr - pixel_min) / (pixel_max - pixel_min)
    return normalized * (physical_max - physical_min) + physical_min


def load_and_flatten(path):
    data = np.load(path).astype(np.float32)
    data = data[-4:, ...].copy()
    flattened = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            flattened.append(data[i, j:j + 3, ...])
    return np.stack(flattened, axis=0)


def build_polygon_mask(img_shape, poly_coords):
    H, W = img_shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack((x.ravel(), y.ravel()), axis=-1)
    poly_path = Path(poly_coords)
    mask_np = poly_path.contains_points(coords).reshape(H, W)
    return mask_np


def save_cropped_polygon(image_2d, poly_coords, mask_roi, out_path, cmap='viridis', vmin=None, vmax=None, padding=5):
    H, W = image_2d.shape
    if vmin is None:
        vmin = np.nanmin(image_2d)
    if vmax is None:
        vmax = np.nanmax(image_2d)

    xs = [p[0] for p in poly_coords]
    ys = [p[1] for p in poly_coords]
    x_min = max(0, int(min(xs)) - padding)
    x_max = min(W, int(max(xs)) + padding)
    y_min = max(0, int(min(ys)) - padding)
    y_max = min(H, int(max(ys)) + padding)

    image_cropped = image_2d[y_min:y_max, x_min:x_max]
    crop_h, crop_w = image_cropped.shape
    poly_coords_shifted = [(px - x_min, py - y_min) for px, py in poly_coords]

    fig, ax = plt.subplots(figsize=(crop_w / 150, crop_h / 150), dpi=150)
    im = ax.imshow(image_cropped, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')

    polygon_patch = Polygon(poly_coords_shifted, closed=True, fill=False, edgecolor='none', lw=0)
    ax.add_patch(polygon_patch)
    im.set_clip_path(polygon_patch)

    ax.axis('off')
    ax.set_xlim(0, crop_w)
    ax.set_ylim(crop_h, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, transparent=True, dpi=150)
    plt.close(fig)


def save_colorbar_only(out_path, cmap='viridis', vmin=0.0, vmax=1.0, label="Value",
                        vmin_display=None, vmax_display=None, nbins=5, labelsize=COLORBAR_LABELSIZE):
    canvas_width = 2.6 + labelsize * 0.05
    canvas_height = 6 + labelsize * 0.05
    fig = plt.figure(figsize=(canvas_width, canvas_height))
    ax = fig.add_axes([0.25, 0.08, 0.2, 0.87])

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax)

    if vmin_display is not None and vmax_display is not None:
        cbar.ax.set_ylim(vmin_display, vmax_display)

    cbar.locator = ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    cbar.update_ticks()

    lower_limit = vmin_display if vmin_display is not None else vmin
    upper_limit = vmax_display if vmax_display is not None else vmax
    current_ticks = [t for t in cbar.get_ticks() if lower_limit <= t <= upper_limit]
    if len(current_ticks) > nbins:
        indices = np.linspace(0, len(current_ticks) - 1, nbins).round().astype(int)
        indices = sorted(set(indices))
        final_ticks = [current_ticks[i] for i in indices]
        cbar.set_ticks(final_ticks)

    cbar.ax.tick_params(labelsize=labelsize, width=2.5, length=7)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2.5)

    _align_colorbar_ticks(cbar, fig)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlapped_histograms(data1, data2, data3, label1, label2, label3, out_path, num_bins=60, x_max=2.0):
    """Overlapped histogram of 3 frames (A/B/C) -- identical design to the original file."""
    viridis = plt.cm.get_cmap('viridis')
    color1 = viridis(0.05)
    color2 = viridis(0.35)
    color3 = viridis(0.65)
    alpha = .4

    global_min = min(np.min(data1), np.min(data2), np.min(data3))
    global_max = max(np.max(data1), np.max(data2), np.max(data3))
    hist_bins = np.linspace(global_min, global_max, num_bins + 1)

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.hist(data1, bins=hist_bins, color=color1, alpha=alpha, label=label1, edgecolor='black', linewidth=1.5)
    ax.hist(data2, bins=hist_bins, color=color2, alpha=alpha, label=label2, edgecolor='black', linewidth=1.5)
    ax.hist(data3, bins=hist_bins, color=color3, alpha=alpha, label=label3, edgecolor='black', linewidth=1.5)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10, integer=True))
    ax.set_ylim(0, 2300)
    ax.set_ylabel('Frequency', fontsize=14)

    max_val = x_max
    step = _nice_tick_step(max_val, target_ticks=9)
    x_ticks = np.arange(0, max_val + step * 0.5, step)
    x_labels = []
    for tick in x_ticks:
        if tick == 0.0:
            x_labels.append('')
        else:
            x_labels.append(_format_tick(tick, step))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.set_xlabel('Absolute Error Range (m/s)', fontsize=14)
    ax.set_xlim(0, max_val)

    ax.tick_params(axis='both', which='major', labelsize=25, width=2.5, color='black')
    ax.legend(fontsize=25, frameon=False)
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)

def make_scatter_figure(x_data, y_data, uref_colors, marker, title, out_path):
    """Scatter RMSE_init vs RMSE_final, colored by U_ref -- identical design to the original."""
    my_font = "Times New Roman"
    number_size = 35
    norm = plt.Normalize(vmin=0, vmax=12)
    major_ticks = np.arange(0, 12 + 1, 2)
    minor_ticks = np.arange(1, 12, 2)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_data, y_data, c=uref_colors, cmap='viridis', alpha=0.8, norm=norm, marker=marker)

    max_rmse = max(max(x_data, default=0), max(y_data, default=0))
    step = _nice_tick_step(max_rmse, target_ticks=6)
    limit = math.ceil(max_rmse / step) * step if max_rmse > 0 else 2.0

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ticks = np.arange(0, limit + step * 0.5, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    xlabels = [_format_tick(t, step) for t in ticks]
    ylabels = [_format_tick(t, step) if t != 0 else '' for t in ticks]
    try:
        ax.set_xticklabels(xlabels, fontname=my_font, fontsize=number_size)
        ax.set_yticklabels(ylabels, fontname=my_font, fontsize=number_size)
    except Exception:
        ax.set_xticklabels(xlabels, fontsize=number_size)
        ax.set_yticklabels(ylabels, fontsize=number_size)

    ax.grid(False)
    ax.set_title(title, fontsize=number_size + 4, pad=30)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_edgecolor('black')

    ax.tick_params(axis='both', width=2.5, color='black', labelsize=number_size)

    cbar = fig.colorbar(scatter, ax=ax, ticks=major_ticks, norm=norm, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_ticks(minor_ticks, minor=True)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2.5)
    cbar.ax.tick_params(axis='y', which='both', labelsize=number_size, color='black', width=2.5)

    _align_colorbar_ticks(cbar, fig)

    fig.subplots_adjust(top=0.85)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the bias correction table (quantile mapping) for U,
    # if enabled.
    bias_pidm_quantiles = None
    bias_gt_quantiles = None
    if APPLY_BIAS_CORRECTION_U:
        print(f"Loading bias correction table from: {BIAS_CORRECTION_U_PATH}")
        bias_data = np.load(BIAS_CORRECTION_U_PATH)
        bias_pidm_quantiles = bias_data['pidm_quantiles']
        bias_gt_quantiles = bias_data['gt_quantiles']
        print("  Bias correction for U: ENABLED")
    else:
        print("Bias correction for U: DISABLED (using raw pidm)")

    # Same for V
    bias_pidm_quantiles_v = None
    bias_gt_quantiles_v = None
    if APPLY_BIAS_CORRECTION_V:
        print(f"Loading bias correction table from: {BIAS_CORRECTION_V_PATH}")
        bias_data_v = np.load(BIAS_CORRECTION_V_PATH)
        bias_pidm_quantiles_v = bias_data_v['pidm_quantiles']
        bias_gt_quantiles_v = bias_data_v['gt_quantiles']
        print("  Bias correction for V: ENABLED")
    else:
        print("Bias correction for V: DISABLED (using raw pidm)")

    print("Loading GT and guide (U and V)...")
    gt_u_all = convert_pixel_to_physical_range(
        load_and_flatten(REF_PATH_GTX), PIXEL_MIN_U, PIXEL_MAX_U, PHYSICAL_MIN_U, PHYSICAL_MAX_U
    )
    guided_u_all = convert_pixel_to_physical_range(
        load_and_flatten(REF_DATA_GUX_PATH), PIXEL_MIN_U, PIXEL_MAX_U, PHYSICAL_MIN_U, PHYSICAL_MAX_U
    )
    gt_v_all = convert_pixel_to_physical_range(
        load_and_flatten(REF_DATA_GTY_PATH), PIXEL_MIN_V, PIXEL_MAX_V, PHYSICAL_MIN_V, PHYSICAL_MAX_V
    )
    guided_v_all = convert_pixel_to_physical_range(
        load_and_flatten(REF_DATA_GUY_PATH), PIXEL_MIN_V, PIXEL_MAX_V, PHYSICAL_MIN_V, PHYSICAL_MAX_V
    )
    print(f"  GT/guide loaded: {gt_u_all.shape[0]} total frames available.")

    batch_folders = sorted(
        glob.glob(os.path.join(LOG_DIR, "sample_batch*")),
        key=lambda p: int(os.path.basename(p).replace("sample_batch", ""))
    )
    print(f"Found {len(batch_folders)} batch folders.")

    if len(batch_folders) == 0:
        print(f"\n[ERROR] No 'sample_batchX' folder was found inside:")
        print(f"        {os.path.abspath(LOG_DIR)}")
        return

    needed_batches = set()
    if FRAME_INDICES:
        needed_batches = set(idx // BATCH_SIZE for idx in FRAME_INDICES)
        print(f"Filtering only frames {FRAME_INDICES} -> needed batches: {sorted(needed_batches)}")

    img_shape = gt_u_all.shape[2:]
    mask_roi = build_polygon_mask(img_shape, POLYGON_COORDS)

    frame_data = {}
    csv_rows = []
    rmse_initial = {'U': [], 'V': [], 'MAG': []}
    rmse_final = {'U': [], 'V': [], 'MAG': []}
    mae_final = {'U': [], 'V': [], 'MAG': []}
    frame_order = []
    uref_matched = []

    hist_pixel_data = {'U': {}, 'V': {}, 'MAG': {}}  

    for batch_folder in batch_folders:
        batch_index = int(os.path.basename(batch_folder).replace("sample_batch", ""))

        if FRAME_INDICES and batch_index not in needed_batches:
            continue

        u_path = os.path.join(batch_folder, f"sample_u_run_{REPEAT}_it{IT}.npy")
        v_path = os.path.join(batch_folder, f"sample_v_run_{REPEAT}_it{IT}.npy")
        mag_path = os.path.join(batch_folder, f"sample_magnitude_run_{REPEAT}_it{IT}.npy")

        if not (os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(mag_path)):
            print(f"[WARNING] One or more .npy files are missing in {batch_folder}, this batch is skipped.")
            continue

        pidm_u_batch = np.load(u_path)
        pidm_v_batch = np.load(v_path)
        pidm_mag_batch = np.load(mag_path)

        batch_actual = pidm_u_batch.shape[0]
        start_idx = batch_index * BATCH_SIZE

        print(f"\nProcessing batch {batch_index}...")

        for j in range(batch_actual):
            global_idx = start_idx + j

            if FRAME_INDICES and global_idx not in FRAME_INDICES:
                continue
            if global_idx >= gt_u_all.shape[0]:
                print(f"  [WARNING] Index {global_idx} out of GT range, skipped.")
                continue

            ch = 1

            pidm_u_2d = pidm_u_batch[j, ch]
            pidm_v_2d = pidm_v_batch[j, ch]
            pidm_mag_2d = pidm_mag_batch[j, ch]

            # Apply bias correction (quantile mapping) to U
            # and/or V, if enabled. The magnitude is recalculated using
            # the already corrected components (the ones that are enabled).
            if APPLY_BIAS_CORRECTION_U:
                pidm_u_2d = np.interp(pidm_u_2d, bias_pidm_quantiles, bias_gt_quantiles)
            if APPLY_BIAS_CORRECTION_V:
                pidm_v_2d = np.interp(pidm_v_2d, bias_pidm_quantiles_v, bias_gt_quantiles_v)

            if APPLY_BIAS_CORRECTION_U or APPLY_BIAS_CORRECTION_V:
                pidm_mag_2d = np.sqrt(pidm_u_2d ** 2 + pidm_v_2d ** 2)

            gt_u_2d = gt_u_all[global_idx][ch]
            gt_v_2d = gt_v_all[global_idx][ch]
            guided_u_2d = guided_u_all[global_idx][ch]
            guided_v_2d = guided_v_all[global_idx][ch]

            gt_mag_2d = np.sqrt(gt_u_2d ** 2 + gt_v_2d ** 2)
            guided_mag_2d = np.sqrt(guided_u_2d ** 2 + guided_v_2d ** 2)

            comps = {
                'U': (guided_u_2d, pidm_u_2d, gt_u_2d),
                'V': (guided_v_2d, pidm_v_2d, gt_v_2d),
                'MAG': (guided_mag_2d, pidm_mag_2d, gt_mag_2d),
            }

            frame_order.append(global_idx)
            uref_matched.append(UREF_DATA[global_idx] if global_idx < len(UREF_DATA) else np.nan)
            csv_row = {'global_frame': global_idx, 'batch_index': batch_index}

            frame_data[global_idx] = {}
            for comp_name, (guided_img, pidm_img, gt_img) in comps.items():
                error_final = np.abs(pidm_img - gt_img)
                error_init = np.abs(guided_img - gt_img)

                error_final_roi = error_final[mask_roi]
                error_init_roi = error_init[mask_roi]

                rmse_f = np.sqrt(np.mean(error_final_roi ** 2))
                rmse_i = np.sqrt(np.mean(error_init_roi ** 2))
                mae_f = error_final_roi.mean()

                rmse_initial[comp_name].append(rmse_i)
                rmse_final[comp_name].append(rmse_f)
                mae_final[comp_name].append(mae_f)

                if global_idx in HISTOGRAM_FRAMES:
                    hist_pixel_data[comp_name][global_idx] = error_final_roi

                csv_row[f'RMSE_Initial_{comp_name}'] = rmse_i
                csv_row[f'RMSE_Final_{comp_name}'] = rmse_f
                csv_row[f'MAE_Final_{comp_name}'] = mae_f

                frame_data[global_idx][comp_name] = (guided_img, pidm_img, gt_img, error_final, error_init)

            csv_rows.append(csv_row)
            print(f"  Frame {global_idx}: RMSE Initial U={csv_row['RMSE_Initial_U']:.4f} -> "
                  f"Final U={csv_row['RMSE_Final_U']:.4f}  |  "
                  f"MAE Final U={csv_row['MAE_Final_U']:.4f}")

    if len(frame_order) == 0:
        print("\n[ERROR] No frame was processed.")
        return

    # =====================================================================
    # --- LOCALSCALE scale (only inside the polygon) ---
    # =====================================================================
    localscale_range = {}
    for comp_name in ['U', 'V', 'MAG']:
        all_values = []
        for global_idx in frame_order:
            guided_img, pidm_img, gt_img, _, _ = frame_data[global_idx][comp_name]
            all_values.extend([guided_img[mask_roi].min(), guided_img[mask_roi].max(),
                                pidm_img[mask_roi].min(), pidm_img[mask_roi].max(),
                                gt_img[mask_roi].min(), gt_img[mask_roi].max()])
        localscale_range[comp_name] = (min(all_values), max(all_values))
        print(f"LOCALSCALE {comp_name}: min={localscale_range[comp_name][0]:.3f}, "
              f"max={localscale_range[comp_name][1]:.3f}")

    # =====================================================================
    # --- Save images (LOCALSCALE and AUTOSCALE, cropped to the polygon) ---
    # =====================================================================
    if SAVE_IMAGES_FRAMES is None:
        frames_to_save_images = frame_order
    else:
        frames_to_save_images = [f for f in frame_order if f in SAVE_IMAGES_FRAMES]

    print(f"\nSaving images ({len(frames_to_save_images)} of {len(frame_order)} processed frames)...")
    for global_idx in frames_to_save_images:
        for comp_name in ['U', 'V', 'MAG']:
            guided_img, pidm_img, gt_img, error_final, error_init = frame_data[global_idx][comp_name]

            vmin_l, vmax_l = localscale_range[comp_name]
            for name, img in [('guided', guided_img), ('pidm', pidm_img), ('gt', gt_img)]:
                out_path = os.path.join(OUTPUT_DIR, f'frame{global_idx}_{name}_{comp_name}_LOCALSCALE.png')
                save_cropped_polygon(img, POLYGON_COORDS, mask_roi, out_path, vmin=vmin_l, vmax=vmax_l)
            colorbar_path = os.path.join(OUTPUT_DIR, f'colorbar_{comp_name}_LOCALSCALE.png')
            save_colorbar_only(colorbar_path, vmin=vmin_l, vmax=vmax_l, label=f"{comp_name} (m/s)")

            # AUTOSCALE -- guided+pidm+GT from the SAME frame share a color
            # range.
            vmin_real = min(guided_img[mask_roi].min(), pidm_img[mask_roi].min(), gt_img[mask_roi].min())
            vmax_real = max(guided_img[mask_roi].max(), pidm_img[mask_roi].max(), gt_img[mask_roi].max())

            vmin_a, vmax_a = vmin_real, vmax_real
            autoscale_range = vmax_a - vmin_a
            if autoscale_range > 0:
                vmin_a -= AUTOSCALE_MARGIN * autoscale_range
                vmax_a += AUTOSCALE_MARGIN * autoscale_range

            for name, img in [('guided', guided_img), ('pidm', pidm_img), ('gt', gt_img)]:
                out_path = os.path.join(OUTPUT_DIR, f'frame{global_idx}_{name}_{comp_name}_AUTOSCALE.png')
                save_cropped_polygon(img, POLYGON_COORDS, mask_roi, out_path, vmin=vmin_a, vmax=vmax_a)
            colorbar_path = os.path.join(OUTPUT_DIR, f'colorbar_frame{global_idx}_{comp_name}_AUTOSCALE.png')

            save_colorbar_only(colorbar_path, vmin=vmin_a, vmax=vmax_a, label=f"{comp_name} (m/s)",
                                vmin_display=vmin_real, vmax_display=vmax_real)

            # OWNSCALE -- each image (guided, pidm, gt) uses ITS OWN
            # independent range, calculated only from its own values.
            for name, img in [('guided', guided_img), ('pidm', pidm_img), ('gt', gt_img)]:
                vmin_own, vmax_own = img[mask_roi].min(), img[mask_roi].max()
                out_path = os.path.join(OUTPUT_DIR, f'frame{global_idx}_{name}_{comp_name}_OWNSCALE.png')
                save_cropped_polygon(img, POLYGON_COORDS, mask_roi, out_path, vmin=vmin_own, vmax=vmax_own)
                colorbar_path = os.path.join(OUTPUT_DIR, f'colorbar_frame{global_idx}_{name}_{comp_name}_OWNSCALE.png')
                save_colorbar_only(colorbar_path, vmin=vmin_own, vmax=vmax_own, label=f"{comp_name} (m/s)")

            out_path_err = os.path.join(OUTPUT_DIR, f'frame{global_idx}_error_abs_{comp_name}.png')
            err_vmin, err_vmax = error_final[mask_roi].min(), error_final[mask_roi].max()
            save_cropped_polygon(error_final, POLYGON_COORDS, mask_roi, out_path_err, cmap='inferno',
                                  vmin=err_vmin, vmax=err_vmax)
            colorbar_path = os.path.join(OUTPUT_DIR, f'colorbar_frame{global_idx}_error_{comp_name}.png')
            save_colorbar_only(colorbar_path, vmin=err_vmin, vmax=err_vmax,
                                cmap='inferno', label=f"Error {comp_name} (m/s)")

    # =====================================================================
    # --- CSV ---
    # =====================================================================
    csv_path = os.path.join(OUTPUT_DIR, 'metrics_breakdown.csv')
    fieldnames = ['global_frame', 'batch_index']
    for comp_name in ['U', 'V', 'MAG']:
        fieldnames += [f'RMSE_Initial_{comp_name}', f'RMSE_Final_{comp_name}', f'MAE_Final_{comp_name}']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"\nCSV saved at: {csv_path}")

    # =====================================================================
    # --- Overlapped histogram A/B/C (per component) ---
    # =====================================================================
    print("Generating overlapped histograms (A/B/C)...")
    x_max_by_comp = {'U': 1.7, 'V': 1.7, 'MAG': 1.7}
    if len(HISTOGRAM_FRAMES) == 3 and all(f in hist_pixel_data['U'] for f in HISTOGRAM_FRAMES):
        for comp_name in ['U', 'V', 'MAG']:
            f_a, f_b, f_c = HISTOGRAM_FRAMES
            hist_path = os.path.join(OUTPUT_DIR, f'overlapped_histogram_{comp_name}.png')
            plot_overlapped_histograms(
                data1=hist_pixel_data[comp_name][f_a],
                data2=hist_pixel_data[comp_name][f_b],
                data3=hist_pixel_data[comp_name][f_c],
                label1='A', label2='B', label3='C',
                out_path=hist_path,
                num_bins=60,
                x_max=x_max_by_comp[comp_name],
            )
            print(f"  Histogram {comp_name} saved (A=frame{f_a}, B=frame{f_b}, C=frame{f_c}).")
    else:
        print(f"  [WARNING] Could not generate the overlapped histogram -- "
              f"make sure HISTOGRAM_FRAMES={HISTOGRAM_FRAMES} is within FRAME_INDICES/all processed frames.")

    # =====================================================================
    # --- Scatter RMSE_init vs RMSE_final, color=U_ref ---
    # =====================================================================
    print("Generating scatter plots (RMSE_init vs RMSE_final, color=U_ref)...")
    markers = {'U': '.', 'V': '.', 'MAG': '.'}
    titles = {'U': 'Component U', 'V': 'Component V', 'MAG': 'Magnitude'}
    for comp_name in ['U', 'V', 'MAG']:
        scatter_path = os.path.join(OUTPUT_DIR, f'rmse_scatter_{comp_name}.png')
        make_scatter_figure(
            rmse_initial[comp_name], rmse_final[comp_name], uref_matched,
            markers[comp_name], titles[comp_name], scatter_path
        )

    # =====================================================================
    # --- Final summary -- identical design to the original large file ---
    # =====================================================================
    ri_u = np.array(rmse_initial['U']); rf_u = np.array(rmse_final['U'])
    ri_v = np.array(rmse_initial['V']); rf_v = np.array(rmse_final['V'])
    ri_mag = np.array(rmse_initial['MAG']); rf_mag = np.array(rmse_final['MAG'])

    print("\n" + "=" * 60)
    print("     Statistical Analysis of RMSE Values (U, V & Magnitude)")
    print("=" * 60)
    print(f"[*] Initial RMSE U:")
    print(f"    - Mean:                 {ri_u.mean():.6f}")
    print(f"    - Standard deviation:   {ri_u.std():.6f}")
    print(f"[*] Initial RMSE V:")
    print(f"    - Mean:                 {ri_v.mean():.6f}")
    print(f"    - Standard deviation:   {ri_v.std():.6f}")
    print(f"[*] Initial RMSE Magnitude:")
    print(f"    - Mean:                 {ri_mag.mean():.6f}")
    print(f"    - Standard deviation:   {ri_mag.std():.6f}")
    print("-" * 60)
    print(f"[*] Final RMSE U:")
    print(f"    - Mean:                 {rf_u.mean():.6f}")
    print(f"    - Standard deviation:   {rf_u.std():.6f}")
    print(f"[*] Final RMSE V:")
    print(f"    - Mean:                 {rf_v.mean():.6f}")
    print(f"    - Standard deviation:   {rf_v.std():.6f}")
    print(f"[*] Final RMSE Magnitude:")
    print(f"    - Mean:                 {rf_mag.mean():.6f}")
    print(f"    - Standard deviation:   {rf_mag.std():.6f}")
    print("=" * 60 + "\n")

    mae_u_np = np.array(mae_final['U'])
    mae_v_np = np.array(mae_final['V'])
    mae_mag_np = np.array(mae_final['MAG'])

    print("\n" + "=" * 60)
    print("     Statistical Analysis of Mean Absolute Error (MAE)")
    print("=" * 60)
    print(f"[*] Final MAE U (for the {len(mae_u_np)} images):")
    print(f"    - Result: MAE = {mae_u_np.mean():.4f} ± {mae_u_np.std():.4f}")
    print(f"[*] Final MAE V (for the {len(mae_v_np)} images):")
    print(f"    - Result: MAE = {mae_v_np.mean():.4f} ± {mae_v_np.std():.4f}")
    print(f"[*] Final MAE Magnitude (for the {len(mae_mag_np)} images):")
    print(f"    - Result: MAE = {mae_mag_np.mean():.4f} ± {mae_mag_np.std():.4f}")
    print("=" * 60 + "\n")

    print("\n" + "=" * 80)
    print("             Breakdown of Frames Used for RMSE Calculation")
    print("=" * 80)
    print(f"{'Global Frame':<12} {'Init RMSE U':<12} {'Final RMSE U':<12} {'Init RMSE V':<12} "
          f"{'Final RMSE V':<12} {'Init RMSE Mag':<14} {'Final RMSE Mag':<14}")
    print("-" * 80)
    for idx, global_idx in enumerate(frame_order):
        print(f"{global_idx:<12} {ri_u[idx]:<12.4f} {rf_u[idx]:<12.4f} {ri_v[idx]:<12.4f} "
              f"{rf_v[idx]:<12.4f} {ri_mag[idx]:<14.4f} {rf_mag[idx]:<14.4f}")
    print("=" * 80 + "\n")

    print(f"Done. All images/plots were saved at: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()