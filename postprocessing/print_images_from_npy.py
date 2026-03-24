import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RGB_CHANNELS = {1, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print and visualize images stored inside an NPY file."
    )
    parser.add_argument("input_file", type=Path, help="Path to the .npy file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where extracted images will be saved as PNG files",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        help="Maximum number of images to preview or save",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Flat image index where extraction starts",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step between extracted images",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Matplotlib colormap for 2D images",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a contact sheet with the selected images",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Hide colorbars in the preview figure",
    )
    parser.add_argument(
        "--split-channels",
        action="store_true",
        help="Save one PNG per channel when an image has multiple channels",
    )
    return parser.parse_args()


def ensure_valid_args(args):
    if not args.input_file.exists():
        raise FileNotFoundError(f"File not found: {args.input_file}")
    if args.max_images < 1:
        raise ValueError("--max-images must be at least 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be 0 or greater")
    if args.step < 1:
        raise ValueError("--step must be at least 1")
    if args.output_dir is None and not args.show:
        args.show = True


def describe_array(array):
    print(f"Loaded: {array.shape} | dtype={array.dtype}")
    print(f"Value range: min={array.min()} | max={array.max()}")


def is_channel_last(shape):
    return len(shape) >= 3 and shape[-1] in RGB_CHANNELS


def is_channel_first(shape):
    return len(shape) >= 3 and shape[-3] in RGB_CHANNELS


def to_channel_last(image):
    if image.ndim == 3 and image.shape[0] in RGB_CHANNELS and image.shape[-1] not in RGB_CHANNELS:
        return np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    return image


def flatten_images(array):
    if array.ndim < 2:
        raise ValueError("The NPY file does not contain image-shaped data.")

    if array.ndim == 2:
        return [array], "single 2D image"

    if array.ndim == 3:
        if is_channel_last(array.shape) or is_channel_first(array.shape):
            return [to_channel_last(array)], "single multi-channel image"
        return [array[index] for index in range(array.shape[0])], "stack of 2D images"

    if array.ndim == 4:
        if is_channel_last(array.shape):
            return [array[index] for index in range(array.shape[0])], "batch of channel-last images"
        if array.shape[1] in RGB_CHANNELS:
            return [to_channel_last(array[index]) for index in range(array.shape[0])], "batch of channel-first images"
        flattened = array.reshape(-1, array.shape[-2], array.shape[-1])
        return [flattened[index] for index in range(flattened.shape[0])], "stacked grayscale sequences"

    trailing_shape = array.shape[-3:]
    if is_channel_last(trailing_shape):
        flattened = array.reshape(-1, *trailing_shape)
        return [flattened[index] for index in range(flattened.shape[0])], "high-dimensional channel-last stack"

    if is_channel_first(trailing_shape):
        flattened = array.reshape(-1, *trailing_shape)
        return [to_channel_last(flattened[index]) for index in range(flattened.shape[0])], "high-dimensional channel-first stack"

    flattened = array.reshape(-1, array.shape[-2], array.shape[-1])
    return [flattened[index] for index in range(flattened.shape[0])], "high-dimensional grayscale stack"


def select_images(images, start_index, max_images, step):
    selected = []
    for index in range(start_index, len(images), step):
        selected.append((index, images[index]))
        if len(selected) == max_images:
            break
    return selected


def normalize_for_png(image):
    if np.issubdtype(image.dtype, np.integer) and image.min() >= 0 and image.max() <= 255:
        return image.astype(np.uint8)

    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (image - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def split_image_channels(image):
    if image.ndim != 3:
        return [(None, image)]

    if image.shape[-1] in RGB_CHANNELS:
        return [(channel_index, image[..., channel_index]) for channel_index in range(image.shape[-1])]

    if image.shape[0] in RGB_CHANNELS:
        return [(channel_index, image[channel_index, ...]) for channel_index in range(image.shape[0])]

    return [(None, image)]


def save_images(selected_images, output_dir, split_channels=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    for flat_index, image in selected_images:
        saveable_images = split_image_channels(image) if split_channels else [(None, image)]
        for channel_index, image_to_save in saveable_images:
            image_uint8 = normalize_for_png(image_to_save)
            filename = f"image_{flat_index:04d}.png"
            if channel_index is not None:
                filename = f"image_{flat_index:04d}_channel_{channel_index}.png"
            output_path = output_dir / filename
            plt.imsave(output_path, image_uint8, cmap="gray" if image_uint8.ndim == 2 else None)
            print(f"Saved: {output_path}")


def show_contact_sheet(selected_images, cmap, include_colorbar):
    columns = min(4, len(selected_images))
    rows = math.ceil(len(selected_images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis in axes[len(selected_images):]:
        axis.axis("off")

    for axis, (flat_index, image) in zip(axes, selected_images):
        rendered = axis.imshow(image, cmap=cmap if image.ndim == 2 else None)
        axis.set_title(f"Index {flat_index}")
        axis.axis("off")
        if include_colorbar and image.ndim == 2:
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    plt.show()


def main():
    args = parse_args()
    ensure_valid_args(args)

    data = np.load(args.input_file)
    describe_array(data)

    images, layout = flatten_images(data)
    print(f"Detected layout: {layout}")
    print(f"Total extracted images: {len(images)}")

    selected_images = select_images(images, args.start_index, args.max_images, args.step)
    if not selected_images:
        raise ValueError("No images selected. Check --start-index, --max-images, and --step.")

    selected_indices = [str(flat_index) for flat_index, _ in selected_images]
    print(f"Selected flat indices: {', '.join(selected_indices)}")

    if args.output_dir is not None:
        save_images(selected_images, args.output_dir, split_channels=args.split_channels)

    if args.show:
        show_contact_sheet(selected_images, args.cmap, include_colorbar=not args.no_colorbar)


if __name__ == "__main__":
    main()