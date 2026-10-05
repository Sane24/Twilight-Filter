import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from skimage import io, color, img_as_float, data, transform as sktr
from align_image_code import align_images

os.makedirs("outputs/part1", exist_ok=True)


# 4-loop convolution

def conv2d_four_loops(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    h, w = image.shape
    kh, kw = kernel.shape

    # total padding = kernel_size - 1
    pad_top = (kh - 1) // 2
    pad_bottom = (kh - 1) - pad_top

    pad_left = (kw - 1) // 2
    pad_right = (kw - 1) - pad_left

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    )

    # convolution flips the kernel
    flipped_kernel = np.flip(kernel, axis=(0, 1))

    output = np.zeros_like(image, dtype=float)

    for y in range(h):
        for x in range(w):
            for ky in range(kh):
                for kx in range(kw):
                    output[y, x] += (
                        padded[y + ky, x + kx]
                        * flipped_kernel[ky, kx]
                    )

    return output


# 2-loop convolution

def conv2d_two_loops(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    h, w = image.shape
    kh, kw = kernel.shape

    pad_top = (kh - 1) // 2
    pad_bottom = (kh - 1) - pad_top

    pad_left = (kw - 1) // 2
    pad_right = (kw - 1) - pad_left

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    )

    flipped_kernel = np.flip(kernel, axis=(0, 1))

    output = np.zeros_like(image, dtype=float)

    for y in range(h):
        for x in range(w):
            region = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(region * flipped_kernel)

    return output


# Display helpers

def normalize_signed(image):
    max_abs = np.max(np.abs(image))

    if max_abs == 0:
        return np.zeros_like(image)

    # map negative -> dark, 0 -> gray, positive -> bright
    return 0.5 + 0.5 * image / max_abs


def save_gray(filename, image, signed=False):
    if signed:
        shown = normalize_signed(image)
    else:
        shown = np.clip(image, 0, 1)

    plt.imsave(
        filename,
        shown,
        cmap="gray",
        vmin=0,
        vmax=1
    )


# Load selfie

selfie = img_as_float(io.imread("selfie.jpg"))

if selfie.ndim == 3:
    if selfie.shape[2] == 4:
        selfie = color.rgba2rgb(selfie)

    selfie_gray = color.rgb2gray(selfie)
else:
    selfie_gray = selfie


# Filters

box_filter = np.ones((9, 9), dtype=float) / 81.0

Dx = np.array([
    [1, -1]
], dtype=float)

Dy = np.array([
    [1],
    [-1]
], dtype=float)


# Runtime / correctness comparison
# Use a smaller crop so 4 loops don't take forever

test = selfie_gray[:200, :200]

print("\n--- CONVOLUTION COMPARISON ---")

start = time.perf_counter()
four_loop_result = conv2d_four_loops(test, box_filter)
four_loop_time = time.perf_counter() - start

start = time.perf_counter()
two_loop_result = conv2d_two_loops(test, box_filter)
two_loop_time = time.perf_counter() - start

start = time.perf_counter()
scipy_result = convolve2d(
    test,
    box_filter,
    mode="same",
    boundary="fill",
    fillvalue=0
)
scipy_time = time.perf_counter() - start


print(f"4-loop time: {four_loop_time:.4f} sec")
print(f"2-loop time: {two_loop_time:.4f} sec")
print(f"SciPy time:  {scipy_time:.4f} sec")

print(
    "4-loop max error vs SciPy:",
    np.max(np.abs(four_loop_result - scipy_result))
)

print(
    "2-loop max error vs SciPy:",
    np.max(np.abs(two_loop_result - scipy_result))
)


# Required selfie results

selfie_blur = conv2d_two_loops(selfie_gray, box_filter)
selfie_dx = conv2d_two_loops(selfie_gray, Dx)
selfie_dy = conv2d_two_loops(selfie_gray, Dy)


save_gray(
    "outputs/part1/selfie_original.jpg",
    selfie_gray
)

save_gray(
    "outputs/part1/selfie_box_9x9.jpg",
    selfie_blur
)

save_gray(
    "outputs/part1/selfie_dx.jpg",
    selfie_dx,
    signed=True
)

save_gray(
    "outputs/part1/selfie_dy.jpg",
    selfie_dy,
    signed=True
)

print("\nSaved Part 1.1 images to outputs/part1/")
