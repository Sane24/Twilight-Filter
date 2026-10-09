import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from scipy.interpolate import splprep, splev
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


# PART 1.2 FINITE DIFFERENCE OPERATORS

print("\n--- PART 1.2: FINITE DIFFERENCE OPERATORS ---")

# Standard Cameraman test image
cameraman = img_as_float(data.camera())

# Compute partial derivatives
camera_dx = convolve2d(
    cameraman,
    Dx,
    mode="same",
    boundary="fill",
    fillvalue=0
)

camera_dy = convolve2d(
    cameraman,
    Dy,
    mode="same",
    boundary="fill",
    fillvalue=0
)

# Gradient magnitude
gradient_magnitude = np.sqrt(camera_dx**2 + camera_dy**2)

# Normalize only for visualization
gradient_display = gradient_magnitude / np.max(gradient_magnitude)


# Threshold gradient magnitude to produce binary edges

threshold = 0.20

binary_edges = gradient_display > threshold



save_gray(
    "outputs/part1/cameraman_original.jpg",
    cameraman
)

save_gray(
    "outputs/part1/cameraman_dx.jpg",
    camera_dx,
    signed=True
)

save_gray(
    "outputs/part1/cameraman_dy.jpg",
    camera_dy,
    signed=True
)

save_gray(
    "outputs/part1/cameraman_gradient_magnitude.jpg",
    gradient_display
)

save_gray(
    "outputs/part1/cameraman_edges.jpg",
    binary_edges.astype(float)
)

print(f"Edge threshold: {threshold}")
print("Saved Part 1.2 images to outputs/part1/")

# PART 1.3 — DERIVATIVE OF GAUSSIAN

print("\n--- PART 1.3: DERIVATIVE OF GAUSSIAN ---")


# Create 2D Gaussian kernel

gaussian_size = 9
gaussian_sigma = 2.0

gaussian_1d = cv2.getGaussianKernel(
    gaussian_size,
    gaussian_sigma
)

gaussian_2d = gaussian_1d @ gaussian_1d.T


# Method 1:
# Gaussian blur first, then finite differences

camera_blurred = convolve2d(
    cameraman,
    gaussian_2d,
    mode="same",
    boundary="fill",
    fillvalue=0
)

blurred_dx = convolve2d(
    camera_blurred,
    Dx,
    mode="same",
    boundary="fill",
    fillvalue=0
)

blurred_dy = convolve2d(
    camera_blurred,
    Dy,
    mode="same",
    boundary="fill",
    fillvalue=0
)

blurred_gradient = np.sqrt(
    blurred_dx**2 + blurred_dy**2
)
# Ignore padding artifacts at the outer boundary when choosing the normalization scale.
margin = 10

gradient_scale = np.max(
    blurred_gradient[
        margin:-margin,
        margin:-margin
    ]
)

blurred_gradient_display = np.clip(
    blurred_gradient / gradient_scale,
    0,
    1
)

blurred_threshold = 0.15

blurred_edges = (
    blurred_gradient_display > blurred_threshold
)

# Suppress artificial edges caused purely by zero padding
blurred_edges[:margin, :] = False
blurred_edges[-margin:, :] = False
blurred_edges[:, :margin] = False
blurred_edges[:, -margin:] = False


# Build Derivative-of-Gaussian filters

dog_x = convolve2d(
    gaussian_2d,
    Dx,
    mode="full"
)

dog_y = convolve2d(
    gaussian_2d,
    Dy,
    mode="full"
)


# Apply DoG filters directly to original image
dog_dx = convolve2d(
    cameraman,
    dog_x,
    mode="same",
    boundary="fill",
    fillvalue=0
)

dog_dy = convolve2d(
    cameraman,
    dog_y,
    mode="same",
    boundary="fill",
    fillvalue=0
)

dog_gradient = np.sqrt(
    dog_dx**2 + dog_dy**2
)

# Use the SAME normalization scale as the two-step method.
dog_gradient_display = np.clip(
    dog_gradient / gradient_scale,
    0,
    1
)

dog_edges = (
    dog_gradient_display > blurred_threshold
)

dog_edges[:margin, :] = False
dog_edges[-margin:, :] = False
dog_edges[:, :margin] = False
dog_edges[:, -margin:] = False


# Verify equivalence

margin = 10

dx_difference = np.max(
    np.abs(
        blurred_dx[margin:-margin, margin:-margin]
        -
        dog_dx[margin:-margin, margin:-margin]
    )
)

dy_difference = np.max(
    np.abs(
        blurred_dy[margin:-margin, margin:-margin]
        -
        dog_dy[margin:-margin, margin:-margin]
    )
)

print(
    f"Gaussian size: {gaussian_size}x{gaussian_size}"
)

print(
    f"Gaussian sigma: {gaussian_sigma}"
)

print(
    f"Smoothed edge threshold: {blurred_threshold}"
)

print(
    "Max interior difference, Dx:",
    dx_difference
)

print(
    "Max interior difference, Dy:",
    dy_difference
)


save_gray(
    "outputs/part1/cameraman_blurred.jpg",
    camera_blurred
)

save_gray(
    "outputs/part1/cameraman_blurred_dx.jpg",
    blurred_dx,
    signed=True
)

save_gray(
    "outputs/part1/cameraman_blurred_dy.jpg",
    blurred_dy,
    signed=True
)

save_gray(
    "outputs/part1/cameraman_blurred_gradient.jpg",
    blurred_gradient_display
)

save_gray(
    "outputs/part1/cameraman_blurred_edges.jpg",
    blurred_edges.astype(float)
)


# DoG filters themselves
save_gray(
    "outputs/part1/dog_x_filter.jpg",
    dog_x,
    signed=True
)

save_gray(
    "outputs/part1/dog_y_filter.jpg",
    dog_y,
    signed=True
)


# Direct DoG results
save_gray(
    "outputs/part1/cameraman_dog_dx.jpg",
    dog_dx,
    signed=True
)

save_gray(
    "outputs/part1/cameraman_dog_dy.jpg",
    dog_dy,
    signed=True
)

save_gray(
    "outputs/part1/cameraman_dog_gradient.jpg",
    dog_gradient_display
)

save_gray(
    "outputs/part1/cameraman_dog_edges.jpg",
    dog_edges.astype(float)
)

print("Saved Part 1.3 images to outputs/part1/")

# COLOR IMAGE FILTERING HELPERS

def ensure_rgb(image):
    image = img_as_float(image)

    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)

    if image.shape[2] == 4:
        image = color.rgba2rgb(image)

    return image


def convolve_color(image, kernel):
    """
    Apply a 2D kernel independently to each RGB channel.
    """
    image = ensure_rgb(image)

    result = np.zeros_like(image, dtype=float)

    for c in range(3):
        result[:, :, c] = convolve2d(
            image[:, :, c],
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0
        )

    return result

def crop_to_common_valid_area(im1, im2, valid_fraction=0.94):
    """
    Remove black padding introduced by alignment/rotation while keeping
    both aligned images identically cropped.
    """

    if im1.ndim == 3:
        valid1 = np.any(np.abs(im1) > 1e-8, axis=2)
    else:
        valid1 = np.abs(im1) > 1e-8

    if im2.ndim == 3:
        valid2 = np.any(np.abs(im2) > 1e-8, axis=2)
    else:
        valid2 = np.abs(im2) > 1e-8

    common = valid1 & valid2

    h, w = common.shape

    top = 0
    bottom = h
    left = 0
    right = w

    # Don't let the automatic crop destroy most of the image.
    min_height = int(0.55 * h)
    min_width = int(0.55 * w)

    while True:
        top_score = np.mean(common[top, left:right])
        bottom_score = np.mean(common[bottom - 1, left:right])
        left_score = np.mean(common[top:bottom, left])
        right_score = np.mean(common[top:bottom, right - 1])

        scores = [
            top_score,
            bottom_score,
            left_score,
            right_score
        ]

        # All four borders are now almost entirely valid.
        if min(scores) >= valid_fraction:
            break

        worst_side = np.argmin(scores)

        if worst_side == 0 and bottom - top > min_height:
            top += 1

        elif worst_side == 1 and bottom - top > min_height:
            bottom -= 1

        elif worst_side == 2 and right - left > min_width:
            left += 1

        elif worst_side == 3 and right - left > min_width:
            right -= 1

        else:
            # Safety stop
            break

    # Small extra inward trim to avoid a thin black line.
    padding = 3

    top = min(top + padding, bottom - 1)
    bottom = max(bottom - padding, top + 1)

    left = min(left + padding, right - 1)
    right = max(right - padding, left + 1)

    print(
        f"Crop bounds: top={top}, bottom={bottom}, "
        f"left={left}, right={right}"
    )

    print(
        f"Cropped shape: {bottom - top} x {right - left}"
    )

    return (
        im1[top:bottom, left:right],
        im2[top:bottom, left:right]
    )

def save_color(filename, image):
    plt.imsave(
        filename,
        np.clip(image, 0, 1)
    )


def normalize_signed_color(image):
    """
    Make signed high-frequency values visible:
    negative -> dark
    zero     -> gray
    positive -> bright
    """
    max_abs = np.max(np.abs(image))

    if max_abs == 0:
        return np.zeros_like(image) + 0.5

    return np.clip(
        0.5 + 0.5 * image / max_abs,
        0,
        1
    )

def gaussian_kernel_from_sigma(sigma):
    size = int(6 * sigma + 1)

    if size % 2 == 0:
        size += 1

    g1d = cv2.getGaussianKernel(size, sigma)
    return g1d @ g1d.T

def align_to_reference(moving, reference):
    """
    Align `moving` to `reference` using two corresponding points.

    Uses a similarity transform:
    translation + rotation + uniform scale.

    Unlike the starter align_images(), this does NOT enlarge the canvas.
    The aligned output has exactly the same HxW as `reference`.
    """

    print("Click TWO corresponding points on the MOVING image.")
    fig = plt.figure()
    plt.imshow(moving)
    plt.title("Moving image: click 2 points")
    plt.axis("off")
    moving_pts = plt.ginput(2, timeout=-1)
    plt.close(fig)

    print("Click the SAME TWO points on the REFERENCE image.")
    fig = plt.figure()
    plt.imshow(reference)
    plt.title("Reference image: click same 2 points")
    plt.axis("off")
    reference_pts = plt.ginput(2, timeout=-1)
    plt.close(fig)

    if len(moving_pts) != 2 or len(reference_pts) != 2:
        raise RuntimeError("You must select exactly two points in each image.")

    moving_pts = np.array(moving_pts)
    reference_pts = np.array(reference_pts)

    transform = sktr.estimate_transform(
        "similarity",
        src=moving_pts,
        dst=reference_pts
    )

    aligned = sktr.warp(
        moving,
        inverse_map=transform.inverse,
        output_shape=reference.shape[:2],
        mode="edge",
        preserve_range=True
    )

    return aligned, reference

def hybrid_image(high_image, low_image, sigma_high, sigma_low):
    """
    high_image contributes high frequencies.
    low_image contributes low frequencies.
    """

    high_image = ensure_rgb(high_image)
    low_image = ensure_rgb(low_image)

    # Low-pass image
    low_kernel = gaussian_kernel_from_sigma(sigma_low)

    low_frequencies = convolve_color(
        low_image,
        low_kernel
    )

    # High-pass image = original - blurred
    high_kernel = gaussian_kernel_from_sigma(sigma_high)

    high_blurred = convolve_color(
        high_image,
        high_kernel
    )

    high_frequencies = (
        high_image - high_blurred
    )

    hybrid = (
        low_frequencies
        +
        high_frequencies
    )

    return (
        np.clip(hybrid, 0, 1),
        low_frequencies,
        high_frequencies
    )
def save_fourier_spectrum(filename, image):
    """
    Save log magnitude of centered 2D Fourier transform.
    Converts RGB images to grayscale first.
    """
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    spectrum = np.fft.fftshift(
        np.fft.fft2(gray)
    )

    magnitude = np.log1p(
        np.abs(spectrum)
    )

    magnitude /= np.max(magnitude)

    save_gray(
        filename,
        magnitude
    )

def run_hybrid_pair(
    name,
    high_path,
    low_path,
    sigma_high,
    sigma_low
):
    high = ensure_rgb(io.imread(high_path))
    low = ensure_rgb(io.imread(low_path))

    print(f"\n--- Hybrid: {name} ---")
    print(f"First select 2 points on {low_path}")
    print(f"Then select the corresponding 2 points on {high_path}")

    # low image rotates/scales to match the high-frequency image
    low_aligned, high_aligned = align_to_reference(
        low,
        high
    )

    hybrid, low_freq, high_freq = hybrid_image(
        high_aligned,
        low_aligned,
        sigma_high,
        sigma_low
    )

    prefix = f"outputs/part2_2/{name}"

    save_color(
        prefix + "_high_aligned.jpg",
        high_aligned
    )

    save_color(
        prefix + "_low_aligned.jpg",
        low_aligned
    )

    save_color(
        prefix + "_high_frequency.jpg",
        normalize_signed_color(high_freq)
    )

    save_color(
        prefix + "_low_frequency.jpg",
        low_freq
    )

    save_color(
        prefix + "_hybrid.jpg",
        hybrid
    )

    far = cv2.resize(
        hybrid,
        None,
        fx=0.18,
        fy=0.18,
        interpolation=cv2.INTER_AREA
    )

    save_color(
        prefix + "_far.jpg",
        far
    )

    print(
        f"{name}: high sigma={sigma_high}, "
        f"low sigma={sigma_low}"
    )

    # Fourier spectra for frequency analysis

    save_fourier_spectrum(
        prefix + "_high_input_fft.jpg",
        high_aligned
    )

    save_fourier_spectrum(
        prefix + "_low_input_fft.jpg",
        low_aligned
    )

    save_fourier_spectrum(
        prefix + "_high_filtered_fft.jpg",
        high_freq
    )

    save_fourier_spectrum(
        prefix + "_low_filtered_fft.jpg",
        low_freq
    )

    save_fourier_spectrum(
        prefix + "_hybrid_fft.jpg",
        hybrid
    )

    return hybrid

def crop_common_valid(im1, im2):
    """
    Crop both aligned images to a central rectangle where neither
    image contains black padding introduced by alignment transforms.
    """

    valid1 = np.max(im1, axis=2) > 1e-6
    valid2 = np.max(im2, axis=2) > 1e-6

    common = valid1 & valid2

    # Fraction of valid pixels in each row/column
    row_fraction = np.mean(common, axis=1)
    col_fraction = np.mean(common, axis=0)

    # Keep rows/cols that are almost entirely valid
    good_rows = np.where(row_fraction > 0.98)[0]
    good_cols = np.where(col_fraction > 0.98)[0]

    if len(good_rows) == 0 or len(good_cols) == 0:
        print("Could not find clean common crop; leaving images unchanged.")
        return im1, im2

    top = good_rows[0]
    bottom = good_rows[-1] + 1
    left = good_cols[0]
    right = good_cols[-1] + 1

    return (
        im1[top:bottom, left:right],
        im2[top:bottom, left:right]
    )
def gaussian_stack(image, levels=5, base_sigma=1.0):
    """
    Gaussian stack with NO downsampling.

    G[0] = original
    G[1], G[2], ... use progressively larger Gaussian sigmas.
    """
    image = ensure_rgb(image)

    stack = [image]

    for level in range(1, levels):
        sigma = base_sigma * (2 ** (level - 1))

        kernel = gaussian_kernel_from_sigma(sigma)

        blurred = convolve_color(
            image,
            kernel
        )

        stack.append(blurred)

    return stack

def gaussian_stack_gray(image, levels=5, base_sigma=1.0):
    """
    Gaussian stack for a grayscale image / mask.
    No downsampling.
    """
    image = np.asarray(image, dtype=float)

    stack = [image]

    for level in range(1, levels):
        sigma = base_sigma * (2 ** (level - 1))

        kernel = gaussian_kernel_from_sigma(
            sigma
        )

        blurred = convolve2d(
            image,
            kernel,
            mode="same",
            boundary="symm"
        )

        stack.append(blurred)

    return stack


def multiresolution_blend(
    image_a,
    image_b,
    mask,
    levels=5,
    base_sigma=1.0
):
    """
    Blend image_a and image_b using:
        Gaussian stack of mask
        Laplacian stacks of both images
    """

    image_a = ensure_rgb(image_a)
    image_b = ensure_rgb(image_b)

    mask = np.asarray(mask, dtype=float)
    mask = np.clip(mask, 0, 1)

    # Image Laplacian stacks
    lap_a = laplacian_stack(
        image_a,
        levels=levels,
        base_sigma=base_sigma
    )

    lap_b = laplacian_stack(
        image_b,
        levels=levels,
        base_sigma=base_sigma
    )

    # Mask Gaussian stack
    mask_stack = gaussian_stack_gray(
        mask,
        levels=levels,
        base_sigma=base_sigma
    )

    blended_levels = []
    masked_a_levels = []
    masked_b_levels = []

    for i in range(levels):

        # Convert HxW mask -> HxWx1
        m = mask_stack[i][:, :, None]

        part_a = m * lap_a[i]
        part_b = (1.0 - m) * lap_b[i]

        blended = part_a + part_b

        masked_a_levels.append(part_a)
        masked_b_levels.append(part_b)
        blended_levels.append(blended)

    # Reconstruct by summing all Laplacian bands
    result = np.sum(
        np.stack(blended_levels),
        axis=0
    )

    return (
        np.clip(result, 0, 1),
        mask_stack,
        lap_a,
        lap_b,
        masked_a_levels,
        masked_b_levels,
        blended_levels
    )


def laplacian_stack(image, levels=5, base_sigma=1.0):
    """
    L[i] = G[i] - G[i+1]
    Final level contains the coarsest Gaussian level.
    """
    g_stack = gaussian_stack(
        image,
        levels,
        base_sigma
    )

    l_stack = []

    for i in range(levels - 1):
        l_stack.append(
            g_stack[i] - g_stack[i + 1]
        )

    # Lowest-frequency residual
    l_stack.append(g_stack[-1])

    return l_stack

# PART 2.1 — IMAGE SHARPENING

print("\n--- PART 2.1: IMAGE SHARPENING ---")

os.makedirs("outputs/part2_1", exist_ok=True)


# Gaussian kernel

sharp_gaussian_size = 9
sharp_gaussian_sigma = 2.0

g1d = cv2.getGaussianKernel(
    sharp_gaussian_size,
    sharp_gaussian_sigma
)

G = g1d @ g1d.T


# Build impulse kernel

impulse = np.zeros_like(G)

center = sharp_gaussian_size // 2
impulse[center, center] = 1.0


# Unsharp mask
#
# sharpened =
# original + alpha * (original - blurred)
#
# =
# [(1 + alpha) * impulse - alpha * Gaussian] * image

def unsharp_mask(image, alpha):
    sharpening_kernel = (
        (1 + alpha) * impulse
        -
        alpha * G
    )

    return convolve_color(
        image,
        sharpening_kernel
    )


# TAJ MAHAL

taj = ensure_rgb(
    io.imread("taj.jpg")
)

taj_blurred = convolve_color(
    taj,
    G
)

taj_high = taj - taj_blurred


save_color(
    "outputs/part2_1/taj_original.jpg",
    taj
)

save_color(
    "outputs/part2_1/taj_blurred.jpg",
    taj_blurred
)

save_color(
    "outputs/part2_1/taj_high_frequency.jpg",
    normalize_signed_color(taj_high)
)


# Try several sharpening strengths
alphas = [0.5, 1.0, 2.0]

for alpha in alphas:

    taj_sharp = unsharp_mask(
        taj,
        alpha
    )

    save_color(
        f"outputs/part2_1/taj_sharpened_alpha_{alpha}.jpg",
        taj_sharp
    )


# SECOND IMAGE — SELFIE

selfie_color = ensure_rgb(
    io.imread("selfie.jpg")
)

selfie_blurred_color = convolve_color(
    selfie_color,
    G
)

selfie_high = (
    selfie_color
    -
    selfie_blurred_color
)

selfie_sharp = unsharp_mask(
    selfie_color,
    1.0
)


save_color(
    "outputs/part2_1/selfie_original_color.jpg",
    selfie_color
)

save_color(
    "outputs/part2_1/selfie_blurred_color.jpg",
    selfie_blurred_color
)

save_color(
    "outputs/part2_1/selfie_high_frequency.jpg",
    normalize_signed_color(selfie_high)
)

save_color(
    "outputs/part2_1/selfie_sharpened.jpg",
    selfie_sharp
)


# SHARP -> BLUR -> RESHARPEN EXPERIMENT

# Artificially blur the original sharp selfie
blurred_test = convolve_color(
    selfie_color,
    G
)

# Sharpen the blurred version
resharpened_test = unsharp_mask(
    blurred_test,
    2.0
)

save_color(
    "outputs/part2_1/experiment_original.jpg",
    selfie_color
)

save_color(
    "outputs/part2_1/experiment_blurred.jpg",
    blurred_test
)

save_color(
    "outputs/part2_1/experiment_resharpened.jpg",
    resharpened_test
)


print(
    f"Gaussian: {sharp_gaussian_size}x{sharp_gaussian_size}, "
    f"sigma={sharp_gaussian_sigma}"
)

print(
    "Sharpening alphas:",
    alphas
)

print(
    "Saved Part 2.1 results to outputs/part2_1/"
)

def run_hybrid_example(
    high_path,
    low_path,
    high_name,
    low_name,
    sigma_high,
    sigma_low,
    out_prefix
):
    high_img = ensure_rgb(io.imread(high_path))
    low_img = ensure_rgb(io.imread(low_path))

    print(f"\nAligning {high_name} (high) with {low_name} (low)")
    print(f"Select {high_name}'s two eyes first.")
    print(f"Then select {low_name}'s corresponding two eyes in the same order.")

    high_aligned, low_aligned = align_images(high_img, low_img)

    hybrid, low_freq, high_freq = hybrid_image(
        high_aligned,
        low_aligned,
        sigma_high,
        sigma_low
    )

    save_color(
        f"outputs/part2_2/{out_prefix}_{high_name.lower()}_aligned.jpg",
        high_aligned
    )

    save_color(
        f"outputs/part2_2/{out_prefix}_{low_name.lower()}_aligned.jpg",
        low_aligned
    )

    save_color(
        f"outputs/part2_2/{out_prefix}_{low_name.lower()}_low_frequency.jpg",
        low_freq
    )

    save_color(
        f"outputs/part2_2/{out_prefix}_{high_name.lower()}_high_frequency.jpg",
        normalize_signed_color(high_freq)
    )

    save_color(
        f"outputs/part2_2/{out_prefix}_hybrid.jpg",
        hybrid
    )

    far_view = cv2.resize(
        hybrid,
        None,
        fx=0.18,
        fy=0.18,
        interpolation=cv2.INTER_AREA
    )

    save_color(
        f"outputs/part2_2/{out_prefix}_far.jpg",
        far_view
    )

    print(
        f"{high_name} high-pass sigma = {sigma_high}"
    )
    print(
        f"{low_name} low-pass sigma = {sigma_low}"
    )
    print(
        f"Saved {out_prefix} results to outputs/part2_2/"
    )

# PART 2.2 — HYBRID IMAGES

print("\n--- PART 2.2: HYBRID IMAGES ---")

os.makedirs("outputs/part2_2", exist_ok=True)


# Derek = high frequencies
# Nutmeg = low frequencies

derek = ensure_rgb(
    io.imread("DerekPicture.jpg")
)

nutmeg = ensure_rgb(
    io.imread("nutmeg.jpg")
)

print("Select Derek's two eyes.")
print("Then select Nutmeg's corresponding two eyes in the same order.")


nutmeg_aligned, derek_aligned = align_images(
    nutmeg,
    derek
)

#nutmeg_aligned, derek_aligned = crop_to_common_valid_area(
#    nutmeg_aligned,
#    derek_aligned
#)


# Starting cutoff guesses
sigma_high = 2
sigma_low = 8


hybrid, nutmeg_low, derek_high = hybrid_image(
    derek_aligned,
    nutmeg_aligned,
    sigma_high,
    sigma_low
)

edward_batman = run_hybrid_pair(
    name="edward_batman",
    high_path="edward2.jpg",
    low_path="batman2.jpg",
    sigma_high=2,
    sigma_low=4
)

zendaya_mona = run_hybrid_pair(
    name="zendaya_mona_lisa",
    high_path="zendaya2.jpg",
    low_path="Mona_Lisa.jpg",
    sigma_high=5,
    sigma_low=4
)



save_color(
    "outputs/part2_2/derek_aligned.jpg",
    derek_aligned
)

save_color(
    "outputs/part2_2/nutmeg_aligned.jpg",
    nutmeg_aligned
)

save_color(
    "outputs/part2_2/nutmeg_low_frequency.jpg",
    nutmeg_low
)

save_color(
    "outputs/part2_2/derek_high_frequency.jpg",
    normalize_signed_color(derek_high)
)

save_color(
    "outputs/part2_2/derek_nutmeg_hybrid.jpg",
    hybrid
)


# Make a tiny version to simulate viewing from far away
far_view = cv2.resize(
    hybrid,
    None,
    fx=0.18,
    fy=0.18,
    interpolation=cv2.INTER_AREA
)

save_color(
    "outputs/part2_2/derek_nutmeg_far.jpg",
    far_view
)


print(
    f"Derek high-pass sigma = {sigma_high}"
)

print(
    f"Nutmeg low-pass sigma = {sigma_low}"
)

print(
    "Saved Derek + Nutmeg hybrid to outputs/part2_2/"
)

# PART 2.2 BELLS & WHISTLES: COLOR COMPARISON

derek_gray_rgb = ensure_rgb(
    color.rgb2gray(derek_aligned)
)

nutmeg_gray_rgb = ensure_rgb(
    color.rgb2gray(nutmeg_aligned)
)

# 1. Color in both components
hybrid_both_color, _, _ = hybrid_image(
    derek_aligned,
    nutmeg_aligned,
    sigma_high,
    sigma_low
)

# 2. Color only in low-frequency Nutmeg
hybrid_low_color, _, _ = hybrid_image(
    derek_gray_rgb,
    nutmeg_aligned,
    sigma_high,
    sigma_low
)

# 3. Color only in high-frequency Derek
hybrid_high_color, _, _ = hybrid_image(
    derek_aligned,
    nutmeg_gray_rgb,
    sigma_high,
    sigma_low
)

# 4. Fully grayscale baseline
hybrid_gray, _, _ = hybrid_image(
    derek_gray_rgb,
    nutmeg_gray_rgb,
    sigma_high,
    sigma_low
)

save_color(
    "outputs/part2_2/color_both.jpg",
    hybrid_both_color
)

save_color(
    "outputs/part2_2/color_low_only.jpg",
    hybrid_low_color
)

save_color(
    "outputs/part2_2/color_high_only.jpg",
    hybrid_high_color
)

save_color(
    "outputs/part2_2/color_grayscale.jpg",
    hybrid_gray
)

# PART 2.3 — GAUSSIAN AND LAPLACIAN STACKS

print("\n--- PART 2.3: GAUSSIAN AND LAPLACIAN STACKS ---")

os.makedirs(
    "outputs/part2_3",
    exist_ok=True
)

apple = ensure_rgb(
    io.imread("apple.jpeg")
)

orange = ensure_rgb(
    io.imread("orange.jpeg")
)

stack_levels = 5
stack_base_sigma = 1.0


# APPLE STACKS

apple_gaussian = gaussian_stack(
    apple,
    levels=stack_levels,
    base_sigma=stack_base_sigma
)

apple_laplacian = laplacian_stack(
    apple,
    levels=stack_levels,
    base_sigma=stack_base_sigma
)


# ORANGE STACKS

orange_gaussian = gaussian_stack(
    orange,
    levels=stack_levels,
    base_sigma=stack_base_sigma
)

orange_laplacian = laplacian_stack(
    orange,
    levels=stack_levels,
    base_sigma=stack_base_sigma
)

for i in range(stack_levels):

    save_color(
        f"outputs/part2_3/apple_gaussian_{i}.jpg",
        apple_gaussian[i]
    )

    save_color(
        f"outputs/part2_3/orange_gaussian_{i}.jpg",
        orange_gaussian[i]
    )

    # Last Laplacian level is already a normal low-frequency image.
    if i == stack_levels - 1:

        save_color(
            f"outputs/part2_3/apple_laplacian_{i}.jpg",
            apple_laplacian[i]
        )

        save_color(
            f"outputs/part2_3/orange_laplacian_{i}.jpg",
            orange_laplacian[i]
        )

    else:

        save_color(
            f"outputs/part2_3/apple_laplacian_{i}.jpg",
            normalize_signed_color(
                apple_laplacian[i]
            )
        )

        save_color(
            f"outputs/part2_3/orange_laplacian_{i}.jpg",
            normalize_signed_color(
                orange_laplacian[i]
            )
        )


# Verify that Laplacian levels reconstruct the original

apple_reconstructed = np.sum(
    np.stack(apple_laplacian),
    axis=0
)

orange_reconstructed = np.sum(
    np.stack(orange_laplacian),
    axis=0
)

print(
    "Apple reconstruction max error:",
    np.max(
        np.abs(
            apple - apple_reconstructed
        )
    )
)

print(
    "Orange reconstruction max error:",
    np.max(
        np.abs(
            orange - orange_reconstructed
        )
    )
)

print(
    f"Stack levels: {stack_levels}"
)

print(
    "Gaussian sigmas:",
    [
        0,
        *[
            stack_base_sigma * (2 ** (i - 1))
            for i in range(1, stack_levels)
        ]
    ]
)

print(
    "Saved Part 2.3 stacks to outputs/part2_3/"
)

def gaussian_stack_gray(image, levels=5, base_sigma=1.0):
    """
    Gaussian stack for a grayscale image / mask.
    No downsampling.
    """
    image = np.asarray(image, dtype=float)
    stack = [image]

    for level in range(1, levels):
        sigma = base_sigma * (2 ** (level - 1))
        kernel = gaussian_kernel_from_sigma(sigma)

        blurred = convolve2d(
            image,
            kernel,
            mode="same",
            boundary="symm"
        )

        stack.append(blurred)

    return stack


def multiresolution_blend(image_a, image_b, mask, levels=5, base_sigma=1.0):
    """
    image_a = selected where mask is white (1)
    image_b = selected where mask is black (0)
    """
    image_a = ensure_rgb(image_a)
    image_b = ensure_rgb(image_b)

    mask = np.asarray(mask, dtype=float)
    mask = np.clip(mask, 0, 1)

    lap_a = laplacian_stack(
        image_a,
        levels=levels,
        base_sigma=base_sigma
    )

    lap_b = laplacian_stack(
        image_b,
        levels=levels,
        base_sigma=base_sigma
    )

    mask_stack = gaussian_stack_gray(
        mask,
        levels=levels,
        base_sigma=base_sigma
    )

    masked_a_levels = []
    masked_b_levels = []
    blended_levels = []

    for i in range(levels):
        m = mask_stack[i][:, :, None]

        part_a = m * lap_a[i]
        part_b = (1.0 - m) * lap_b[i]
        blended = part_a + part_b

        masked_a_levels.append(part_a)
        masked_b_levels.append(part_b)
        blended_levels.append(blended)

    result = np.sum(
        np.stack(blended_levels),
        axis=0
    )

    return (
        np.clip(result, 0, 1),
        mask_stack,
        lap_a,
        lap_b,
        masked_a_levels,
        masked_b_levels,
        blended_levels
    )


def resize_and_center_crop(image, target_h, target_w):
    """
    Resize while preserving aspect ratio, then center crop
    to exactly (target_h, target_w).
    """
    image = ensure_rgb(image)
    h, w = image.shape[:2]

    scale = max(target_h / h, target_w / w)

    new_h = int(np.ceil(h * scale))
    new_w = int(np.ceil(w * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2

    cropped = resized[
        top:top + target_h,
        left:left + target_w
    ]

    return cropped


def match_to_reference(image, reference):
    """
    Match image to the reference image size using
    resize + center crop.
    """
    target_h, target_w = reference.shape[:2]
    return resize_and_center_crop(
        image,
        target_h,
        target_w
    )


def vertical_mask(h, w, split_col=None):
    """
    White on the left, black on the right.
    White selects image_a.
    """
    if split_col is None:
        split_col = w // 2

    mask = np.zeros((h, w), dtype=float)
    mask[:, :split_col] = 1.0
    return mask


def ellipse_mask(
    h,
    w,
    cx_frac=0.50,
    cy_frac=0.48,
    rx_frac=0.22,
    ry_frac=0.30
):
    """
    Elliptical irregular mask.
    White region selects image_a.
    """
    Y, X = np.ogrid[:h, :w]

    cx = cx_frac * w
    cy = cy_frac * h
    rx = rx_frac * w
    ry = ry_frac * h

    mask = (
        ((X - cx) / rx) ** 2
        +
        ((Y - cy) / ry) ** 2
        <= 1
    ).astype(float)

    return mask


def traced_mask(h, w, outline, smoothing=2.0, samples=400):
    """
    Irregular mask from an outline traced around a shape.

    outline: (x, y) points as fractions of the width / height,
    listed in order around the shape. A closed spline through the
    points smooths out the corners before the shape is filled.
    White region selects image_a.
    """
    pts = np.array(outline, dtype=float) * [w, h]

    tck, _ = splprep(
        [pts[:, 0], pts[:, 1]],
        s=len(pts) * smoothing,
        per=True
    )

    xs, ys = splev(np.linspace(0, 1, samples), tck)
    contour = np.round(np.stack([xs, ys], axis=1)).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)

    return mask.astype(float)


def place_onto(image, mask, src_points, dst_points, tilt_deg=0.0):
    """
    Resize, tilt and move image (and its mask) so src_points land on
    dst_points as closely as possible. The tilt is chosen by hand;
    the scale and shift are then the least-squares fit for that tilt.
    Negative tilt turns the image counter-clockwise on screen.
    Points are (x, y) fractions of the width / height.
    """
    h, w = image.shape[:2]

    src = np.array(src_points, dtype=float) * [w, h]
    dst = np.array(dst_points, dtype=float) * [w, h]

    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)

    theta = np.radians(tilt_deg)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated = (src - src_center) @ rotation.T

    # least-squares scale between the two point sets
    scale = np.sqrt(
        np.sum((dst - dst_center) ** 2)
        /
        np.sum(rotated ** 2)
    )

    transform = sktr.SimilarityTransform(
        scale=scale,
        rotation=theta,
        translation=dst_center - scale * (rotation @ src_center)
    )

    moved = sktr.warp(
        image,
        inverse_map=transform.inverse,
        output_shape=(h, w),
        mode="edge",
        preserve_range=True
    )

    moved_mask = sktr.warp(
        mask,
        inverse_map=transform.inverse,
        output_shape=(h, w),
        order=1,
        preserve_range=True
    ) > 0.5

    return moved, moved_mask.astype(float), transform


def round_off_bottom(mask, cut_row, radius):
    """
    Cut mask off flat at cut_row, with the two bottom corners rounded
    (quarter circles of the given radius), so the sides curve into the
    flat edge instead of meeting it at a sharp corner.
    """
    h, w = mask.shape
    mask = mask.copy()
    mask[cut_row:, :] = 0.0

    # how wide the mask is just above the cut
    band = mask[cut_row - radius:cut_row] > 0.5
    cols = np.nonzero(band.any(axis=0))[0]
    left, right = cols.min(), cols.max()

    Y, X = np.mgrid[:h, :w]
    corner_y = cut_row - radius
    in_band = (Y >= corner_y) & (Y < cut_row)

    for cx, side in (
        (left + radius, X < left + radius),
        (right - radius, X > right - radius)
    ):
        outside = (X - cx) ** 2 + (Y - corner_y) ** 2 > radius ** 2
        mask[in_band & side & outside] = 0.0

    return mask


def replace_background(image, silhouette, background, grow=3, sigma=2.0):
    """
    Keep image inside silhouette and use background everywhere else.
    The silhouette is grown by a few pixels and feathered with a small
    Gaussian, so the cut itself doesn't add a hard edge.
    """
    kernel = np.ones((2 * grow + 1, 2 * grow + 1), dtype=np.uint8)
    grown = cv2.dilate(silhouette.astype(np.uint8), kernel).astype(float)

    soft = convolve2d(
        grown,
        gaussian_kernel_from_sigma(sigma),
        mode="same",
        boundary="symm"
    )

    return soft[:, :, None] * image + (1.0 - soft[:, :, None]) * background


def blend_and_save(image_a, image_b, mask, prefix, levels=5, base_sigma=1.0, debug=True):
    """
    Hard-mask composite + multiresolution blend of image_a (white part
    of the mask) over image_b, saved as <prefix>_mask / _hard_mask /
    _final, plus the per-level images when debug is True.
    """
    hard = (
        mask[:, :, None] * image_a
        +
        (1.0 - mask[:, :, None]) * image_b
    )

    save_gray(f"{prefix}_mask.jpg", mask)
    save_color(f"{prefix}_hard_mask.jpg", hard)

    (
        final,
        mask_stack,
        lap_a,
        lap_b,
        a_masked,
        b_masked,
        blended_levels
    ) = multiresolution_blend(
        image_a,
        image_b,
        mask,
        levels=levels,
        base_sigma=base_sigma
    )

    save_color(f"{prefix}_final.jpg", final)

    if debug:
        save_blend_debug(
            prefix,
            mask_stack,
            a_masked,
            b_masked,
            blended_levels
        )

    return final


def save_blend_debug(
    prefix,
    mask_stack,
    masked_a_levels,
    masked_b_levels,
    blended_levels
):
    """
    Save the multiresolution process for webpage/report.
    """
    levels = len(mask_stack)

    for i in range(levels):
        save_gray(
            f"{prefix}_mask_level_{i}.jpg",
            mask_stack[i]
        )

        if i == levels - 1:
            save_color(
                f"{prefix}_A_masked_level_{i}.jpg",
                masked_a_levels[i]
            )
            save_color(
                f"{prefix}_B_masked_level_{i}.jpg",
                masked_b_levels[i]
            )
            save_color(
                f"{prefix}_blended_level_{i}.jpg",
                blended_levels[i]
            )
        else:
            save_color(
                f"{prefix}_A_masked_level_{i}.jpg",
                normalize_signed_color(masked_a_levels[i])
            )
            save_color(
                f"{prefix}_B_masked_level_{i}.jpg",
                normalize_signed_color(masked_b_levels[i])
            )
            save_color(
                f"{prefix}_blended_level_{i}.jpg",
                normalize_signed_color(blended_levels[i])
            )

# PART 2.4 — MULTIRESOLUTION BLENDING

print("\n--- PART 2.4: MULTIRESOLUTION BLENDING ---")

os.makedirs(
    "outputs/part2_4",
    exist_ok=True
)
blend_levels = 5
blend_base_sigma = 1.0


# ORAPLE MASK

h, w = apple.shape[:2]

mask = np.zeros(
    (h, w),
    dtype=float
)

# Left half = apple
# Right half = orange
mask[:, :w // 2] = 1.0


# SIMPLE HARD-SEAM RESULT

hard_seam = (
    mask[:, :, None] * apple
    +
    (1.0 - mask[:, :, None]) * orange
)

save_color(
    "outputs/part2_4/oraple_hard_seam.jpg",
    hard_seam
)

save_gray(
    "outputs/part2_4/oraple_mask.jpg",
    mask
)

(
    oraple,
    mask_stack,
    apple_lap,
    orange_lap,
    apple_masked_levels,
    orange_masked_levels,
    blended_levels
) = multiresolution_blend(
    apple,
    orange,
    mask,
    levels=5,
    base_sigma=1.0
)


save_color(
    "outputs/part2_4/oraple_final.jpg",
    oraple
)


for i in range(5):

    # Gaussian mask at this scale
    save_gray(
        f"outputs/part2_4/mask_gaussian_{i}.jpg",
        mask_stack[i]
    )

    # Last Laplacian level is low-frequency residual
    if i == 4:

        save_color(
            f"outputs/part2_4/apple_masked_level_{i}.jpg",
            apple_masked_levels[i]
        )

        save_color(
            f"outputs/part2_4/orange_masked_level_{i}.jpg",
            orange_masked_levels[i]
        )

        save_color(
            f"outputs/part2_4/blended_level_{i}.jpg",
            blended_levels[i]
        )

    else:

        save_color(
            f"outputs/part2_4/apple_masked_level_{i}.jpg",
            normalize_signed_color(
                apple_masked_levels[i]
            )
        )

        save_color(
            f"outputs/part2_4/orange_masked_level_{i}.jpg",
            normalize_signed_color(
                orange_masked_levels[i]
            )
        )

        save_color(
            f"outputs/part2_4/blended_level_{i}.jpg",
            normalize_signed_color(
                blended_levels[i]
            )
        )


print(
    "Saved Oraple and multiresolution stack visualization "
    "to outputs/part2_4/"
)



# 2.4B — CUSTOM BLEND 1
# Desperate Man + Get Out (vertical seam)

desperate_blend = ensure_rgb(io.imread("desperateman.jpg"))
getout_blend = ensure_rgb(io.imread("getout.jpg"))

getout_blend, _ = align_to_reference(
    getout_blend,
    desperate_blend
)

h, w = desperate_blend.shape[:2]

# Left = Desperate Man, Right = Get Out
desperate_getout_mask = vertical_mask(h, w)

desperate_getout_hard = (
    desperate_getout_mask[:, :, None] * desperate_blend
    +
    (1.0 - desperate_getout_mask[:, :, None]) * getout_blend
)

save_gray(
    "outputs/part2_4/desperate_getout_mask.jpg",
    desperate_getout_mask
)

save_color(
    "outputs/part2_4/desperate_getout_hard_seam.jpg",
    desperate_getout_hard
)

(
    desperate_getout_final,
    desperate_getout_mask_stack,
    desperate_getout_lap_a,
    desperate_getout_lap_b,
    desperate_getout_a_masked,
    desperate_getout_b_masked,
    desperate_getout_blended_levels
) = multiresolution_blend(
    desperate_blend,
    getout_blend,
    desperate_getout_mask,
    levels=blend_levels,
    base_sigma=blend_base_sigma
)

save_color(
    "outputs/part2_4/desperate_getout_final.jpg",
    desperate_getout_final
)

save_blend_debug(
    "outputs/part2_4/desperate_getout",
    desperate_getout_mask_stack,
    desperate_getout_a_masked,
    desperate_getout_b_masked,
    desperate_getout_blended_levels
)


# Edward + Batman (irregular mask)

batman_blend = ensure_rgb(io.imread("batman.jpg"))
edward_blend = ensure_rgb(io.imread("edward.jpg"))

batman_blend, _ = align_to_reference(
    batman_blend,
    edward_blend
)

h, w = edward_blend.shape[:2]

# First attempt: an ellipse around his head. It's wider than his head,
# so his hair and a ring of teal background come in with him.
edward_batman_oval = ellipse_mask(
    h,
    w,
    cx_frac=0.50,
    cy_frac=0.45,
    rx_frac=0.24,
    ry_frac=0.34
)

blend_and_save(
    edward_blend,   # inside white mask
    batman_blend,   # outside mask
    edward_batman_oval,
    "outputs/part2_4/edward_batman_oval",
    levels=blend_levels,
    base_sigma=blend_base_sigma,
    debug=False
)

# Final version: outline of Edward in edward.jpg, going clockwise from
# the top of his hair. It covers the inner part of his hair, his face,
# neck, collar and the top of his jacket out to the shoulder seams
# (the jacket gets cut shorter below, after he's placed).
# The wild outer hair and the teal background stay outside, so Batman's
# cowl ears still rise above him.
edward_outline = [
    (0.526, 0.060), (0.580, 0.065), (0.620, 0.085), (0.640, 0.130),
    (0.645, 0.200), (0.635, 0.270), (0.622, 0.333), (0.615, 0.400),
    (0.611, 0.457), (0.619, 0.495), (0.650, 0.525), (0.690, 0.550),
    (0.730, 0.575), (0.760, 0.605), (0.775, 0.650), (0.770, 0.705),
    (0.745, 0.760), (0.700, 0.800), (0.620, 0.825), (0.530, 0.830),
    (0.440, 0.822), (0.360, 0.795), (0.305, 0.750), (0.280, 0.695),
    (0.285, 0.645), (0.315, 0.610), (0.355, 0.580), (0.395, 0.535),
    (0.415, 0.495), (0.411, 0.429), (0.379, 0.381), (0.361, 0.314),
    (0.354, 0.238), (0.365, 0.162), (0.401, 0.100), (0.454, 0.068),
]

# The blend mask is the same outline with the four points along his
# left cheek pushed out ~10 px. Right on the outline, the coarse levels
# of the blend are only half Edward, which blurred the side of his face.
blend_outline = list(edward_outline)
blend_outline[5:9] = [(0.642, 0.270), (0.638, 0.333), (0.632, 0.400), (0.622, 0.457)]

# Placement. Edward's head leans about 8 degrees counter-clockwise and
# Batman's leans about 6 degrees clockwise, so Edward gets tilted 13
# degrees clockwise. His eyes and the middle of his mouth are then fit
# onto target points inside the cowl. The targets start from Batman's
# own eyes and mouth, moved 28 px left and pulled 5% closer together:
# Batman's head is turned slightly, so his eyes sit right of the middle
# of the cowl, and matching them exactly pushed Edward's cheek and hair
# out past the cowl into the sky. (Fractions of width / height.)
edward_landmarks = [(0.465, 0.322), (0.578, 0.300), (0.535, 0.452)]
batman_eyes_mouth = np.array([(0.487, 0.366), (0.575, 0.380), (0.523, 0.505)])

shifted = batman_eyes_mouth + [-28 / w, 0]
eye_middle = shifted[:2].mean(axis=0)
cowl_landmarks = eye_middle + 0.95 * (shifted - eye_middle)

edward_placed, edward_batman_mask, edward_move = place_onto(
    edward_blend,
    traced_mask(h, w, blend_outline),
    edward_landmarks,
    cowl_landmarks,
    tilt_deg=13.0
)

_, edward_silhouette, _ = place_onto(
    edward_blend,
    traced_mask(h, w, edward_outline),
    edward_landmarks,
    cowl_landmarks,
    tilt_deg=13.0
)

# Cut the jacket off flat, about halfway between his chin and the bottom
# of the traced jacket, with rounded corners so the shoulders curve into
# the flat edge.
jacket_cut = int(round(0.72 * h))
edward_batman_mask = round_off_bottom(edward_batman_mask, jacket_cut, radius=50)
edward_silhouette = round_off_bottom(edward_silhouette, jacket_cut, radius=50)

# Swap the teal Forks background around Edward for the Batman image
# before blending, so the soft edge of the blend pulls in Batman's sky
# and cowl instead of a teal glow.
edward_placed = replace_background(
    edward_placed,
    edward_silhouette,
    batman_blend
)

print(
    f"Edward placed onto Batman: scale {edward_move.scale:.3f}, "
    f"tilt {np.degrees(edward_move.rotation):.1f} deg, "
    f"shift {np.round(edward_move.translation, 1)} px"
)

edward_batman_final = blend_and_save(
    edward_placed,   # inside white mask
    batman_blend,    # outside mask
    edward_batman_mask,
    "outputs/part2_4/edward_batman",
    levels=blend_levels,
    base_sigma=blend_base_sigma
)

print("Saved Part 2.4 results to outputs/part2_4/")