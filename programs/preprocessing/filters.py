import cv2
import numpy as np


def _to_gray(image):
    """Convert BGR image to grayscale, or return as-is if already grayscale."""
    if image is None:
        raise ValueError("Input image is None")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binary_threshold(image):
    """
    Convert image to grayscale and apply binary threshold.
    Simple approach; works well when text has high contrast with background.
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        Processed image with text isolated in black and white
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def adaptive_threshold(image):
    """
    Apply adaptive thresholding to handle varying lighting conditions.
    Better than regular threshold for uneven illumination across the image.
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        Processed image with adaptive thresholding applied
    """
    gray = _to_gray(image)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    return adaptive


def adaptive_threshold_inv(image, block_size=35, c=10):
    """
    Apply adaptive thresholding with inverted binary output.

    Args:
        image: Input image (BGR or grayscale)
        block_size: Odd local neighborhood size for adaptive threshold
        c: Constant subtracted from local weighted mean

    Returns:
        Inverted adaptive threshold image
    """
    gray = _to_gray(image)
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def canny_edge_detection(image):
    """
    Detect edges using Canny edge detection algorithm.
    Highlights text boundaries and shapes clearly.
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        Image showing detected edges
    """
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def morphological_operations(image):
    """
    Apply morphological operations (erode/dilate) to clean up noise.
    Preserves text structure while removing small noise artifacts.
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        Processed image with morphological operations applied
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Apply erosion to remove noise, then dilation to restore text
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph


def erosion_filter(image, kernel_size=(3, 3), iterations=1):
    """
    Apply erosion to shrink bright regions and reduce small white noise.

    Args:
        image: Input image (BGR format from OpenCV)
        kernel_size: Size of the rectangular structuring element
        iterations: Number of erosion passes

    Returns:
        Eroded binary image
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded = cv2.erode(binary, kernel, iterations=iterations)
    return eroded


def dilation_filter(image, kernel_size=(3, 3), iterations=1):
    """
    Apply dilation to expand bright regions and strengthen thin text strokes.

    Args:
        image: Input image (BGR format from OpenCV)
        kernel_size: Size of the rectangular structuring element
        iterations: Number of dilation passes

    Returns:
        Dilated binary image
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    return dilated


def erosion_dilation_filter(image, kernel_size=(3, 3), erosion_iterations=1, dilation_iterations=1):
    """
    Apply erosion followed by dilation using configurable parameters.

    Args:
        image: Input image (BGR format from OpenCV)
        kernel_size: Size of the rectangular structuring element
        erosion_iterations: Number of erosion passes
        dilation_iterations: Number of dilation passes

    Returns:
        Binary image after erosion then dilation
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded = cv2.erode(binary, kernel, iterations=erosion_iterations)
    combined = cv2.dilate(eroded, kernel, iterations=dilation_iterations)
    return combined


def hsv_color_filter(image, lower_hsv=(0, 0, 100), upper_hsv=(180, 50, 255)):
    """
    Filter image by HSV color range to isolate specific colored text.
    Default range targets light colors (white/light gray text).
    Adjust lower_hsv and upper_hsv tuples to target different colors.
    
    Args:
        image: Input image (BGR format from OpenCV)
        lower_hsv: Lower HSV bound (H, S, V) - default targets light colors
        upper_hsv: Upper HSV bound (H, S, V) - default targets light colors
    
    Returns:
        Mask highlighting pixels within the specified HSV range
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    return mask


def bilateral_filter_threshold(image):
    """
    Apply bilateral filter to smooth and preserve edges, then threshold.
    Reduces noise while keeping text sharp and well-defined.
    
    Args:
        image: Input image (BGR format from OpenCV)
    
    Returns:
        Processed image with bilateral filtering and thresholding applied
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, result = cv2.threshold(bilateral, 127, 255, cv2.THRESH_BINARY)
    return result


def gaussian_blur_filter(image, kernel_size=(5, 5), sigma_x=0):
    """
    Apply Gaussian blur to smooth the image using neighboring pixels.

    Args:
        image: Input image (BGR or grayscale)
        kernel_size: Odd-valued kernel dimensions (width, height)
        sigma_x: Gaussian kernel standard deviation in X direction

    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def clahe_filter(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Improves local contrast and can make text stand out in uneven lighting.

    Args:
        image: Input image (BGR format from OpenCV)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced grayscale image
    """
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def gamma_correction(image, gamma=1.5):
    """
    Apply gamma correction to adjust overall image brightness.
    Gamma > 1 darkens mid-tones, gamma < 1 brightens mid-tones.

    Args:
        image: Input image (BGR or grayscale)
        gamma: Gamma value; must be greater than 0

    Returns:
        Gamma-corrected image in the same channel format as input
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def gamma_darken_filter(image, gamma=1.6):
    """
    Darken mid-tones using direct gamma exponent (gamma > 1 darkens).

    Args:
        image: Input image (BGR or grayscale)
        gamma: Gamma value; must be greater than 0

    Returns:
        Gamma-darkened grayscale image
    """
    gray = _to_gray(image)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def black_hat_transform_filter(image, kernel_size=(15, 15)):
    """
    Apply Black Hat morphology to isolate dark structures on bright background.

    Args:
        image: Input image (BGR or grayscale)
        kernel_size: Rectangular kernel size for Black Hat transform

    Returns:
        Black Hat response image
    """
    gray = _to_gray(image)
    kx = max(3, int(kernel_size[0]))
    ky = max(3, int(kernel_size[1]))
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def otsu_binarization_filter(image):
    """
    Apply global Otsu thresholding to produce a binary image.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Otsu-binarized image
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def inverse_filter(image):
    """
    Invert image intensities so dark pixels become light and vice versa.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Inverted image
    """
    return cv2.bitwise_not(image)


def high_contrast_transition_filter(image, gradient_threshold=45, value_threshold=127):
    """
    Keep only pixels near strong light-dark transitions and force them to black/white.

    Args:
        image: Input image (BGR or grayscale)
        gradient_threshold: Minimum local intensity-change magnitude to keep
        value_threshold: Split point for assigning absolute black or white

    Returns:
        Binary image where strong-transition pixels are 0/255 and other pixels are 0
    """
    gray = _to_gray(image)

    # Estimate local change magnitude from x/y gradients.
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    transition_mask = magnitude >= float(gradient_threshold)
    result = np.zeros_like(gray, dtype=np.uint8)
    result[transition_mask & (gray >= value_threshold)] = 255
    result[transition_mask & (gray < value_threshold)] = 0
    return result


def contour_geometry_cleanup_filter(
    image,
    min_area=40,
    max_width_ratio=0.70,
    max_line_height_ratio=0.25,
):
    """
    Remove contour noise based on simple geometry constraints.

    This targets:
    - tiny speckles (very small contour area)
    - long bezel-like horizontal bands (very wide and relatively short)

    Args:
        image: Binary/grayscale image (non-zero treated as foreground)
        min_area: Minimum contour area to keep
        max_width_ratio: Width ratio (to image width) above which contour is treated as too wide
        max_line_height_ratio: Height ratio (to image height) below which wide contours are removed

    Returns:
        Cleaned binary image with only kept contours filled white
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    height, width = binary.shape[:2]
    max_w = width * float(max_width_ratio)
    max_line_h = height * float(max_line_height_ratio)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        is_wide_line = (w >= max_w) and (h <= max_line_h)
        if is_wide_line:
            continue

        cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)

    return cleaned


def straight_diagonal_lines_filter(
    image,
    canny_low=50,
    canny_high=150,
    hough_threshold=60,
    min_line_length=30,
    max_line_gap=10,
    angle_tolerance=20,
):
    """
    Keep only prominent straight/diagonal lines using Canny + Hough transform.

    Args:
        image: Input image (BGR or grayscale)
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
        hough_threshold: Votes required in Hough accumulator
        min_line_length: Minimum accepted line segment length
        max_line_gap: Maximum gap to link line segments
        angle_tolerance: Degrees around 0, 45, 90, 135 to keep

    Returns:
        Binary mask where detected straight/diagonal lines are white
    """
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    result = np.zeros_like(gray)
    if lines is None:
        return result

    target_angles = (0, 45, 90, 135)
    for segment in lines[:, 0]:
        x1, y1, x2, y2 = segment
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if any(abs(angle - target) <= angle_tolerance for target in target_angles):
            cv2.line(result, (x1, y1), (x2, y2), 255, 2)

    return result


def lcd_digit_pipeline_filter(image):
    """
    Apply a fixed 5-step pipeline tuned for dark 7-segment LCD digits.

    Steps:
    1) Gaussian blur (5x5)
    2) Adaptive threshold (Gaussian, Binary INV, block=35, C=10)
    3) Contour geometry cleanup (remove tiny + wide bezel-like contours)
    4) Morphological opening (3x3)
    5) Morphological closing (9x9)
    6) Horizontal dilation (25x3)
    + light trim erosion (3x3)

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Binary mask emphasizing assembled LCD digit regions
    """
    gray = _to_gray(image)

    # Step 1: remove high-frequency background grain
    step1 = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: localized inverted binarization for dark segments
    step2 = cv2.adaptiveThreshold(
        step1,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
    )

    # Step 3: contour filtering to remove bezel-like bands and tiny speckles.
    step3 = contour_geometry_cleanup_filter(step2)

    # Step 4: remove residual salt noise
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    step4 = cv2.morphologyEx(step3, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Step 5: connect broken segments into solid digit blobs
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    step5 = cv2.morphologyEx(step4, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Step 6: bridge horizontal gaps between neighboring digits
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    step6 = cv2.dilate(step5, dilate_kernel, iterations=1)

    # Final trim: reduce blob thickness while preserving assembled structure.
    trim_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    step7 = cv2.erode(step6, trim_kernel, iterations=1)

    return step7


def gamma_blackhat_otsu_filter(image, gamma=1.6, blackhat_kernel=(15, 15)):
    """
    Extract dark text-like structures using gamma darkening + black hat + Otsu.

    Steps:
    1) Gamma correction with gamma > 1 to darken mid-tones.
    2) Black Hat morphology to isolate dark structures on bright background.
    3) Global Otsu threshold to produce a clean binary mask.

    Args:
        image: Input image (BGR or grayscale)
        gamma: Gamma value (>1 darkens mid-tones)
        blackhat_kernel: Rectangular kernel for Black Hat transform

    Returns:
        Binary mask with emphasized segment-like structures
    """
    step1 = gamma_darken_filter(image, gamma=gamma)
    step2 = black_hat_transform_filter(step1, kernel_size=blackhat_kernel)
    step3 = otsu_binarization_filter(step2)
    return step3
