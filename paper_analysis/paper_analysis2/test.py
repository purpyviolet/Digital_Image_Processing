import numpy as np
import cv2


def is_salt_pepper_noise(image, i, j, H):
    # 这里是一个示例判断条件，你可能需要根据实际情况进行调整
    pixel_value = image[i, j]
    return pixel_value >= 255 - H or pixel_value <= 0 + H


def pixel_median_filter(image, i, j, H):
    kernel_size = 3  # 使用3x3的滤波核
    half_kernel = kernel_size // 2
    rows, cols = image.shape
    pixel_values = []

    for k in range(-half_kernel, half_kernel + 1):
        for l in range(-half_kernel, half_kernel + 1):
            if 0 <= i + k < rows and 0 <= j + l < cols and not is_salt_pepper_noise(image, i + k, j + l, H):
                pixel_values.append(image[i + k, j + l])

    if len(pixel_values) == 0:
        return image[i, j]
    else:
        return np.median(pixel_values)


def image_filter(image, H):
    filtered_image = image.copy()
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            if is_salt_pepper_noise(image, i, j, H):
                filtered_image[i, j] = pixel_median_filter(image, i, j, H)

    return filtered_image


# 示例用法
image = cv2.imread('original_grey_image.png', cv2.IMREAD_GRAYSCALE)
H = 20  # 一个示例阈值，用于判断椒盐噪声，需要根据实际情况调整
filtered_image = image_filter(image, H)

cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
