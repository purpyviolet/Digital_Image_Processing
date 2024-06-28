import cv2
import numpy as np
import matplotlib.pyplot as plt


def median_filter(image, kernel_size, mode='rectangular'):
    if mode == 'rectangular':
        return cv2.medianBlur(image, kernel_size)
    elif mode == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError("Invalid mode. Use 'rectangular' or 'cross'.")


def main():
    # 读取彩色图片
    color_img = cv2.imread('original_RGB_image.png')

    # 读取灰度图片
    gray_img = cv2.imread('original_grey_image.png', cv2.IMREAD_GRAYSCALE)

    # 对彩色图片进行5x5原形窗口和十字窗口中值滤波
    color_filtered_circle = median_filter(color_img, 5, 'rectangular')
    color_filtered_cross = median_filter(color_img, 5, 'cross')

    # 对灰度图片进行5x5原形窗口和十字窗口中值滤波
    gray_filtered_circle = median_filter(gray_img, 5, 'rectangular')
    gray_filtered_cross = median_filter(gray_img, 5, 'cross')

    # 展示和保存彩色图片处理结果
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 彩色图片对比
    axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Color Image')

    axes[0, 1].imshow(cv2.cvtColor(color_filtered_circle, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('5x5 Circular Median Filtered Color Image')

    axes[1, 0].imshow(cv2.cvtColor(color_filtered_cross, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('5x5 Cross Median Filtered Color Image')

    # 禁用灰度图片的坐标轴
    for ax in axes[:, 0]:
        ax.axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存彩色处理结果
    plt.savefig('median_filtered_color_images_comparison.png')

    # 展示彩色处理结果
    plt.show()

    # 展示和保存灰度图片处理结果
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 灰度图片对比
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title('Original Grayscale Image')

    axes[0, 1].imshow(gray_filtered_circle, cmap='gray')
    axes[0, 1].set_title('5x5 Circular Median Filtered Grayscale Image')

    axes[1, 0].imshow(gray_filtered_cross, cmap='gray')
    axes[1, 0].set_title('5x5 Cross Median Filtered Grayscale Image')

    # 禁用彩色图片的坐标轴
    for ax in axes[:, 0]:
        ax.axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存灰度处理结果
    plt.savefig('median_filtered_gray_images_comparison.png')

    # 展示灰度处理结果
    plt.show()


if __name__ == "__main__":
    main()
