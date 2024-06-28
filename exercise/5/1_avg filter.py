import cv2
import numpy as np
import matplotlib.pyplot as plt


def mean_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


def main():
    # 读取彩色图片
    color_img = cv2.imread('original_RGB_image.jpg')

    # 读取灰度图片
    gray_img = cv2.imread('original_grey_image.jpg', cv2.IMREAD_GRAYSCALE)

    # 对彩色图片进行3x3和5x5均值滤波
    color_filtered_3x3 = mean_filter(color_img, 3)
    color_filtered_5x5 = mean_filter(color_img, 5)

    # 对灰度图片进行3x3和5x5均值滤波
    gray_filtered_3x3 = mean_filter(gray_img, 3)
    gray_filtered_5x5 = mean_filter(gray_img, 5)

    # 展示和保存彩色图片处理结果
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 彩色图片对比
    axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Color Image')

    axes[0, 1].imshow(cv2.cvtColor(color_filtered_3x3, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('3x3 Mean Filtered Color Image')

    axes[1, 0].imshow(cv2.cvtColor(color_filtered_5x5, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('5x5 Mean Filtered Color Image')

    # 禁用灰度图片的坐标轴
    for ax in axes[:, 0]:
        ax.axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存彩色处理结果
    plt.savefig('mean_filtered_color_images_comparison.png')

    # 展示彩色处理结果
    plt.show()

    # 展示和保存灰度图片处理结果
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 灰度图片对比
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title('Original Grayscale Image')

    axes[0, 1].imshow(gray_filtered_3x3, cmap='gray')
    axes[0, 1].set_title('3x3 Mean Filtered Grayscale Image')

    axes[1, 0].imshow(gray_filtered_5x5, cmap='gray')
    axes[1, 0].set_title('5x5 Mean Filtered Grayscale Image')

    # 禁用彩色图片的坐标轴
    for ax in axes[:, 0]:
        ax.axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存灰度处理结果
    plt.savefig('mean_filtered_gray_images_comparison.png')

    # 展示灰度处理结果
    plt.show()


if __name__ == "__main__":
    main()
