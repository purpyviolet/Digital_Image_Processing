import cv2
import numpy as np
import matplotlib.pyplot as plt



def image_processing(file_path):
    # 加载图像（确保路径正确）
    image = cv2.imread(file_path)
    # 保存原图像到文件
    cv2.imwrite('static/original_image.jpg', image)
    # 将图像从BGR色彩空间转换到HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 保存处理后的HSV图像到文件
    cv2.imwrite('static/hsv_image.jpg', hsv_image)

    # 显示原始图像和HSV图像（仅用于验证，实际应用中可能不需要）
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB))
    plt.title('HSV Image')
    plt.axis('off')

    plt.show()

    # 假设 hsv_image 是你已经有的 HSV 格式图像
    # 分离HSV图像的三个通道
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    # 保存分离出的通道图像到文件
    cv2.imwrite('static/h_image.jpg', h_channel)
    cv2.imwrite('static/s_image.jpg', s_channel)
    cv2.imwrite('static/v_image.jpg', v_channel)

    # 显示每个通道的图像
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(h_channel, cmap='gray')
    plt.title('H Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(s_channel, cmap='gray')
    plt.title('S Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(v_channel, cmap='gray')
    plt.title('V Channel')
    plt.axis('off')

    plt.show()

    # 对V通道进行直方图均衡化
    equalized_v_channel = cv2.equalizeHist(v_channel)

    # 对均衡化后的V通道进行颜色平滑处理
    # equalized_v_channel = cv2.GaussianBlur(equalized_v_channel, (5, 5), 0.5)

    cv2.imwrite('static/equalized_v_image.jpg', equalized_v_channel)

    # 显示均衡化前后的V通道图像（仅用于验证，实际应用中可能不需要）
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(v_channel, cmap='gray')
    plt.title('Original V Channel')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_v_channel, cmap='gray')
    plt.title('Equalized V Channel')
    plt.axis('off')

    plt.show()

    # 假设 h_channel, s_channel 是你已经分离出的 H 和 S 通道图像
    # equalized_v_channel 是均衡化后的 V 通道图像
    # 将均衡化后的V通道重新合并到HSV图像中
    hsv_image_equalized = cv2.merge([h_channel, s_channel, equalized_v_channel])

    # 将HSV图像转换回RGB格式
    rgb_image_equalized = cv2.cvtColor(hsv_image_equalized, cv2.COLOR_HSV2BGR)

    # 保存处理后的RGB图像到文件
    cv2.imwrite('static/equalized_rgb_image.jpg', rgb_image_equalized)

    # 显示处理前后的RGB图像（仅用于验证，实际应用中可能不需要）
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original RGB Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rgb_image_equalized, cv2.COLOR_BGR2RGB))
    plt.title('Equalized RGB Image')
    plt.axis('off')

    plt.show()

file_path = 'test.png'

# image_processing('game of thrones.jpg')