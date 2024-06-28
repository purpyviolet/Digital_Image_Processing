import cv2
import numpy as np


def piecewise_linear_transformation(image_path, threshold, lower_slope, upper_slope):
    """
    应用分段线性灰度变换。
    :param image_path: 图像的路径。
    :param threshold: 分割暗区和亮区的阈值。
    :param lower_slope: 阈值以下部分的斜率。
    :param upper_slope: 阈值以上部分的斜率。
    """
    # 读取图片
    image = cv2.imread(image_path)

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用分段线性变换
    # 初始化输出图像
    transformed_image = np.zeros_like(gray_image, dtype=np.float32)

    # 应用斜率
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i, j] < threshold:
                transformed_image[i, j] = np.clip(lower_slope * gray_image[i, j], 0, 255)
            else:
                transformed_image[i, j] = np.clip(
                    lower_slope * threshold + upper_slope * (gray_image[i, j] - threshold), 0, 255)

    # 将浮点结果转换回uint8
    transformed_image = np.uint8(transformed_image)

    # 显示原始灰度图像和变换后的图像
    cv2.imshow('Original Gray Image', gray_image)
    cv2.imshow('Piecewise Linear Transformation Image', transformed_image)

    # 等待键盘输入，之后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 调用函数
piecewise_linear_transformation('boy.png', threshold=128, lower_slope=0.5, upper_slope=2)

