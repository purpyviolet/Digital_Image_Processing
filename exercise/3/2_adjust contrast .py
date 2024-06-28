import cv2
import numpy as np

def adjust_contrast(image_path, alpha, beta):
    """
    调整图像对比度和亮度。
    :param image_path: 图像路径
    :param alpha: 对比度控制（1.0 = 原始对比度，<1.0 = 降低对比度，>1.0 = 提高对比度）
    :param beta: 亮度调整（0 = 没有改变，正值增加亮度，负值减少亮度）
    """
    # 读取图片
    image = cv2.imread(image_path)

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调整对比度和亮度
    adjusted = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

    # 显示原始灰度图像和调整后的图像
    cv2.imshow('Original Gray Image', gray_image)
    cv2.imshow('Adjusted Contrast Image', adjusted)

    # 等待键盘输入，之后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用函数
adjust_contrast('boy.png', alpha=2.0, beta=50)
