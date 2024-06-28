import cv2
import numpy as np

def apply_gamma_transformation(image_path, gamma):
    """
    应用幂次转换以调整图像亮度。
    :param image_path: 图像的路径。
    :param gamma: 伽马值，用于控制亮度调整的非线性程度。
    """
    # 读取图片
    image = cv2.imread(image_path)

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 首先将图像像素值范围从0-255转换到0-1
    gray_normalized = gray_image / 255.0

    # 应用幂次转换
    gamma_corrected = np.power(gray_normalized, gamma)

    # 再将结果缩放回0-255
    gamma_corrected = np.uint8(gamma_corrected * 255)

    # 显示原始灰度图像和幂次转换后的图像
    cv2.imshow('Original Gray Image', gray_image)
    cv2.imshow('Gamma Corrected Image', gamma_corrected)

    # 等待键盘输入，之后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用函数
apply_gamma_transformation('boy.png', gamma=2.0)

# 当 γ>1 时：随着
# γ 的增大，图像的暗部会变得更暗，而亮部的变化不那么显著。这种效果会使得图像中的暗区域细节减少，对比度在暗区域降低，
# 使得这些区域看起来更加模糊或者细节不那么清晰。然而，在一些场景中，增加
# γ 值可能会帮助提升亮部与暗部的对比度，使某些细节变得更易于观察，但这通常以牺牲暗部细节为代价。

# 当 γ<1 时：随着
# γ 的减小，图像的暗部细节会被增强，变得更亮，而亮部的变化相对较少。这可以增加暗部的可见细节和对比度，
# 使得这些区域的内容更加明显和清晰。这种调整特别适合于提升在暗处拍摄或者曝光不足的图像的可见性和细节。然而，如果
# γ 值过小，可能会导致图像的亮部过曝，细节丢失。