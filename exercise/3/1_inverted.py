import cv2
import numpy as np

# 读取图片
image = cv2.imread('boy.png')

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用翻转变换
inverted_image = 255 - gray_image

# 显示原始灰度图像和翻转后的图像
cv2.imshow('Original Gray Image', gray_image)
cv2.imshow('Inverted Gray Image', inverted_image)

# 等待键盘输入，之后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
