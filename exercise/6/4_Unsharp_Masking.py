import cv2
import numpy as np

# 读取图像
img = cv2.imread('the_woman.png', cv2.IMREAD_GRAYSCALE)  # 确保图像路径正确

# 应用高斯模糊
blurred_image = cv2.GaussianBlur(img, (5, 5), 3)
# blurred_image = cv2.medianBlur(img, 9)

# 计算原始图像和模糊图像的差
unsharp_mask = cv2.subtract(img, blurred_image)

# 将边缘加回原图以增强边缘
sharpened_img = cv2.add(img, unsharp_mask)

# 从原图中减去边缘图像，查看效果
softened_img = cv2.subtract(img, unsharp_mask)

# 显示原图、加强后的图和软化后的图
cv2.imshow('Original Image', img)
# cv2.imshow('Unsharp Mask', unsharp_mask)
cv2.imshow('Sharpened Image', sharpened_img)
cv2.imshow('Softened Image', softened_img)


cv2.waitKey(0)  # 等待按键事件
cv2.destroyAllWindows()  # 关闭所有窗口
