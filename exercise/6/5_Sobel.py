import cv2
import numpy as np

# 读取图像
img = cv2.imread('the_moon.png', cv2.IMREAD_GRAYSCALE)

# 计算x方向上的Sobel边缘
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# 计算y方向上的Sobel边缘
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 组合x和y方向上的边缘
sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)

# 显示原图和Sobel边缘
cv2.imshow('Original', img)
cv2.imshow('Sobel X', cv2.convertScaleAbs(sobelx))
cv2.imshow('Sobel Y', cv2.convertScaleAbs(sobely))
cv2.imshow('Sobel Combined', sobel_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
