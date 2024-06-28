import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread('the_moon.png', 0)  # 确保图像路径正确，0 表示以灰度模式读取

# 应用拉普拉斯算子
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# 使用自定义拉普拉斯掩膜
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_custom = cv2.filter2D(img, -1, kernel)

# 使用 Matplotlib 显示结果
plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian CV2'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.imshow(laplacian_custom, cmap='gray')
plt.title('Custom Laplacian Mask'), plt.xticks([]), plt.yticks([])

plt.show()
