import cv2
from matplotlib import pyplot as plt

# 读取图片
image = cv2.imread('noir.jpg')

# 将图片转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算灰度图的直方图
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 显示灰度图
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘制直方图
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()

import cv2
from matplotlib import pyplot as plt

# 读取图片
image = cv2.imread('noir.jpg')

# 将图片转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用直方图均衡化
equalized_image = cv2.equalizeHist(gray_image)

# 计算均衡化后的直方图
histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# 显示均衡化后的灰度图
cv2.imshow('Equalized Gray Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘制均衡化后的直方图
plt.figure()
plt.title('Equalized Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()
