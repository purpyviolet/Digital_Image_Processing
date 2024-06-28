import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像a
img_a = cv2.imread('skeleton.png', cv2.IMREAD_GRAYSCALE)

# 应用拉普拉斯算子处理图像a得到图像b
kernel_laplacian = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img_b = cv2.filter2D(img_a, -1, kernel_laplacian)

# 将原图像a与拉普拉斯处理后的图像b相加得到图像c
img_c = cv2.add(img_a, img_b)

# 对图像c应用中值滤波得到图像d
img_d = cv2.medianBlur(img_c, 5)

# 定义Sobel算子核
Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

# 应用Sobel算子得到图像e
sobelx = cv2.filter2D(img_a, cv2.CV_32F, Gx)
sobely = cv2.filter2D(img_a, cv2.CV_32F, Gy)

# 使用addWeighted结合两个梯度
# 权重可以根据需要进行调整，这里使用0.5和0.5，以平均它们的贡献
# gamma值设置为0，因为这里不需要加入额外的亮度
# img_e = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# 计算梯度幅值
img_e = cv2.magnitude(sobelx, sobely)

# 将幅值映射到0-255
img_e = cv2.normalize(img_e, None, 0, 255, cv2.NORM_MINMAX)

# 转换为8位图像
img_e = np.uint8(img_e)

# 对图像e应用平均滤波得到图像f
kernel_average = np.ones((5, 5), np.float32) / 25
img_f = cv2.filter2D(img_e, -1, kernel_average)

# 确保img_c是正确的数据类型
# 如果img_c是由两个uint8图像相加得到，它可能会有超过255的值
# 所以我们首先要将其转换为float32，然后做归一化处理
img_c = img_c.astype(np.float32)
img_c = cv2.normalize(img_c, None, 0, 1, cv2.NORM_MINMAX)

# 确保img_f也是float32类型，并且在0到1的范围内
img_f = img_f.astype(np.float32)
img_f = cv2.normalize(img_f, None, 0, 1, cv2.NORM_MINMAX)

# 现在我们可以安全地相乘了
img_g = cv2.multiply(img_f, img_c)

# 将结果img_g转换回uint8类型，以便可以显示或进一步处理
img_g = cv2.normalize(img_g, None, 0, 255, cv2.NORM_MINMAX)
img_g = img_g.astype(np.uint8)

# 将图像g和图像a相加得到图像h
img_h = cv2.add(img_a, img_g.astype(np.uint8))

# 对图像h进行幂律变换得到最终图像i
gamma = 0.5  # 这个gamma值可能需要调整以得到好的结果
img_i = cv2.pow(img_h / 255.0, gamma)
img_i = np.uint8(img_i * 255)

# 显示所有图像
cv2.imshow('Image A (Original)', img_a)
#cv2.imshow('Image B (Laplacian)', img_b)
#cv2.imshow('Image C (A + B)', img_c)
#cv2.imshow('Image D (Median of C)', img_d)
#cv2.imshow('Image E (Sobel)', img_e)
#cv2.imshow('Image F (Average of E)', img_f)
#cv2.imshow('Image G (C * F)', img_g)
#cv2.imshow('Image H (A + G)', img_h)
cv2.imshow('Image I (Power-law)', img_i)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用matplotlib展示所有图像
plt.figure(figsize=(20, 10))

plt.subplot(3, 3, 1), plt.imshow(img_a, cmap='gray'), plt.title('Image A (Original)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 2), plt.imshow(img_b, cmap='gray'), plt.title('Image B (Laplacian)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 3), plt.imshow(img_c, cmap='gray'), plt.title('Image C (A + B)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 4), plt.imshow(img_d, cmap='gray'), plt.title('Image D (Median of C)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 5), plt.imshow(img_e, cmap='gray'), plt.title('Image E (Sobel)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 6), plt.imshow(img_f, cmap='gray'), plt.title('Image F (Average of E)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 7), plt.imshow(img_g, cmap='gray'), plt.title('Image G (C * F)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 8), plt.imshow(img_h, cmap='gray'), plt.title('Image H (A + G)')
plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 9), plt.imshow(img_i, cmap='gray'), plt.title('Image I (Power-law)')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
