import cv2
import numpy as np

# 读取第一个图像
image1 = cv2.imread('high_res.png')

# 创建一个黑色图像
image2 = np.zeros(image1.shape, dtype=np.uint8)

# 定义白色区域的范围
start_point = (200, 200)
end_point = (500, 500)

# 在图像上绘制白色矩形
color = (255, 255, 255)
thickness = -1  # 使用负值填充矩形
image2 = cv2.rectangle(image2, start_point, end_point, color, thickness)

# 将图像转换为浮点数格式
image1 = image1.astype(np.float32) / 255.0
image2 = image2.astype(np.float32) / 255.0

# 将图像相乘
result = cv2.multiply(image1, image2)

# 将结果转换回整数格式
result = (result * 255).astype(np.uint8)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
