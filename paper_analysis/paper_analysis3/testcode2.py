import cv2
import numpy as np

# 载入图像
input_image_path = 'car_house.png'
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否载入成功
if input_image is None:
    raise ValueError("Image not found or path is incorrect")

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_image = clahe.apply(input_image)

# 拉普拉斯锐化
# 扩展图像的灰度级别以使用拉普拉斯算子进行锐化处理
ddepth = cv2.CV_16S
laplacian = cv2.Laplacian(equalized_image, ddepth, ksize=3)
abs_laplacian = cv2.convertScaleAbs(laplacian)
sharpened_image = cv2.subtract(equalized_image, abs_laplacian)

# 将处理后的图像保存至文件
output_image_path = 'output.png'
cv2.imwrite(output_image_path, sharpened_image)

