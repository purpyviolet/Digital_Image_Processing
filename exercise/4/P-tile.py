import cv2
import numpy as np

def p_tile_thresholding(image_path, p):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, 0)
    # 计算像素总数
    total_pixels = image.size
    # 计算先验概率对应的像素数
    target_pixel_count = total_pixels * p

    # 计算累积直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)

    # 找到累积直方图大于或等于目标像素数的最小灰度值
    threshold = np.searchsorted(cumulative_hist, target_pixel_count)
    # 应用阈值进行二值化
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image

# 使用示例
binary_img = p_tile_thresholding('noir.jpg', 0.3)  # 假设先验概率为50%
cv2.imshow('Binary Image', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
