import cv2

# 读取原始图像
original_image = cv2.imread("128x128.png")

# 线性插值函数
def bilinear_interpolation(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# 最近邻插值函数
def nearest_neighbor_interpolation(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

# 使用线性插值方法得到新图像
new_image_bilinear = bilinear_interpolation(original_image, 1024, 1024)
cv2.imshow("Bilinear Interpolation", new_image_bilinear)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用最近邻插值方法得到新图像
new_image_nearest = nearest_neighbor_interpolation(original_image, 1024, 1024)
cv2.imshow("Nearest Neighbor Interpolation", new_image_nearest)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要保存图像，可以使用以下代码
# cv2.imwrite("bilinear_interpolation.png", new_image_bilinear)
# cv2.imwrite("nearest_neighbor_interpolation.png", new_image_nearest)
