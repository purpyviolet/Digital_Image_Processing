import cv2
import numpy as np
import itertools

# 读取图像
image = cv2.imread('game of thrones.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 1.8)

# 进行Canny边缘检测
edged = cv2.Canny(blurred, 50, 150)

# 找到边缘的轮廓
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def is_contour_close(contour1, contour2, contour):
    """
    Determine if two contours are close to each other based on their bounding boxes.
    """
    # 获取轮廓1的边界框
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    # 获取轮廓2的边界框
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    # 计算两个边界框的中心点
    #center1 = (x1 + w1 // 2, y1 + h1 // 2)
    #center2 = (x2 + w2 // 2, y2 + h2 // 2)

    # 计算两个中心点之间的距离
    if x1 + w1 // 2 > x2 + w2 // 2:
        x_distance = x1 + w1 // 2 - x2 - w2 // 2 - w1 // 2 - w2 // 2
    else:
        x_distance = x2 + w2 // 2 - x1 - w1 // 2 - w1 // 2 - w2 // 2

    if y1 + h1 // 2 > y2 + h2 // 2:
        y_distance = y1 + h1 // 2 - y2 - h2 // 2 - h1 // 2 - h2 // 2
    else:
        y_distance = y2 + h2 // 2 - y1 - h1 // 2 - h1 // 2 - h2 // 2

    distance = max(abs(x_distance), abs(y_distance))

    # 计算两个边界框的中心点
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)

    # 计算两个中心点之间的距离
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    print(distance)
    # 设定阈值，判断是否相近
    min_distance = 40  # 设定最小距离


    if distance < min_distance:
        print(1)
        # 合并轮廓区域的边界框
        new_x = min(x1, x2)
        new_y = min(y1, y2)
        new_w = max(x1 + w1, x2 + w2) - new_x
        new_h = max(y1 + h1, y2 + h2) - new_y

        # 创建新的合并后的轮廓区域
        merged_contour = np.array(
            [[new_x, new_y], [new_x + new_w, new_y], [new_x + new_w, new_y + new_h], [new_x, new_y + new_h]])

        # 从轮廓列表中移除原轮廓并添加新合并的轮廓
        contours_merged = [merged_contour] + [c for c in contours if c is not contour1 and c is not contour2]

        return contours_merged

    return contour


def is_person_contour(contour, image):
    """
    Determine if a contour is likely to be a person based on shape characteristics.
    """
    # Calculate bounding box dimensions
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate aspect ratio
    aspect_ratio = float(w) / h

    # Check if aspect ratio is within a range that's typical for a standing person
    # This range would need to be determined based on observation and testing
    return 0.1 < aspect_ratio < 0.9

def is_proper_contour(c, image):
    x, y, w, h = cv2.boundingRect(c)
    # 计算轮廓周长
    perimeter = cv2.arcLength(c, True)
    # 过滤掉周长太短的轮廓
    if perimeter < 100:
        return False
    # 过滤掉靠近图像边缘的轮廓
    if x < 20 or x + w > image.shape[1] - 20 or y < 40:
        return False

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 计算轮廓的矩
    M = cv2.moments(c)
    if M["m00"] == 0:
        return False

    return True

# 假定具有特定宽高比的轮廓为人物
filtered_contours = [c for c in contours if is_person_contour(c, image) and is_proper_contour(c, image) and c is not None]



# 将元组转换为列表
filtered_contours = list(filtered_contours)
# print(type(filtered_contours))

# # 遍历所有不重复的轮廓对
# for contour1, contour2 in itertools.combinations(filtered_contours, 2):
#     filtered_contours = is_contour_close(contour1, contour2, filtered_contours)


centers = []

for c in filtered_contours:
    #计算轮廓的矩
    M = cv2.moments(c)
    # 跳过面积为零的轮廓
    if M["m00"] != 0:
        # print(True)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # 获取轮廓1的边界框
        x1, y1, w1, h1 = cv2.boundingRect(c)
        centers.append((cX, cY))
        # 在质心处画圆
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        # 绘制轮廓
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    else:
        # 处理面积为零的轮廓的逻辑，或者简单地忽略它们
        pass






# 计算两个中心点之间的距离
if len(centers) >= 2:
    dX, dY = centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]
    distance = np.sqrt((dX ** 2) + (dY ** 2))
    cv2.line(image, centers[0], centers[1], (255, 0, 0), 1)
    cv2.putText(image, f"{distance:.2f}", (int((centers[0][0] + centers[1][0]) / 2), int((centers[0][1] + centers[1][1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# 显示结果
cv2.imshow("Image", image)
cv2.imwrite('distance_show_error.jpg', image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
