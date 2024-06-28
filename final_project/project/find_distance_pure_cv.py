import cv2
import numpy as np
import itertools

def find_distance_cv(file_path):
    # 读取图像
    image = cv2.imread(file_path)
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
        # center1 = (x1 + w1 // 2, y1 + h1 // 2)
        # center2 = (x2 + w2 // 2, y2 + h2 // 2)

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
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

        # print(distance)
        # 设定阈值，判断是否相近
        min_distance = 40  # 设定最小距离

        if distance < 40:
            return [contour1, contour2]
        else:
            pass

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

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 计算轮廓的矩
        M = cv2.moments(c)
        if M["m00"] == 0:
            return False

        return True

    # 假定具有特定宽高比的轮廓为人物
    filtered_contours = [c for c in contours if
                         is_person_contour(c, image) and is_proper_contour(c, image) and c is not None]

    # 将元组转换为列表
    filtered_contours = list(filtered_contours)
    # print(type(filtered_contours))

    need_merged_contours = []
    # 遍历所有不重复的轮廓对
    for contour1, contour2 in itertools.combinations(filtered_contours, 2):
        need_merged_contours.append(is_contour_close(contour1, contour2, filtered_contours))

    # 存储所有的边界框
    all_bounding_boxes = []

    # 遍历所有轮廓
    for contour in filtered_contours:
        # 获取当前轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 将边界框添加到存储所有边界框的列表中
        all_bounding_boxes.append((x, y, w, h))

    # 遍历所有不重复的轮廓对
    for contour_pair in need_merged_contours:
        if contour_pair:
            contour1, contour2 = contour_pair
            # 获取轮廓1的边界框
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            # 获取轮廓2的边界框
            x2, y2, w2, h2 = cv2.boundingRect(contour2)

            # 合并轮廓区域的边界框
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y

            # 创建新的合并后的轮廓区域
            merged_contour_box = (new_x, new_y, new_w, new_h)

            # 绘制合并后的框
            cv2.rectangle(image, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)

            # 将合并后的边界框添加到所有边界框列表中
            all_bounding_boxes.append(merged_contour_box)

            # 删除原始的边界框
            if (x1, y1, w1, h1) in all_bounding_boxes:
                all_bounding_boxes.remove((x1, y1, w1, h1))
            if (x2, y2, w2, h2) in all_bounding_boxes:
                all_bounding_boxes.remove((x2, y2, w2, h2))

    # 遍历所有边界框
    for box in all_bounding_boxes:
        x, y, w, h = box
        # 在图像中绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 遍历所有边界框的中心点
    for box1 in all_bounding_boxes:
        x1, y1, w1, h1 = box1
        center1 = (x1 + w1 // 2, y1 + h1 // 2)

        # 计算当前框和其他框之间的距离
        for box2 in all_bounding_boxes:
            x2, y2, w2, h2 = box2
            center2 = (x2 + w2 // 2, y2 + h2 // 2)

            # 计算两个中心点之间的水平距离
            horizontal_distance = abs(center1[0] - center2[0])

            # 在图像上输出距离
            cv2.putText(image, str(horizontal_distance),
                        (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 显示结果
    # 保存处理后的RGB图像到文件
    cv2.imwrite('static/find_distance/edged_pic.jpg', edged)
    cv2.imwrite('static/find_distance/distance_pic.jpg', image)
    #cv2.imshow("Edged", edged)
    #cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



