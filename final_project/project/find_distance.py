import cv2
import numpy as np

def find_distance_dl(file_path):
    # 加载预训练的模型和配置文件
    global output
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

    # 读取图像
    image = cv2.imread(file_path)
    (h, w) = image.shape[:2]

    # 准备图像输入
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.013, (300, 300), 127.5)

    # 输入数据到网络
    net.setInput(blob)
    detections = net.forward()

    # 存储检测到的人物坐标
    persons = []

    # 循环检测结果
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # 置信度阈值
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # 类别编号15对应于'person'
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                persons.append((startX, startY, endX, endY))
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # 输出检测到的每个人物的坐标和置信度
                print(f"Detected person at [{startX}, {startY}, {endX}, {endY}] with confidence {confidence}")

    # 如果检测到的人物数量少于两个，进行图像分割和检测
    if len(persons) < 2:
        print('The number of detected person is less than 2, we try to cut the region and detect again.')
        # 图像分割为左右两个区域
        left_region = image[:, :w // 2]
        right_region = image[:, w // 2:]

        # 对左右两个区域分别进行人物检测
        left_blob = cv2.dnn.blobFromImage(cv2.resize(left_region, (300, 300)), 0.013, (300, 300),
                                          127.5)
        right_blob = cv2.dnn.blobFromImage(cv2.resize(right_region, (300, 300)), 0.013, (300, 300),
                                           127.5)

        net.setInput(left_blob)
        left_detections = net.forward()

        net.setInput(right_blob)
        right_detections = net.forward()

        # 合并左右两个区域的检测结果
        for detections in [left_detections, right_detections]:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.3:  # 置信度阈值
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:  # 类别编号15对应于'person'
                        box = detections[0, 0, i, 3:7] * np.array([w // 2, h, w // 2, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        persons.append((startX, startY, endX, endY))
                        # 将相对位置调整为全局位置
                        startX += w // 2
                        endX += w // 2
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        # 输出检测到的每个人物的坐标和置信度
                        print(f"Detected person at [{startX}, {startY}, {endX}, {endY}] with confidence {confidence}")

        if len(persons) < 2:
            output = 'Not enough person in the image to calculate distance.'
            return output



    # 定义假设的平均身高和肩宽，单位为像素
    average_height = 170
    average_shoulder_width = 40

    # 初始化距离信息
    distance_info = []

    # 循环检测结果
    for i in range(len(persons) - 1):
        person1 = persons[i]
        person2 = persons[i + 1]

        # 计算每个人的高度和宽度
        person1_height = person1[3] - person1[1]
        person2_height = person2[3] - person2[1]
        person1_width = person1[2] - person1[0]
        person2_width = person2[2] - person2[0]

        # 计算人物的身高和宽度比例
        person1_height_ratio = average_height / person1_height
        person2_height_ratio = average_height / person2_height
        person1_width_ratio = average_shoulder_width / person1_width
        person2_width_ratio = average_shoulder_width / person2_width

        person1_trans = (person1_height_ratio + person1_width_ratio) / 2
        person2_trans = (person2_height_ratio + person2_width_ratio) / 2

        # 检查人物是否满足比例要求
        # person1_meets_ratio = 0.8 <= person1_height_ratio <= 1.2 and 0.8 <= person1_width_ratio <= 1.2
        # person2_meets_ratio = 0.8 <= person2_height_ratio <= 1.2 and 0.8 <= person2_width_ratio <= 1.2

        # 检查人物是否满足比例要求
        person1_rat = person1_height / person1_width
        person2_rat = person2_height / person2_width
        person1_meets_ratio = (2 <= person1_rat <= 6)
        person2_meets_ratio = (2 <= person2_rat <= 6)

        # print(person1_meets_ratio)
        # print(person2_meets_ratio)

        # 计算两个人之间的像素距离
        dx = (person1[0] + person1[2]) / 2 - (person2[0] + person2[2]) / 2
        # dy = (person1[1] + person1[3]) / 2 - (person2[1] + person2[3]) / 2
        # pixel_distance = np.sqrt(dx**2 + dy**2)
        pixel_distance = abs(dx)

        # 如果两个人物的身高和宽度都满足比例要求，则计算实际距离
        if person1_meets_ratio or person2_meets_ratio:
            # 如果两个人都满足，计算距离区间
            if person1_meets_ratio and person2_meets_ratio:
                person1_distance = pixel_distance * person1_trans
                person2_distance = pixel_distance * person2_trans
                distance_interval = (min(abs(person1_distance), abs(person2_distance)), max(abs(person1_distance), abs(person2_distance)))
                distance_info.append((distance_interval, "Actual distance interval"))
            # 如果只有一个人满足，用满足的人的比例来计算
            elif person1_meets_ratio:
                person_distance = pixel_distance * person1_trans
                distance_info.append((person_distance, "Actual distance"))
            else:
                person_distance = pixel_distance * person2_trans
                distance_info.append((person_distance, "Actual distance"))
        else:
            # 如果两个人都不满足比例要求，不进行距离计算
            distance_info.append((pixel_distance, "Pixel distance"))



    # 输出距离信息
    for distance, info in distance_info:
        if info == "Actual distance":
            print(f"{info}: {distance}cm")
            output = f"{info}: {distance}cm"
        elif info == 'Actual distance interval':
            print(f"{info}: {distance}cm")
            output = f"{info}: {distance}cm"
        else:
            addition_info = ("The two person detected in the pic didn't meet the expected ratio, we don't calculate the "
                             "actual distance.\n")
            print(f"{addition_info}\n{info}: {distance} pixels")
            output = f"{addition_info}\n{info}: {distance} pixels"

    # 显示结果
    # cv2.imshow('Output', image)
    cv2.imwrite('static/find_distance/distance_dl.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output

# find_distance_dl('game of thrones.jpg')