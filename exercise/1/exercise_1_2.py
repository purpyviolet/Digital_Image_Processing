import cv2
import numpy as np

def resize_image(image, max_height=700):
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def digitize_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_original = cv2.imread(image_path)

    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return None

    # 缩放图像以适应屏幕
    image = resize_image(image)
    image_original = resize_image(image_original)

    # 显示原始图像
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # 显示RGB图像
    cv2.imshow("RGB Image", image_original)
    cv2.waitKey(0)

    # 阈值化，生成黑白图
    _, black_white_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    black_white_image = resize_image(black_white_image)

    cv2.imshow("Black & White Image", black_white_image)
    cv2.waitKey(0)

    # 逆转0和1
    inverted_image = cv2.bitwise_not(black_white_image)
    cv2.imshow("Inverted Image", inverted_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # 输出图像的尺寸
    print("图像尺寸:", image.shape)

    # 输出图像的数字矩阵
    print("灰度图数字矩阵:")
    print(image)

    # 输出黑白图数字矩阵
    print("黑白图数字矩阵")
    print(black_white_image)

    # 输出逆转后黑白图数字矩阵
    print("逆转后黑白图数字矩阵")
    print(inverted_image)

    # 输出原始图数字矩阵
    print("RGB图数字矩阵")
    print(image_original)
    return image, black_white_image, inverted_image

if __name__ == "__main__":
    image_path = "11.png"
    digitize_image(image_path)
