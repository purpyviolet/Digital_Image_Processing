import cv2
import numpy as np
import os
def combine_images(image1, image2):
    # 确定合成后图像的尺寸
    max_height = max(image1.shape[0], image2.shape[0])
    max_width = max(image1.shape[1], image2.shape[1])

    # 创建一个足够大的画布
    combined_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # 将第一个图像加到画布上
    combined_image[:image1.shape[0], :image1.shape[1]] += image1

    # 将第二个图像加到画布上
    combined_image[:image2.shape[0], :image2.shape[1]] += image2

    return combined_image

def resize_images(image1, image2):
    # 确定合成后图像的尺寸
    max_height = max(image1.shape[0], image2.shape[0])
    max_width = max(image1.shape[1], image2.shape[1])

    # 调整图像大小
    resized_image1 = cv2.resize(image1, (max_width, max_height))
    resized_image2 = cv2.resize(image2, (max_width, max_height))

    return resized_image1, resized_image2

def blend_images(image1, image2, alpha):
    # 混合图像
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    return blended_image


if __name__ == "__main__":
    image1_path = "2.jpeg"
    image2_path = "3.jpg"

    # 读取图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("无法读取图像，请检查路径是否正确。")
    else:
        # 合成图像
        combined_image = combine_images(image1, image2)

        # 调整图像大小
        resized_image1, resized_image2 = resize_images(image1, image2)

        # 设置混合的透明度
        alpha = 0.5  # 调整透明度以改变若隐若现效果

        # 混合图像
        blended_image = blend_images(resized_image1, resized_image2, alpha)

        # 显示混合图像
        cv2.imshow("Blended Image", blended_image)
        cv2.waitKey(0)
        # 显示合成图像
        cv2.imshow("Combined Image", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存混合图像
        blend_output_path = os.path.join("blended_image.jpg")
        cv2.imwrite(blend_output_path, blended_image)

        # 保存合成图像
        combined_output_path = os.path.join("combined_image.jpg")
        cv2.imwrite(combined_output_path, combined_image)
