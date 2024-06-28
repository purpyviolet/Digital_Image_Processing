import cv2
import numpy as np

# 步骤1: 加载原始带雾霾的图片
image_path = 'foggy.png' # 替换为你的图片路径
original_image = cv2.imread(image_path)

# 步骤2: 创建一张同样大小的雾霾颜色图
# 假设雾霾颜色为轻灰色，你可以根据需要调整这个颜色
haze_color = [120, 120, 120] # BGR格式
haze_image = np.full(original_image.shape, haze_color, dtype=np.uint8)

# 步骤3: 相减实现去雾
# 这里使用cv2.subtract确保结果在0到255之间
dehazed_image = cv2.subtract(original_image, haze_image)

# 步骤4: 调整结果
# 提高亮度和对比度（简单示例，具体值需要调整）
dehazed_image = cv2.convertScaleAbs(dehazed_image, alpha=2, beta=30)

# 显示结果
cv2.imshow('Original', original_image)
cv2.imshow('Dehazed', dehazed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

