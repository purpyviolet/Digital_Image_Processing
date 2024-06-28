import cv2
import numpy as np


def calculate_gini_coefficient(cdf):
    """计算基尼系数"""
    n = len(cdf)
    coefficient = 2.0 / n
    constant = (n + 1) / n
    weighted_sum = sum((i + 1) * val for i, val in enumerate(cdf))
    gini = coefficient * weighted_sum / sum(cdf) - constant
    return gini


def adaptive_histogram_equalization(image, gini_coeff):
    """自适应直方图均衡化"""
    h, w = image.shape
    hmax = np.max(image)
    n = w * h
    Tu = (1 - gini_coeff) * hmax + gini_coeff * (n / hmax)
    Tl = n / hmax

    image[image < Tu] = (image[image < Tu] / Tu) * Tl
    image[image >= Tu] = Tl + (image[image >= Tu] - Tu)
    return image


def bilinear_interpolation(sub_blocks, original_shape, block_size):
    """Use bilinear interpolation to smooth block edges after processing."""
    # Determine number of blocks in each dimension
    num_blocks_y = original_shape[0] // block_size + (1 if original_shape[0] % block_size != 0 else 0)
    num_blocks_x = original_shape[1] // block_size + (1 if original_shape[1] % block_size != 0 else 0)

    # Initialize empty list to store the new blocks
    new_blocks = []

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            i = y * num_blocks_x + x
            if i < len(sub_blocks):
                block = sub_blocks[i]
                # Check if the block needs padding
                if block.shape[0] < block_size:
                    block = cv2.copyMakeBorder(block, 0, block_size - block.shape[0], 0, 0, cv2.BORDER_REFLECT)
                if block.shape[1] < block_size:
                    block = cv2.copyMakeBorder(block, 0, 0, 0, block_size - block.shape[1], cv2.BORDER_REFLECT)
                new_blocks.append(block)
            else:
                # Create a new block if we're out of actual blocks (for edge cases)
                new_blocks.append(np.zeros((block_size, block_size), dtype=np.uint8))

    def reconstruct_image_from_blocks(blocks, orig_shape, block_size):
        """Reconstruct the image from its blocks after processing."""
        # Calculate the number of blocks along the height and width
        num_blocks_vert = int(np.ceil(orig_shape[0] / block_size))
        num_blocks_horiz = int(np.ceil(orig_shape[1] / block_size))

        # Initialize an empty image with the original shape
        reconstructed = np.zeros(orig_shape, dtype=blocks[0].dtype)

        # Place each block back into the image at the correct location
        for idx, block in enumerate(blocks):
            # Calculate the block's position in the image
            row = (idx // num_blocks_horiz) * block_size
            col = (idx % num_blocks_horiz) * block_size

            # Calculate the dimensions of the block
            block_height = min(block_size, orig_shape[0] - row)
            block_width = min(block_size, orig_shape[1] - col)

            # If the block is smaller than the expected block_size, resize it
            if block.shape[0] != block_height or block.shape[1] != block_width:
                block = cv2.resize(block, (block_width, block_height), interpolation=cv2.INTER_LINEAR)

            # Insert the block into the reconstructed image
            reconstructed[row:row + block_height, col:col + block_width] = block

        return reconstructed

    # Usage
    reconstructed_img = reconstruct_image_from_blocks(enhanced_blocks, original_img.shape, block_size)

    # Resize to original image size using bilinear interpolation
    new_image = cv2.resize(reconstructed_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    return new_image


# 加载图像并转换为灰度图
image_path = 'plane.png'  # 更换为您的图像路径
original_img = cv2.imread(image_path, 0)

# 图像分割
block_size = 8  # 分割的块的大小
sub_blocks = [original_img[y:y + block_size, x:x + block_size] for x in range(0, original_img.shape[1], block_size) for
              y in range(0, original_img.shape[0], block_size)]

# 处理每个块
enhanced_blocks = []
for block in sub_blocks:
    # 计算累积分布函数(cdf)
    hist = cv2.calcHist([block], [0], None, [256], [0, 256]).ravel()
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # 计算基尼系数
    gini_coeff = calculate_gini_coefficient(cdf_normalized)

    # 自适应直方图均衡化
    block = adaptive_histogram_equalization(block, gini_coeff)
    enhanced_blocks.append(block)

# After enhancing the blocks, before reconstructing the image
reconstructed_img = bilinear_interpolation(enhanced_blocks, original_img.shape, block_size)


# 平滑滤波去噪
smoothed_img = cv2.GaussianBlur(reconstructed_img, (3, 3), 0)

# 拉普拉斯锐化处理
laplacian = cv2.Laplacian(smoothed_img, cv2.CV_64F)
sharpened_img = cv2.convertScaleAbs(smoothed_img - 0.3 * laplacian)

# 展示原始图像和增强后的图像
cv2.imshow('Original Image', original_img)
cv2.imshow('Enhanced Image', sharpened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
