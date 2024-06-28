import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Read the image and convert to grayscale
img = cv2.imread('noir.jpg', cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# 2. Draw the histogram of the original image
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Original Histogram')

# 3. Calculate the cumulative histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.subplot(2, 2, 2)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histogram'), loc='upper left')

# 4. Equalize the histogram
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]

# 5. Draw the histogram of the equalized image
plt.subplot(2, 2, 3)
plt.hist(img2.ravel(), 256, [0, 256])
plt.title('Equalized Histogram')

# 6. Display the original and equalized image
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.title('Original Image')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))
plt.title('Histogram Equalized Image')

plt.show()
