import cv2
import numpy as np

def butterworth_lowpass_filter(d, d0, n):
    # d: distance matrix
    # d0: cutoff frequency
    # n: order of the filter
    return 1 / (1 + (d / d0) ** (2 * n))

def apply_histogram_equalization(image):
    # Apply histogram equalization to an image
    return cv2.equalizeHist(image)

# Load the image
image_path = 'foggy.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Or cv2.IMREAD_COLOR for RGB

# Fourier transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create Butterworth low pass filter mask
rows, cols = image.shape
crow,ccol = rows//2 , cols//2
d = np.sqrt((np.arange(-crow, crow)**2).reshape(-1, 1) + (np.arange(-ccol, ccol)**2))
lowpass = butterworth_lowpass_filter(d, 500, 5) # d0=30, n=2 are chosen for demonstration

# Apply filter
fshift_filtered = fshift * lowpass
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Histogram equalization on low-frequency component
low_freq_component = img_back.astype(np.uint8)
equalized_low_freq = apply_histogram_equalization(low_freq_component)

# High frequency component calculation and final image recombination might need
# additional steps, especially for handling the color images and combining the channels

# Display the original and the processed image
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', equalized_low_freq)
cv2.waitKey(0)
cv2.destroyAllWindows()
