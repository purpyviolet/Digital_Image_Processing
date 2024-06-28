import cv2
import numpy as np

# Function for histogram equalization of a channel
def hist_eq(channel, bins=256):
    hist, bins = np.histogram(channel.flatten(), bins, [0, bins])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    return cdf_normalized[channel]

# Read the image
img = cv2.imread('noir.jpg')

# Display the original image in a window
cv2.imshow('Original Image', img)
cv2.waitKey(0)  # Wait for a key press to proceed

# Split the channels
R, G, B = cv2.split(img)

# Apply histogram equalization to each channel
R_eq = hist_eq(R)
G_eq = hist_eq(G)
B_eq = hist_eq(B)

# Merge the equalized channels and display
img_eq_rgb = cv2.merge((R_eq, G_eq, B_eq))
cv2.imshow('Equalized RGB Image', img_eq_rgb)
cv2.waitKey(0)  # Wait for a key press to proceed

# Convert to HSV, equalize the V channel, convert back to RGB, and display
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(img_hsv)
V_eq = hist_eq(V)
img_eq_hsv = cv2.merge((H, S, V_eq))
img_eq_hsv_rgb = cv2.cvtColor(img_eq_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('Equalized HSV Image', img_eq_hsv_rgb)
cv2.waitKey(0)  # Wait for a key press to proceed

# Close all windows
cv2.destroyAllWindows()
