import cv2
import numpy as np
from matplotlib import pyplot as plt

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Take the cumsum of the counts and normalize by the number of pixels to get the empirical cumulative distribution functions for the source and template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Interpolate linearly to find the pixel values in the template image that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

# Load the images in grayscale
source_img = cv2.imread('noir.jpg', 0)
template_img = cv2.imread('zhenbai.jpg', 0)

# Perform histogram matching
matched_img = hist_match(source_img, template_img)

# Display the images
cv2.imshow('Source Image', source_img)
cv2.imshow('Template Image', template_img)
cv2.imshow('Matched Image', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot the histograms
plt.figure()
plt.subplot(131)
plt.hist(source_img.ravel(), 256, [0,256])
plt.title('Source Histogram')
plt.subplot(132)
plt.hist(template_img.ravel(), 256, [0,256])
plt.title('Template Histogram')
plt.subplot(133)
plt.hist(matched_img.ravel(), 256, [0,256])
plt.title('Matched Histogram')
plt.show()
