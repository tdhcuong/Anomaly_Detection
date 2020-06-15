import cv2
import numpy as np
from scipy import ndimage

# Calculate Mutual Information Score
def mutual_information_2d(x, y, sigma=1, normalized=False):
    EPS = np.finfo(float).eps
    bins = (256, 256)
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                - np.sum(s2 * np.log(s2)))

    return mi

# Calculate Correlation Coefficient Score
def corr2d(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a*b) / float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    return r

# Calculate Correlation Coefficient Score between 2 RGB images
def calculate_correlation_coef(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return corr2d(image1_gray, image2_gray)

# Remove black border
def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame