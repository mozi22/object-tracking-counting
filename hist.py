import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculateCloseness(prevFeatures, currentFeatures):
    return cv2.compareHist(prevFeatures, currentFeatures, cv2.HISTCMP_BHATTACHARYYA)



im = cv2.imread("samples/object_images2/125.jpg")
im2 = cv2.imread("samples/object_images2/159.jpg")

# im = cv2.resize(im, dsize=(63, 193), interpolation=cv2.INTER_NEAREST)
# im2 = cv2.resize(im2, dsize=(93, 98), interpolation=cv2.INTER_CUBIC)

print(im.shape)
print(im2.shape)

hist = cv2.calcHist([im], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([im2], [0], None, [256], [0, 256])


closenets = calculateCloseness(hist,hist2)

# print(closenets)
plt.hist(im.ravel(),256,[0,256]); plt.show()


#
# print(hist)
# plt.hist(hist, bins='auto')
# plt.title("Histogram with 'auto' bins")
# plt.show()