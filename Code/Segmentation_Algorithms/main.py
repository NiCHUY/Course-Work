import cv2

import edge_detection
import threshold_algorithms


result = edge_detection.apply_roberts_operator('image.jpg', 250)
cv2.imshow('Roberts', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
result = edge_detection.apply_prewitt_operator('image.jpg', 100)
cv2.imshow('Prewitt', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
result = edge_detection.apply_sobel_operator('image.jpg', 140)
cv2.imshow('Sobel', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


result = threshold_algorithms.threshold_segmentation('image.jpg', 129, 255)
cv2.imshow('Threshold Algorithm', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
result = threshold_algorithms.niblack_thresholding('image.jpg', 5, 0.5)
cv2.imshow('Niblack', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
