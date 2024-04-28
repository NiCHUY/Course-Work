import cv2
import numpy as np


def threshold_segmentation(image_path, lower_threshold, upper_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(image, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    return thresholded


def niblack_thresholding(image_path, window_size, k):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = np.copy(gray)
    half_window = window_size // 2

    for i in range(half_window, gray.shape[0] - half_window):
        for j in range(half_window, gray.shape[1] - half_window):
            window = gray[i - half_window: i + half_window + 1, j - half_window: j + half_window + 1]
            mean = np.mean(window)
            std_dev = np.std(window)
            threshold = mean + k * std_dev
            if gray[i, j] > threshold:
                thresholded_image[i, j] = 255
            else:
                thresholded_image[i, j] = 0

    return thresholded_image
