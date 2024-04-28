import cv2
import numpy as np


def apply_roberts_operator(image_path, lower_bound):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roberts_x = np.array([[1, 0],
                          [0, 1]], dtype=np.float32)
    roberts_y = np.array([[0, -1],
                          [-1, 0]], dtype=np.float32)
    edges_x = cv2.filter2D(image, -1, roberts_x).astype(np.float32)
    edges_y = cv2.filter2D(image, -1, roberts_y).astype(np.float32)
    edges = cv2.sqrt(cv2.pow(edges_x, 2) + cv2.pow(edges_y, 2))
    _, binary_image = cv2.threshold(edges, lower_bound, 255, cv2.THRESH_BINARY)

    return binary_image


def apply_prewitt_operator(image_path, lower_bound):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    prewitt_x = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float32)
    prewitt_y = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]], dtype=np.float32)
    edges_x = cv2.filter2D(image, -1, prewitt_x).astype(np.float32)
    edges_y = cv2.filter2D(image, -1, prewitt_y).astype(np.float32)
    edges = cv2.sqrt(cv2.pow(edges_x, 2) + cv2.pow(edges_y, 2))
    _, binary_image = cv2.threshold(edges, lower_bound, 255, cv2.THRESH_BINARY)
    return cv2.erode(cv2.dilate(binary_image, np.ones((5, 5), np.uint8), iterations=1) , np.ones((5, 5), np.uint8), iterations=1)


def apply_sobel_operator(image_path, lower_bound):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]], dtype=np.float32)
    edges_x = cv2.filter2D(image, -1, sobel_x).astype(np.float32)
    edges_y = cv2.filter2D(image, -1, sobel_y).astype(np.float32)
    edges = cv2.sqrt(cv2.pow(edges_x, 2) + cv2.pow(edges_y, 2))
    _, binary_image = cv2.threshold(edges, lower_bound, 255, cv2.THRESH_BINARY)
    return binary_image
