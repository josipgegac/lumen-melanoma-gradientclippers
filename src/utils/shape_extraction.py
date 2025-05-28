import os
import cv2
import numpy as np

def get_shape_mask_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Blur image to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), cv2.BORDER_DEFAULT)

    # Apply Otsu's thresholding (binary inverse)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

def get_shape_mask_kmeans(image, k=2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Flatten image into a list of pixel intensities
    h, w = enhanced.shape
    X = enhanced.reshape((-1, 1)).astype(np.float32)

    # Apply KMeans clustering to classify pixels into clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(X, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get binary mask (assuming the darker cluster is the shape region)
    shape_label = np.argmin(centers)
    binary_mask = (labels.flatten() == shape_label).astype(np.uint8) * 255
    binary_mask = binary_mask.reshape((h, w))

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

if __name__ == '__main__':
    data_dir = '../../data/train'
    dest_dir = '../../data/masks'

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file in os.listdir(data_dir):
        image_path = os.path.join(data_dir, file)
        dest_path = os.path.join(dest_dir, file)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        shape_mask = get_shape_mask_otsu(image)

        cv2.imwrite(dest_path, shape_mask)