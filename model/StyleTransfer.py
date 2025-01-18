import cv2
import numpy as np
from PIL import Image

def apply_cartoon_effect(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

def apply_grayscale(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray)

def Cartoonify(image):
    cartoon_image = apply_cartoon_effect(image)
    cartoon_grayscale = apply_grayscale(cartoon_image)
    return cartoon_grayscale