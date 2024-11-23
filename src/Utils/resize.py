"""
Utility for resizing images
"""

import cv2


def resize_picture(im_path: str):
    """
    Returns a resized picture
    """
    image = cv2.imread(im_path)

    points = (150, 150)
    im_shape = image.shape
    if im_shape(0) != points(0) and im_shape(1) != im_shape(1):
        resized = cv2.resize(image, points, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(im_path, resized)
        print("This image was resized")

    print("This image was not resized")
