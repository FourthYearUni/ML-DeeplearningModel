"""
Utility for resizing images
"""

import cv2
from .files import File


def resize_picture(im_path: str) -> int:
    """
    Returns a resized picture
    """
    files = File()
    image = cv2.imread(im_path)
    points = (150, 150)
    try:
        im_shape = image.shape
    except Exception as e:
        print(f"{e}. Image is {im_path}")
        files.delete_file(im_path)

    if im_shape[0] != points[0] and im_shape[1] != im_shape[1]:
        resized = cv2.resize(image, points, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(im_path, resized)
        print("This image was resized")
        return 1
    print("This image was not resized")
    return 0
