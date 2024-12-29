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
        if im_shape[0] != points[0] and im_shape[1] != im_shape[1]:
             resized = cv2.resize(image, points, interpolation=cv2.INTER_LINEAR)
             cv2.imwrite(im_path, resized)
        return 1

    except Exception as e:
        print(f"{e}. Image is {im_path}")
        parent_folder = im_path.split("CleanData/")[1]
        stage = parent_folder.split("_")[0]
        files.report_error(im_path, 'Data Corruption', 'Resizing', stage) 
        files.delete_file(im_path)

        return 0
