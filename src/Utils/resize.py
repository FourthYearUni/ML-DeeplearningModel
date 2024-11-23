"""
Utility for resizing images
"""

import cv2
import os


def resize_picture(im_path: str):
    """
    Returns a resized picture
    """
    image = cv2.imread(im_path)
    print(f'This is the file {im_path}')
    points = (150, 150)
    resized = cv2.resize(image, points, interpolation=cv2.INTER_LINEAR)

    return resized


if __name__ == "__main__":
    im_folder = "AIimages"
    res_folder = "resized_images"

    for file_name in os.listdir(im_folder):
        f = os.path.join(im_folder, file_name)
        # checking if it is a file

        if os.path.isdir(f):
            print(f"Entering path {f}")
            for index, file in enumerate(os.listdir(f)):
                f_new = os.path.join(f, file)
                if os.path.isfile(f_new):
                    new_im = resize_picture(f_new)
                    ext = file.split(".")[1]
                    new_f = f"Stage_{f.split(' ')[1]}_2083264_{index}.{ext}"
                    out_path = os.path.join(res_folder, new_f)

                    # Resizing the image
                    cv2.imwrite(out_path, new_im)
                    print(f"Saving file {new_f}")
