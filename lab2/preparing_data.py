import os
import cv2
from utils import read_exr

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"


def resize_and_crop(image_path, output_path, new_size=256):
    img = read_exr(image_path)

    height, width = img.shape[:2]

    # Get smaller edge
    mini = min(height, width)
    scale = mini / new_size

    # Resize
    img_resized = cv2.resize(img, (int(width/scale), int(height/scale)))

    new_height, new_width = img_resized.shape[:2]

    x_start = (new_width - new_size) // 2
    y_start = (new_height - new_size) // 2

    img_cropped = img_resized[y_start:y_start+new_size,
                              x_start:x_start+new_size]

    cv2.imwrite(output_path, img_cropped)


def process_folder(input_folder, output_folder, new_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        resize_and_crop(input_path, output_path, new_size=new_size)


process_folder("./data/reference", './data/processed')
