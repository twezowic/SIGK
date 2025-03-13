import cv2
import os
import random


def resize_and_crop(image_path, output_path, new_size=256):
    img = cv2.imread(image_path)
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


def cut(image_path, output_path, cut_size=3):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    x_start = random.randint(0, w-cut_size)
    y_start = random.randint(0, h-cut_size)
    img[y_start:y_start+cut_size, x_start:x_start+cut_size] = 0
    cv2.imwrite(output_path, img)


def prepare_impainting(input_folder, output_folder, cut_size=3, samples=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        name, ext = filename.split(os.extsep)
        for i in range(samples):
            output_path = os.path.join(output_folder, name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, str(i)+os.extsep+ext)
            cut(input_path, output_path, cut_size)


# Preparing data: orignal data -> 256 x 256
input_folder = "./data/raw/train"
output_folder = "./data/intermediate/train"
process_folder(input_folder, output_folder)

input_folder = "./data/raw/valid"
output_folder = "./data/intermediate/valid"
process_folder(input_folder, output_folder)

# Prepare for scalling task: 256 x 256 -> 32 x 32
input_folder = "./data/intermediate/train"
output_folder = "./data/scalling/train"
process_folder(input_folder, output_folder, 32)

input_folder = "./data/intermediate/valid"
output_folder = "./data/scalling/valid"
process_folder(input_folder, output_folder, 32)

# Preparing for inpainting: Removing random squares from image
samples = 10
cut_size = 32

input_folder = "./data/intermediate/train"
output_folder = "./data/inpainting/train"
prepare_impainting(input_folder, output_folder, cut_size=cut_size,
                   samples=samples)

input_folder = "./data/intermediate/valid"
output_folder = "./data/inpainting/valid"
prepare_impainting(input_folder, output_folder, cut_size=cut_size,
                   samples=samples)
