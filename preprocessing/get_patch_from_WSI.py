import os
import openslide
import cv2
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count

import time

Image.MAX_IMAGE_PIXELS = None

def split_image(img_path, mask_path, output_dir, mask_output_dir, patch_size, overlap, threshold, white_pixel_threshold, downsample_rate):
    slide = openslide.OpenSlide(img_path)
    mask_image = Image.open(mask_path)

    level0_dimensions = slide.dimensions
    step_size = patch_size - overlap

    for i in range(0, level0_dimensions[1] - patch_size + 1, step_size):
        for j in range(0, level0_dimensions[0] - patch_size + 1, step_size):
            img_patch = np.array(slide.read_region((j, i), 0, (patch_size, patch_size)))
            mask_patch = np.array(mask_image.crop((j, i, j+patch_size, i+patch_size)))

            # Binarize the patch
            _, binarized_patch = cv2.threshold(cv2.cvtColor(img_patch, cv2.COLOR_RGBA2GRAY), threshold, 255, cv2.THRESH_BINARY)
            white_pixels = np.sum(binarized_patch == 255)

            # Check if patch meets the white pixel threshold criteria
            if white_pixels < white_pixel_threshold:
                base_name = os.path.basename(img_path).split('.')[0]
                patch_name = f"{base_name}_{int(j*downsample_rate)}_{int(i*downsample_rate)}.png"

                # Downsample the patch using nearest-neighbor interpolation
                img_patch_downsampled = cv2.resize(img_patch, None, fx=downsample_rate, fy=downsample_rate, interpolation=cv2.INTER_NEAREST)
                mask_patch_downsampled = cv2.resize(mask_patch, None, fx=downsample_rate, fy=downsample_rate, interpolation=cv2.INTER_NEAREST)
                
                cv2.imwrite(os.path.join(output_dir, patch_name), img_patch_downsampled)
                cv2.imwrite(os.path.join(mask_output_dir, patch_name), mask_patch_downsampled)

    slide.close()
    mask_image.close()
    print(img_path)

def process_images(input_dir, output_dir, mask_output_dir, patch_size, overlap, threshold, white_pixel_threshold, downsample_rate, processes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(".tif") and "_mask" not in f]

    with Pool(processes=processes) as pool:
        for img_file in image_files:
            mask_file = img_file.replace(".tif", "_mask.tif")
            if mask_file in os.listdir(input_dir):
                pool.apply_async(split_image, (os.path.join(input_dir, img_file),
                                               os.path.join(input_dir, mask_file),
                                               output_dir, mask_output_dir, patch_size, overlap, threshold, white_pixel_threshold, downsample_rate))
        pool.close()
        pool.join()

if __name__ == "__main__":
    start_time = time.time()
    input_dir = "/home/ubuntu/sdb/huangruiwei/data/camelyon16_original"  # replace with your directory
    output_dir = "/home/ubuntu/sdb/huangruiwei/data/camelyon16/patch_images"
    mask_output_dir = "/home/ubuntu/sdb/huangruiwei/data/camelyon16/patch_labels"
    patch_size = 2048
    overlap = 0
    threshold = 180  # adjust as required
    white_pixel_threshold = patch_size * patch_size * 0.95
    downsample_rate = 0.125  # adjust as required, e.g., 0.5 for 2x downsample
    processes = min(cpu_count(), 3)  # by default, use 4 processes or less if CPU has fewer cores

    process_images(input_dir, output_dir, mask_output_dir, patch_size, overlap, threshold, white_pixel_threshold, downsample_rate, processes)
    print((time.time() - start_time)/60, " min(s) ")

