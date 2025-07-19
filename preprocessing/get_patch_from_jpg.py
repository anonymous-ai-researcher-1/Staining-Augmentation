import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

import time

def split_image(img_path, mask_path, output_dir, mask_output_dir, patch_size, overlap, threshold, white_pixel_threshold):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    h, w, _ = img.shape
    step_size = patch_size - overlap
    
    for i in range(0, h - patch_size + 1, step_size):
        for j in range(0, w - patch_size + 1, step_size):
            img_patch = img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            
            # Binarize the patch
            _, binarized_patch = cv2.threshold(cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
            white_pixels = np.sum(binarized_patch == 255)
            
            # Check if patch meets the white pixel threshold criteria
            if white_pixels < white_pixel_threshold:
                base_name = os.path.basename(img_path).split('.')[0]
                patch_name = f"{base_name}_{i}_{j}.png"
                # mask_name = f"{base_name}_mask_{i}_{j}.png"
                mask_name = f"{base_name}_{i}_{j}.png"
                cv2.imwrite(os.path.join(output_dir, patch_name), img_patch)
                cv2.imwrite(os.path.join(mask_output_dir, mask_name), mask_patch)

    print(img_path)

def process_images(input_dir, output_dir, mask_output_dir, patch_size, overlap, threshold, processes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg") and "_mask" not in f]
    # image_files = ['2019_02170_1-2_2019-02-20 19_37_17-lv1-18315-29727-7652-5581.jpg']
    
    with Pool(processes=processes) as pool:
        for img_file in image_files:
            mask_file = img_file.replace(".jpg", "_mask.jpg")
            if mask_file in os.listdir(input_dir):
                pool.apply_async(split_image, (os.path.join(input_dir, img_file), 
                                               os.path.join(input_dir, mask_file), 
                                               output_dir, mask_output_dir, patch_size, overlap, threshold, white_pixel_threshold))
        pool.close()
        pool.join()

if __name__ == "__main__":
    start_time = time.time()
    input_dir = "/home/sdb/huangruiwei/data/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1"  # replace with your directory
    output_dir = "/home/sdb/huangruiwei/data/DigestPath2019/overlap128/images2"
    mask_output_dir = "/home/sdb/huangruiwei/data/DigestPath2019/overlap128/labels2"
    patch_size = 256
    overlap = 128
    threshold = 180  # adjust as required
    white_pixel_threshold = patch_size * patch_size * 0.95
    processes = min(cpu_count(), 4)  # by default, use 4 processes or less if CPU has fewer cores

    process_images(input_dir, output_dir, mask_output_dir, patch_size, overlap, threshold, processes)
    print((time.time() - start_time)/60, " min(s) ")
