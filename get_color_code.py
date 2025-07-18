import os
import cv2
import numpy as np

def calculate_lab_stats(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mu_L, sigma_L = np.mean(L), np.std(L)
    mu_A, sigma_A = np.mean(A), np.std(A)
    mu_B, sigma_B = np.mean(B), np.std(B)
    return mu_L, sigma_L, mu_A, sigma_A, mu_B, sigma_B

def process_images(input_dir, output_txt):
    with open(output_txt, 'w') as f:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(input_dir, filename)
                stats = calculate_lab_stats(image_path)
                if stats:
                    line = ','.join(f"{x:.4f}" for x in stats) + f",{filename}\n"
                    f.write(line)
    print(f"Process done. Save as {output_txt}")


input_directory = './train_folder'   # 替换为你的图片目录
output_file = './color_code.txt'         # 替换为你要保存的输出文件路径

process_images(input_directory, output_file)
