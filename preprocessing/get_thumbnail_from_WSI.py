import openslide
from PIL import Image, ImageFile
import os
from multiprocessing import Pool, cpu_count
import time

Image.MAX_IMAGE_PIXELS = None

def process_image(filename, source_directory, dest_directory, max_size=5000):
    """
    Process a single image: generate thumbnail and save to destination directory.
    
    Parameters:
    - filename: Name of the tif file to process.
    - source_directory: The directory containing the tif file.
    - dest_directory: The directory to save the jpg thumbnail.
    - max_size: The maximum size (width or height) for the thumbnail.
    """
    file_path = os.path.join(source_directory, filename)
    try:
        with openslide.OpenSlide(file_path) as slide:
            # If only level 0 has data, it's likely a mask
            if len(slide.level_dimensions) == 1:
                thumbnail = Image.open(file_path)
            else:
                aspect_ratio = slide.dimensions[0] / slide.dimensions[1]
                if slide.dimensions[0] > slide.dimensions[1]:
                    new_width = max_size
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_size
                    new_width = int(new_height * aspect_ratio)
                thumbnail = slide.get_thumbnail((new_width, new_height))
            thumbnail.save(os.path.join(dest_directory, f"{os.path.splitext(filename)[0]}.jpg"), "JPEG")
            print(f"Done: {filename}")
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        # print(f"Use PIL file: {filename}")
        with Image.open(os.path.join(source_directory, filename)) as img:
            # Calculate aspect ratio
            aspect_ratio = img.width / img.height
            if img.width > img.height:
                new_width = max_size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(new_height * aspect_ratio)

            img.thumbnail((new_width, new_height), Image.NEAREST)
            # Save thumbnail to destination directory
            img.save(os.path.join(dest_directory, f"{os.path.splitext(filename)[0]}.jpg"), "JPEG")
            print(f"Done: {filename}")


def generate_thumbnails(source_directory, dest_directory, max_size=5000, processes=None):
    """
    Generate thumbnails for all tif files in the source directory and save them to the destination directory.
    
    Parameters:
    - source_directory: The directory containing the tif files.
    - dest_directory: The directory to save the jpg thumbnails.
    - max_size: The maximum size (width or height) for the thumbnail.
    - processes: Number of processes to use. If None, use all available CPUs.
    """
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    tif_files = [f for f in os.listdir(source_directory) if f.endswith(".tif")]
    
    # Use all available CPUs if processes is not specified
    if processes is None:
        processes = cpu_count()
    
    with Pool(processes=processes) as pool:
        pool.starmap(process_image, [(f, source_directory, dest_directory, max_size) for f in tif_files])


if __name__ == "__main__":
    start_time = time.time()
    src_dir = "/home/ubuntu/sdb/huangruiwei/data/camelyon16"
    dest_dir = "/home/ubuntu/sdb/huangruiwei/data/C16_thumbnail"
    num_processes = 4  # Adjust this value as needed
    generate_thumbnails(src_dir, dest_dir, processes=num_processes)
    print((time.time() - start_time)/60, " min(s) ")

