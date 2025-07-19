# Here's the full code for your task

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import openslide
from multiprocessing import Pool

def xml2contours(xml_name, base=1):
    tree = ET.parse(xml_name)
    root = tree.getroot()
    res = []
    for coordinate in root.iter('Coordinates'):
        arr = []
        for child in coordinate.iter('Coordinate'):
            arr.append((float(child.attrib['X']) / base, float(child.attrib['Y']) / base))
        res.append(np.array(arr, np.int32))
    return tuple(res)

def xml2mask(xml_name, shape, base=1):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, xml2contours(xml_name, base), 255)
    return mask

def read_svs(svs_name, level=2, needs=None, mode='L'):
    def process_needs(needs):
        res = []
        for need in needs:
            if need == 'shape':
                res.append(slide.dimensions)
            elif need == 'base':
                res.append(round(slide.level_downsamples[level]))
        return res

    with openslide.open_slide(svs_name) as slide:
        img = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert(mode))
        return (img, *process_needs(needs)) if needs else img

def thresholding(img, threshold=None):
    if not threshold:
        val, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th, int(val) + 1
    return cv2.threshold(img, threshold - 1, 255, cv2.THRESH_BINARY)[1]

def gen_mask_for_file(file_tuple):
    svs_name, xml_name, out_name, level, threshold = file_tuple
    img, shape, base = read_svs(svs_name, level, ['shape', 'base'])
    mask = xml2mask(xml_name, (img.shape[0], img.shape[1]), base) & ~thresholding(img, threshold)
    mask[mask >= 127] = 255
    mask[mask < 127] = 0
    mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_name, mask)
    print(out_name)

def main(directory, level=2, threshold=200, num_processes=4):
    files = os.listdir(directory)
    tif_files = [f for f in files if f.endswith('.tif') and not f.endswith('_mask.tif')]
    file_tuples = []

    for tif_file in tif_files:
        xml_name = os.path.splitext(tif_file)[0] + '.xml'
        if xml_name in files:
            out_name = os.path.join(directory, os.path.splitext(tif_file)[0] + '_mask.tif')
            file_tuples.append((os.path.join(directory, tif_file), os.path.join(directory, xml_name), out_name, level, threshold))

    with Pool(num_processes) as p:
        p.map(gen_mask_for_file, file_tuples)

import time
start_time = time.time()
print((time.time() - start_time)/60, " min(s) ")


main('/home/ubuntu/sdb/huangruiwei/data/camelyon16', level=0, threshold=200, num_processes=4)
