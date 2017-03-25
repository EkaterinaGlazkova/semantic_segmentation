import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"

SEGM_NUM_LIST = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
IND_LIST = [5, 23, 26, 37]

MAX_ITER = 30
SIGMA = 0

def get_rgb_pic(num):
    rgb_path = MAIN_PATH + "/top/top_mosaic_09cm_area{}.tif". format(num)
    res = io.imread(rgb_path)
    return res

for ind in IND_LIST:
    rgb = get_rgb_pic(ind)
    for max_seg in SEGM_NUM_LIST:

        segments = slic(rgb, 
                    n_segments = max_seg, 
                    max_iter = MAX_ITER, 
                    sigma = SIGMA)

        io.imsave(MAIN_PATH + "/Output/SLIC/SLIC_{}_area_mask_{}_segments.png".format(ind, max_seg), segments)

        io.imsave(MAIN_PATH + "/Output/SLIC/SLIC_{}_area_with_boundaries_{}_segments.png".format(ind, max_seg), 
                  mark_boundaries(rgb, segments))