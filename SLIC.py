#creates superpicsels using SLIC
import numpy as np
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"
TOP_FORMAT = "/top/top_mosaic_09cm_area{}.tif"
SLIC_RES_FORMAT = "/Output/SLIC/{}_area_{}_segm.txt"
SLIC_WITH_BOUNDARIES_FORMAT = "/Output/SLIC/{}_area_{}_segm_boundaries.jpg"

SEGM_NUM_LIST = [300000]
ALL_IND_LIST = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,37,38]

MAX_ITER = 30
COMPACTNESS = 15

def get_top(num):
    top_path = MAIN_PATH + TOP_FORMAT. format(num)
    top = io.imread(top_path)
    return top

for ind in ALL_IND_LIST:
    top = get_top(ind)
    for max_seg in SEGM_NUM_LIST:
        slic_res = slic(top, 
                        n_segments = max_seg, 
                        max_iter = MAX_ITER,
                        compactness=COMPACTNESS)
        
        np.savetxt(MAIN_PATH + SLIC_RES_FORMAT.format(ind, max_seg), slic_res, fmt='%d')

        io.imsave(MAIN_PATH + SLIC_WITH_BOUNDARIES_FORMAT.format(ind, max_seg), mark_boundaries(top, slic_res))