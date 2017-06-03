#counts accuracy loss after segmentation
import numpy as np
from skimage import io

MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"
GTS_FORMAT = "/gts_for_participants/top_mosaic_09cm_area{}.tif"
SLIC_RES_FORMAT = "/Output/SLIC/{}_area_{}_segm.txt"
TOP_FORMAT = "/top/top_mosaic_09cm_area{}.tif"

RES_FILE = "/Output/SLIC/acc_loss_res.txt"

RES_FORMAT = "AREA_IND = {} SEGM_MAX_NUM = {} SEGM_REAL_NUM = {} IMG_SIZE = {} WRONG_LABELS = {} LOSS_PERCENTAGE = {}\n"

LABEL_NUM = 6
SEGM_NUM_LIST = [30000]
ALL_IND_LIST = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,37,38]
GTS_IND_LIST = [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]

categories = np.zeros((6, 3), dtype = 'uint8')
categories[0]  = [255, 255, 255]
categories[1]  = [0, 0, 255]
categories[2]  = [0, 255, 255]
categories[3]  = [0, 255, 0]
categories[4]  = [255, 255, 0]
categories[5]  = [255, 0, 0]

#gets array with labels from gts
def convert_color_to_labels(color_map):
    label_map = np.zeros((color_map.shape[0],color_map.shape[1]), dtype = 'uint8')
    for i in range(color_map.shape[0]):
        for j in range (color_map.shape[1]):
            pics = color_map[i][j][:]
            for k in range(6):
                if np.array_equal(pics, categories[k, :]):
                    label_map[i][j] = k
                    break
    return label_map

def get_gts(num):
    label_path = MAIN_PATH + GTS_FORMAT. format(num)
    res = io.imread(label_path)
    return res

def get_slic_res(num, seg_num):
    res_path = MAIN_PATH + SLIC_RES_FORMAT.format(num, seg_num)
    res = np.loadtxt(res_path, dtype = "int")
    return res

#counts accuracy loss after segmentation
def find_loss(ind, segm_num):
    segm_res = get_slic_res(ind, segm_num)
    img_size = segm_res.shape[0]*segm_res.shape[1]

    real_segm_num = np.amax(segm_res) + 1
    hystogram = np.zeros((real_segm_num, LABEL_NUM), dtype = 'uint64')

    for i in range(segm_res.shape[0]):
        for j in range(segm_res.shape[1]):
            hystogram[segm_res[i][j]][ground_truth_labels[i][j]] += 1
        
    loss_res = 0
    for i in range(real_segm_num):
        loss_res += np.sum(hystogram[i]) - np.amax(hystogram[i])

    res_file.write(RES_FORMAT.format(ind, segm_num, real_segm_num, img_size, loss_res, loss_res*100/ img_size))

for ind in GTS_IND_LIST:
    res_file = open(MAIN_PATH + RES_FILE, 'a')
    ground_truth_labels = convert_color_to_labels(get_gts(ind))
    for i in SEGM_NUM_LIST:
        find_loss(ind, i)
    res_file.close()