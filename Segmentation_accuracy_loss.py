import numpy as np
from skimage import io

LABEL_NUM = 6
#SEGM_NUM_LIST = [25000]
SEGM_NUM_LIST = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
IND_LIST = [3]
MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"
GROUND_TRUTH_NAME_PATTERN = "/gts_for_participants/top_mosaic_09cm_area{}.tif"
SLIC_RESULT_PATTERN = "/Output/SLIC/SLIC_{}_area_mask_{}_segments.png"

def convert_color_to_labels(color_map):
    categories = np.zeros((6, 3), dtype = 'uint8')
    categories[0]  = [255, 255, 255]
    categories[1]  = [0, 0, 255]
    categories[2]  = [0, 255, 255]
    categories[3]  = [0, 255, 0]
    categories[4]  = [255, 255, 0]
    categories[5]  = [255, 0, 0]

    label_map = np.zeros((color_map.shape[0],color_map.shape[1]), dtype = 'uint8')
    for i in range(color_map.shape[0]):
        for j in range (color_map.shape[1]):
            pics = color_map[i][j][:]
            for k in range(6):
                if (pics[0] == categories[k][0]) and (pics[1] == categories[k][1]) and (pics[2] == categories[k][2]):
                    label_map[i][j] = k
                    break
    
    return label_map

def show_pic(pic):
    io.imshow(pic)
    io.show()

def get_label_pic(num):
    label_path = MAIN_PATH + "/gts_for_participants/top_mosaic_09cm_area{}.tif". format(num)
    res = io.imread(label_path)
    return res

def find_loss(ind, segm_num):
    segm_res = io.imread(MAIN_PATH + SLIC_RESULT_PATTERN.format(ind, segm_num))
    pics_num = segm_res.shape[0]*segm_res.shape[1]

    real_segm_num = np.amax(segm_res) + 1
    hystogram = np.zeros((real_segm_num, LABEL_NUM,1), dtype = 'uint32')

    for i in range(segm_res.shape[0]):
        for j in range(segm_res.shape[1]):
            hystogram[segm_res[i][j]][ground_truth_labels[i][j]] += 1
        
    loss_res = 0
    for i in range(real_segm_num):
        loss_res += np.sum(hystogram[i]) - np.amax(hystogram[i])

    print("SEGM_MAX_NUM =", ind, "SEGM_REAL_NUM =", segm_num, "LOSS_RES = ", loss_res, 
          "PICS_NUM = ", pics_num, "LOSS_PERCENTAGE = ", loss_res*100 / pics_num)

for ind in IND_LIST:
    ground_truth_colors = io.imread(GROUND_TRUTH_NAME_PATTERN.format(ind))
    ground_truth_labels = convert_color_to_labels(ground_truth_colors)
    for i in SEGM_NUM_LIST:
        find_loss(ind, i)