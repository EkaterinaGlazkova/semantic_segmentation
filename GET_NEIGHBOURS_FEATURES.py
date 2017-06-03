#extracts neighbours features from all features and adjacent graph

import numpy as np
from skimage import io

FEATURES_NUM = 20
LABEL_NUM = 6
GTS_IND_LIST  = [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37] 

MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"

FEATURES_FORMAT = "/Output/FEATURES/{}_area_{}_segm.txt"
FEATURES_WITH_NEIGHB_FORMAT = "/Output/FEATURES_WITH_NEIGHBOURS/{}_area_{}_segm.txt"
ADJ_GRAPH_FORMAT = "/Output/ADJ_GRAPHS/{}_area_{}_segm.txt"

SEGM_MAX_NUM = 30000

def get_own_features(num, segm_num):
    return np.loadtxt(MAIN_PATH + FEATURES_FORMAT.format(num, segm_num), dtype = 'float64')

def get_adj_gr(num, segm_num):
    return np.loadtxt(MAIN_PATH + ADJ_GRAPH_FORMAT.format(num, segm_num), dtype = 'uint8')

for i in GTS_IND_LIST:
    print("Loading features for ", i)
    segm_features = get_own_features(i, SEGM_MAX_NUM)
    segm_features = np.append(segm_features, np.zeros((segm_features.shape)), axis=1)
    print("Loading adjacent graph")
    adj_graph = get_adj_gr(i, SEGM_MAX_NUM)
    segm_num = segm_features.shape[0]
    for segm_ind in range(segm_num):
        neighbours = np.nonzero(adj_graph[segm_ind, :])
        for neighb_ind in neighbours[0]:
            segm_features[segm_ind, FEATURES_NUM:] += segm_features[neighb_ind, :FEATURES_NUM]
        segm_features[segm_ind, FEATURES_NUM:] /= len(neighbours[0])
    np.savetxt(MAIN_PATH + FEATURES_WITH_NEIGHB_FORMAT.format(i, SEGM_MAX_NUM), segm_features)