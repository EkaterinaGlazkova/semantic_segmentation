#get superpixels features and build adjacency graph

import numpy as np
import queue
from skimage import color
from skimage import io

PICS_FEATURES_NUM = 10
LABEL_NUM = 6
SEGM_NUM_LIST = [30000]
GTS_IND_LIST  = [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37] 
TEST_IND_LIST = [1,7,15]
TRAIN_IND_LIST = [i for i in GTS_IND_LIST if not i in TEST_IND_LIST]

MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"
TOP_FORMAT = "/top/top_mosaic_09cm_area{}.tif"
SLIC_RES_FORMAT = "/Output/SLIC/{}_area_{}_segm.txt"
GTS_FORMAT = "/gts_for_participants/top_mosaic_09cm_area{}.tif"
NDSM_FORMAT = "/ndsm/ndsm_09cm_matching_area{}.bmp"

FEATURES_FORMAT = "/Output/FEATURES/{}_area_{}_segm.txt"
LABELS_FORMAT = "/Output/LABELS/{}_area_{}_segm.txt"
ADJ_GRAPH_FORMAT = "/Output/ADJ_GRAPHS/{}_area_{}_segm.txt"


SEGM_MAX_NUM = 30000

categories = np.zeros((6, 3), dtype = 'uint8')
categories[0]  = [255, 255, 255]
categories[1]  = [0, 0, 255]
categories[2]  = [0, 255, 255]
categories[3]  = [0, 255, 0]
categories[4]  = [255, 255, 0]
categories[5]  = [255, 0, 0]

def show_pic(pic):
    io.imshow(pic)
    io.show()

def combine_features(source1, source2, source3, source4):
    data = np.zeros((source1.shape[0],source1.shape[1], 10), dtype = 'float64')
    data[:,:,0:3] = source1[:,:,:]
    data[:,:,3:6] = source2[:,:,:]
    data[:,:,6:9] = source3[:,:,:]
    data[:,:,9] = source4[:,:]
    return data

def get_gts(num):
    label_path = MAIN_PATH + GTS_FORMAT. format(num)
    res = io.imread(label_path)
    return res

def get_top(num):
    top_path = MAIN_PATH + TOP_FORMAT. format(num)
    top = io.imread(top_path)
    return top

def get_ndsm(num):
    dsm_path = MAIN_PATH + NDSM_FORMAT. format(num)
    res = io.imread(dsm_path)
    return res

def get_slic_res(num, seg_num):
    res_path = MAIN_PATH + SLIC_RES_FORMAT.format(num, seg_num)
    res = np.loadtxt(res_path, dtype = "int")
    return res

def get_hsv(image):
    return color.rgb2hsv(image)

def get_lab(image):
    return  color.rgb2lab(image)

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

def bfs():
    global raw_features
    while not bfs_queue.empty():
        x,y = bfs_queue.get()
        raw_features = np.concatenate((raw_features, [img_data[x][y]]))
        label_array[labels_map[x][y]] +=1
        if (x>0 and visited[x-1][y]): 
            if (segm_res[x-1][y] == segm_res[x][y]):
                visited[x-1][y] = 0
                bfs_queue.put((x-1,y))
            else:
                adj_graph[segm_res[x-1][y],segm_res[x][y]] = 1
        if (y>0 and visited[x][y-1]):
            if (segm_res[x][y-1] == segm_res[x][y]):
                visited[x][y-1] = 0
                bfs_queue.put((x,y-1)) 
            else:
                adj_graph[segm_res[x][y-1],segm_res[x][y]] = 1
        if ((x!= (segm_res.shape[0] - 1)) and visited[x+1][y]):
            if (segm_res[x+1][y] == segm_res[x][y]):
                visited[x+1][y] = 0
                bfs_queue.put((x+1,y)) 
            else:
                adj_graph[segm_res[x+1][y],segm_res[x][y]] = 1
        if ((y!= (segm_res.shape[1] - 1)) and visited[x][y+1]):
            if (segm_res[x][y+1] == segm_res[x][y]):
                visited[x][y+1] = 0
                bfs_queue.put((x,y+1))
            else:
                adj_graph[segm_res[x][y+1],segm_res[x][y]] = 1
    return 0

for ind in GTS_IND_LIST:
    print(ind)
    
    ndsm = get_ndsm(ind)
    rgb = get_top(ind)
    hsv = get_hsv(rgb)
    lab = get_lab(rgb)
    img_data = combine_features(rgb,hsv,lab,ndsm)
    
    segm_res = get_slic_res(ind, SEGM_MAX_NUM)
    segm_num = np.amax(segm_res) + 1
    
    adj_graph = np.zeros((segm_num, segm_num), dtype = 'uint8')
    
    labels_map = convert_color_to_labels(get_gts(ind))
    segments_labels = np.zeros((segm_num), dtype = 'uint8')
    
    visited = np.ones(rgb.shape[:2])
    
    segments_features = np.zeros((segm_num, PICS_FEATURES_NUM*2), dtype = 'float64') 
    
    while len(np.nonzero(visited)[0]) > 0:
        start = np.nonzero(visited)
        label_array = np.zeros((LABEL_NUM), dtype = 'uint32') 
        raw_features = np.zeros((0, PICS_FEATURES_NUM), dtype = 'float64') 
        
        bfs_queue = queue.Queue()
        bfs_queue.put((start[0][0], start[1][0]))
        bfs()
        segments_labels[segm_res[start[0][0], start[1][0]]] = np.argmax(label_array)
            
        for i in range(10):  
            segments_features[segm_res[start[0][0], start[1][0]],i] = np.mean(raw_features[:,i])
            
        for i in range(10, 20):   
            raw_features[:,i-10] = (raw_features[:,i-10] - segments_features[segm_res[start[0][0], start[1][0]],i-10])**2
            segments_features[segm_res[start[0][0], start[1][0]],i] = np.mean(raw_features[:,i-10])
            
    adj_graph = adj_graph + np.transpose(adj_graph)
    np.clip(adj_graph, 0, 1)

    np.savetxt(MAIN_PATH + FEATURES_FORMAT.format(ind,SEGM_MAX_NUM), segments_features)
    np.savetxt(MAIN_PATH + LABELS_FORMAT.format(ind,SEGM_MAX_NUM), segments_labels, fmt='%d')
    np.savetxt(MAIN_PATH + ADJ_GRAPH_FORMAT.format(ind,SEGM_MAX_NUM), adj_graph, fmt='%d')