import numpy as np
import queue
from skimage import io

GTS_IND_LIST  = [3,5,11,13,17,21,23,26,28,30,32,34,37] 

MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"

GTS_FORMAT = "/gts_for_participants/top_mosaic_09cm_area{}.tif"
COV_RES = "/Output/class_dependence.txt"

categories = np.zeros((6, 3), dtype = 'uint8')
categories[0]  = [255, 255, 255]
categories[1]  = [0, 0, 255]
categories[2]  = [0, 255, 255]
categories[3]  = [0, 255, 0]
categories[4]  = [255, 255, 0]
categories[5]  = [255, 0, 0]

def get_gts(num):
    label_path = MAIN_PATH + "/gts_for_participants/top_mosaic_09cm_area{}.tif". format(num)
    res = io.imread(label_path)
    return res

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

def bfs(img):
    bfs_queue= queue.Queue()
    bfs_queue.put((0,0))
    visited = np.zeros((img.shape[0],img.shape[1],1), dtype = 'uint64')
    while not bfs_queue.empty():
        x,y = bfs_queue.get()
        if (x>0 and not visited[x-1][y]):
            if (img[x,y] != img[x-1,y]):
                neighbours[img[x,y], img[x-1,y]] += 1
            visited[x-1][y] = 1
            bfs_queue.put((x-1,y)) 
        if (y>0 and not visited[x][y-1]):
            if (img[x,y-1] != img[x,y]):
                neighbours[img[x,y], img[x,y-1]] += 1
            visited[x][y-1] = 1
            bfs_queue.put((x,y-1)) 
        if ((x!= (img.shape[0] - 1)) and not visited[x+1][y]):
            if (img[x+1,y] != img[x,y]):
                neighbours[img[x,y], img[x+1,y]] += 1
            visited[x+1][y] = 1
            bfs_queue.put((x+1,y)) 
        if ((y!= (img.shape[1] - 1)) and not visited[x][y+1]):
            if (img[x,y+1] != img[x,y]):
                neighbours[img[x,y], img[x,y+1]] += 1
            visited[x][y+1] = 1
            bfs_queue.put((x,y+1))
    return 0

neighbours = np.zeros((6, 6), dtype = 'uint64')
for i in GTS_IND_LIST:
    bfs(convert_color_to_labels(get_gts(i)))

neighbours = np.transpose(neighbours) + neighbours
res = np.divide(neighbours, np.sum(neighbours)/200)
np.savetxt(MAIN_PATH + COV_RES, res)