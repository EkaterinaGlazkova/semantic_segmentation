import sklearn
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import queue
from PIL import Image
from skimage import color
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

PICS_FEATURES_NUM = 10
LABEL_NUM = 6
SEGM_NUM_LIST = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
IND_LIST = [5]
#IND_LIST = [1,3,5,11,13,17,21,23,26,28,30,32,34,37] #ideal
MAIN_PATH = "/Users/ekaterina/Documents/Semantic Segmentation/ISPRS_semantic_labeling_Vaihingen"
GROUND_TRUTH_NAME_PATTERN = "/gts_for_participants/top_mosaic_09cm_area{}.tif"
SLIC_RESULT_PATTERN = "/Output/SLIC/SLIC_{}_area_mask_{}_segments.png"
SEGM_MAX_NUM = 300
TEST_PICS = [1,7,10,12,15]

def combine_pics_features(x,y, rgb_source, hsv_source, lab_source, dsm_source):
    data = np.zeros((PICS_FEATURES_NUM), dtype = 'float64')
    data[0:3] = rgb_source[x][y][:]
    data[3:6] = hsv_source[x][x][:]
    data[6:9] = lab_source[i][j][:]
    data[9] = dsm_source[x][y]
    return data

def combine_features(source1, source2, source3, source4):
    data = np.zeros((source1.shape[0],source1.shape[1], 10), dtype = 'float64')
    for i in range(source1.shape[0]):
        for j in range (source1.shape[1]):
            data[i][j][0:3] = source1[i][j][:]
            data[i][j][3:6] = source2[i][j][:]
            data[i][j][6:9] = source3[i][j][:]
            data[i][j][9] = source4[i][j]
    return data

def get_label_pic(num):
    label_path = MAIN_PATH + "/gts_for_participants/top_mosaic_09cm_area{}.tif". format(num)
    res = io.imread(label_path)
    return res

def get_rgb_pic(num):
    rgb_path = MAIN_PATH + "/top/top_mosaic_09cm_area{}.tif". format(num)
    res = io.imread(rgb_path)
    return res

def get_dsm_pic(num):
    dsm_path = MAIN_PATH + "/dsm/dsm_09cm_matching_area{}.tif". format(num)
    res = io.imread(dsm_path)
    return res

def get_mask(num):
    mask_path = MAIN_PATH + "/Output/SLIC/SLIC_{}_area_mask_{}_segments.png". format(num,SEGM_MAX_NUM)
    res = io.imread(mask_path)
    return res

def get_hsv(image):
    return color.rgb2hsv(image)

def get_lab(image):
    return  color.rgb2lab(image)

def dfs_label():
    global segment_array, pics_data, label_array, labels_map, visited, mask 
    while not dfs_queue.empty():
        x,y = dfs_queue.get()
        segment_array= np.concatenate((segment_array, [pics_data[x][y]]))
        label_array[labels_map[x][y]] +=1
        if (x>0 and visited[x-1][y] and mask[x-1][y] == mask[x][y]):
            visited[x-1][y] = 0
            dfs_queue.put((x-1,y)) 
        if (y>0 and visited[x][y-1] and mask[x][y-1] == mask[x][y]):
            visited[x][y-1] = 0
            dfs_queue.put((x,y-1)) 
        if ((x!= (pics_data.shape[0] - 1)) and visited[x+1][y] and mask[x+1][y] == mask[x][y]):
            visited[x+1][y] = 0
            dfs_queue.put((x+1,y)) 
        if ((y!= (pics_data.shape[1] - 1)) and visited[x][y+1] and mask[x][y+1] == mask[x][y]):
            visited[x][y+1] = 0
            dfs_queue.put((x,y+1))
    return 0

def dfs_simple():
    global segment_array, pics_data, visited, mask 
    while not dfs_queue.empty():
        x,y = dfs_queue.get()
        segment_array= np.concatenate((segment_array, [pics_data[x][y]]))
        if (x>0 and visited[x-1][y] and mask[x-1][y] == mask[x][y]):
            visited[x-1][y] = 0
            dfs_queue.put((x-1,y)) 
        if (y>0 and visited[x][y-1] and mask[x][y-1] == mask[x][y]):
            visited[x][y-1] = 0
            dfs_queue.put((x,y-1)) 
        if ((x!= (pics_data.shape[0] - 1)) and visited[x+1][y] and mask[x+1][y] == mask[x][y]):
            visited[x+1][y] = 0
            dfs_queue.put((x+1,y)) 
        if ((y!= (pics_data.shape[1] - 1)) and visited[x][y+1] and mask[x][y+1] == mask[x][y]):
            visited[x][y+1] = 0
            dfs_queue.put((x,y+1))
    return 0

global pics_data
global segment_array
global visited, mask, label_array, labels_map
global dfs_queue
dfs_queue= queue.Queue()
def get_pics_segments_features(pic_ind, flag = 0):
    global pics_data, segment_array, visited, mask, label_array, labels_map, dfs_queue
    
    dsm = get_dsm_pic(pic_ind)
    rgb = get_rgb_pic(pic_ind)
    hsv = get_hsv(rgb)
    lab = get_lab(rgb)
    pics_data = combine_features(rgb,hsv,lab,dsm)
    mask = get_mask(pic_ind)
    segm_num = len(np.unique(mask))
    print("Segments number is {} for {} area.".format(segm_num, pic_ind))
    if flag:
        labels_map = get_label_pic(pic_ind)
        labels_map = convert_color_to_labels(labels_map)
        labels = np.zeros((segm_num, LABEL_NUM), dtype = 'uint8')
     
    visited = np.ones(rgb.shape[:2])
    segments = np.zeros((segm_num, PICS_FEATURES_NUM*2), dtype = 'float64') 

    while len(np.nonzero(visited)[0]) > 0:
        start = np.nonzero(visited)
        print(start)
        if flag:
            label_array = np.zeros((LABEL_NUM), dtype = 'uint32') 
        segment_array = np.zeros((0, PICS_FEATURES_NUM), dtype = 'float64') 
        
        if flag:
            dfs_queue.put((start[0][0], start[1][0]))
            dfs_label()
            labels[mask[start[0][0], start[1][0]]] = np.argmax(label_array)
        else:
            dfs_simple(start[0][0], start[1][0])
            
        for i in range(10):       
            segments[mask[start[0][0], start[1][0]]][i] = np.mean(segment_array[:][i])
            
        for i in range(10, 20):   
            segment_array[:][i-10] = (segment_array[:][i-10] - segments[mask[start[0][0], start[1][0]]][i-10])**2
            segments[mask[start[0][0], start[1][0]]][i] = np.mean(segment_array[:][i-10])
        
    if flag:
        return segments, labels
    else:
        return segments


 def get_segments_features(pics_list,flag = 0):
    segments_features = np.zeros((0, PICS_FEATURES_NUM*2), dtype = 'float64') 
    if flag:
        segments_labels = np.zeros((0,1), dtype = 'uint8')
    for index in pics_list:
        if flag:
            res1, res2 = get_pics_segments_features(index, flag)
            segments_features = np.concatenate((segments_features, res1))
            segments_labels = np.concatenate((segments_labels, res2))
        else:
            segments_features = np.concatenate(segments_features, get_pics_segments_features(index));
    if flag:
        return segments_features, segments_labels
    return segments_features


def convert_labels_to_color(label_map):
    categories = np.zeros((6, 3), dtype = 'uint8')
    categories[0]  = [255, 255, 255]
    categories[1]  = [0, 0, 255]
    categories[2]  = [0, 255, 255]
    categories[3]  = [0, 255, 0]
    categories[4]  = [255, 255, 0]
    categories[5]  = [255, 0, 0]

    color_map = np.zeros((label_map.shape[0],label_map.shape[1], 3), dtype = 'uint8')
    for i in range(label_map.shape[0]):
        for j in range (label_map.shape[1]):
            for k in range(6):
                if label_map[i][j] == k:
                    color_map[i][j] = categories[k]
                    break
                    
    return color_map

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


def make_data_plain(data, features_num):
    plain_data = data.copy()
    plain_data.resize(data.shape[0]*data.shape[1], features_num)
    return plain_data

train_data, train_labels = get_segments_features(IND_LIST, 1)

clf = RandomForestClassifier(n_estimators=TREES_NUM, n_jobs = JOBS_NUM)

clf.fit(train_data, train_labels)

for j in TEST_PICS:
    test_data, test_labels = get_segments_features(j, 1)
    predicted_plain_labels = clf.predict(test_data)
    predicted_labels = predicted_plain_labels.copy()
    predicted_labels.resize(rgb.shape[0],rgb.shape[1])
    predicted_pic = convert_labels_to_color(predicted_labels)
    io.imsave(MAIN_PATH + "Output/RES/SLIC_Predicted labels for {}.jpg".format(j), predicted_pic)
    show_pic(predicted_pic)
    test_label_map = get_label_pic(j)
    show_pic(test_label_map)
    test_label_map = make_data_plain(test_label_map)
    f1_score(test_label_map, predicted_plain_labels, labels = [0,1,2,3,4,5,6], average = 'micro')