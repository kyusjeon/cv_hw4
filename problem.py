import os
import cv2 
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def filter(weights,roi):
    weights = weights.astype(float)   
    roi = roi.astype(float)   
    filtered = np.zeros_like(roi)
    width = int((weights.shape[1]-1)/2)
    height = int((weights.shape[0]-1)/2)
    
    for i in range(height,roi.shape[1]-height):
        for j in range(width,roi.shape[0]-width):
            filtered[j,i] = (weights * roi[j-width:j+width+1, i-height:i+height+1]).sum()       # how do you create the output of the filtering?
    
    return  filtered

def gaussian_kernel(l):
    sig = int((l-1)/2)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def harris_oper(a,b,c):
    oper = np.nan_to_num((a * c - b * b) / (a + c))

    return oper

def sobel_filter(img):
    filter_sobelx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1],
    ])

    filter_sobely = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1],
    ])
    
    filtered_x = filter(filter_sobelx, img)
    filtered_y = filter(filter_sobely, img)
    
    return filtered_x, filtered_y
    
def corner_detect(filtered_x, filtered_y, win_size=7):
    window = gaussian_kernel(win_size)

    harris_a = filter(window, filtered_x*filtered_x)
    harris_b = filter(window, filtered_x*filtered_y)
    harris_c = filter(window, filtered_y*filtered_y)

    harris_h = harris_oper(harris_a, harris_b, harris_c)

    return harris_h

def get_orientation(filtered_x, filtered_y, win_size=7):
    window = np.ones((win_size, win_size))
    
    intence_x = filter(window, filtered_x)
    intence_y = filter(window, filtered_y)
    
    matrix_ori = np.arctan(intence_x / (intence_y + 1e-13))
    
    return matrix_ori

def load_image(root_path, image_path_list):
    image_list = list()
    for _i in image_path_list:
        _img = cv2.imread(os.path.join(root_path, _i))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _img = cv2.resize(_img, dsize=(800,1000))
        image_list.append(_img)
    
    return image_list

def non_maximum_suppression(feature_map, win_size=7):
    for _i in range(0,feature_map.shape[1]-win_size):
        for _j in range(0,feature_map.shape[0]-win_size):
            feature_map[_j:_j + win_size, _i:_i + win_size] = np.where(feature_map[_j:_j + win_size, _i:_i + win_size] == np.max(feature_map[_j:_j + win_size, _i:_i + win_size]), 
                                                                       feature_map[_j:_j + win_size, _i:_i + win_size], 
                                                                       0)
    
    return feature_map

def get_feature_point(filtered_x, filtered_y, win_size=7, thr=0.3):
    feature_map = corner_detect(filtered_x, filtered_y, win_size=win_size)
    feature_map = np.where(feature_map > feature_map.max()*thr, feature_map, 0)
    feature_map = non_maximum_suppression(feature_map, win_size=win_size)
    
    return np.argwhere(feature_map != 0), feature_map

def get_feature_descriptors(matrix_ori, feature_points, window_size=16):
    margin = window_size // 2
    high, width = matrix_ori.shape
    high -= margin
    width -= margin
    descript_points = list()
    for _i, _j in feature_points:
        if _i > margin and _j > margin and _i < high and _j < width:
            descript_points.append(matrix_ori[_i - margin:_i + margin, _j - margin:_j + margin])
      
    return descript_points

root_path = './image_folder'
image_path_list = sorted(os.listdir(root_path))
image_path_sample = list()
for _i, _j in enumerate(image_path_list):
    if _i % 8 == 0:
        image_path_sample.append(_j)

image_list = load_image(root_path, image_path_sample)
for _i in image_list:
    pass
 
def shift(des):
    pass

def match2images():
    pass

def ransac():
    pass

def refine_matchees():
    pass

def warp_images():
    pass

