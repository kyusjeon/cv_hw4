import os
from platform import machine
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
        _img = cv2.resize(_img, dsize=(1000,800))
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
 
def shift(feature_point_map):
    if (16, 16) != feature_point_map.shape:
        raise ValueError("check your SHIFT input matrix")
    
    feature_vector = np.zeros((4,4,8))
    for _i in range(4):
        for _j in range(4):
            feature_vector[_i, _j, :] = np.histogram(feature_point_map[_i * 4:(_i+1) * 4,_j * 4:(_j+1) * 4], bins=8, range=(0, np.pi * 2))[0]
    
    return feature_vector

def get_feature_descriptors(matrix_ori, feature_points):
    margin = 8  # window_size 16
    high, width = matrix_ori.shape
    high -= margin
    width -= margin
    descript_points = list()
    feature_point_list = list()
    for _i, _j in feature_points:
        if _i > margin and _j > margin and _i < high and _j < width:
            feature_point_map = matrix_ori[_i - margin:_i + margin, _j - margin:_j + margin]
            feature_point_map = shift(feature_point_map)
            descript_points.append(np.ndarray.flatten(feature_point_map))
            feature_point_list.append(np.array([_i, _j]))
      
    return np.asarray(descript_points), np.asarray(feature_point_list)

def match2images(descript_points1, descript_points2, feature_points1 ,feature_points2 ,ratio = 0.9, type=1):
    if type == 0:
        dist = np.sum(((descript_points1[:, np.newaxis, ...] - descript_points2[np.newaxis, ...])**2)**(1/2), axis=2)
    elif type == 1:
        mean1 = np.mean(descript_points1, axis=1)
        mean2 = np.mean(descript_points2, axis=1)
        std1 = np.mean(descript_points1, axis=1)
        std2 = np.mean(descript_points2, axis=1)
        descript_points1_bar = descript_points1 - mean1[:, np.newaxis]
        descript_points2_bar = descript_points2 - mean2[:, np.newaxis]
        dist = np.sum(descript_points1_bar[:, np.newaxis, ...] * descript_points2_bar[np.newaxis, ...] / std1[:, np.newaxis, np.newaxis] / std2[np.newaxis, :, np.newaxis], axis=2)
        
    top_ind = np.argsort(dist, axis=1)
    
    matching_points1 = list()
    matching_points2 = list()
    _, ind2 = top_ind.shape
    for _i in range(ind2):
        if dist[_i, top_ind[_i, 0]] / dist[_i, top_ind[_i, 1]] <= ratio:
            matching_points1.append(feature_points1[_i])
            matching_points2.append(feature_points2[top_ind[_i, 0]])
    
    return np.asarray(matching_points1), np.asarray(matching_points2)

root_path = './png_folder'
image_path_list = sorted(os.listdir(root_path))
image_path_sample = list()
for _i, _j in enumerate(image_path_list):
    image_path_sample.append(_j)

image_list = load_image(root_path, image_path_sample)
for _i in image_list:
    pass

descript_points_list = list()
feature_points_list = list()
for i in range(2):
    filtered_x, filtered_y = sobel_filter(image_list[i])
    feature_points, feature_map = get_feature_point(filtered_x, filtered_y, win_size=15, thr = 0.01)
    matrix_ori = get_orientation(filtered_x, filtered_y)
    descript_points, feature_points  = get_feature_descriptors(matrix_ori, feature_points)
    descript_points_list.append(descript_points)
    feature_points_list.append(feature_points)
    
matching_points1, matching_points2 = match2images(descript_points_list[0], descript_points_list[1], feature_points_list[0], feature_points_list[1], ratio=0.9)
matching_points1 = np.stack([matching_points1[:,1], matching_points1[:,0]], axis=1)
matching_points2 = np.stack([matching_points2[:,1], matching_points2[:,0]], axis=1)

H, status = cv2.findHomography(matching_points2, matching_points1, cv2.RANSAC, 4.0)
result = cv2.warpPerspective(image_list[1], H, (2000, 1600))
result[0:800,0:1000] = image_list[0]

feature_points, feature_map = get_feature_point(filtered_x, filtered_y, win_size=15, thr = 0.01)
# feature map
plt.figure(dpi=400)
plt.imshow(image_list[1], cmap='gray')
for _x, _y in feature_points:
    plt.scatter(_y, _x, color='r', marker='.')
plt.show()

# point match map
img = np.concatenate((image_list[0], image_list[1]), axis=1)
plt.figure(dpi=400)
plt.imshow(img, cmap='gray')
plt.axis('off')
for _i in range(len(matching_points1)):
    x1, y1 = matching_points1[_i]
    plt.scatter(x1+1000, y1, color='r', marker='.')
    x2, y2 = matching_points2[_i]
    plt.scatter(x2, y2, color='r', marker='.')
    plt.plot([x1+1000, x2], [y1, y2], color="blue", linewidth=1)
plt.show()

def ransac():
    pass

def refine_matchees():
    pass

def warp_images():
    pass

#################
matching_points1, matching_points2 = match2images(descript_points_list[0], 
                                                  descript_points_list[1], 
                                                  feature_points_list[0], 
                                                  feature_points_list[1], 
                                                  ratio=0.9)
iter = 10000
sample_points = 4
for i in range(iter):
    random_indices = np.random.choice(np.arange(len(matching_points1)), sample_points)
    
    _matching_points1 = matching_points1[random_indices]
    _matching_points2 = matching_points2[random_indices]

    matrix_affine = list()
    for (y1, x1), (y2, x2) in zip(_matching_points1, _matching_points2):
        matrix_affine.append([x1, y1, 1, 0,  0,  0, -x2 * x1, -x2 * y1, -x2])
        matrix_affine.append([0,  0,  0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    matrix_affine = np.stack(matrix_affine)

    _, sigma, eigenvector = np.linalg.svd(matrix_affine)    # Singular Value Decomposition (SVD)

    homography = np.reshape(eigenvector[np.argmin(sigma)], (3, 3))
    homography /= homography[2, 2]    # normalize
    
    # Compute distances of selected corners
    _matching_points2 = np.concatenate([matching_points2, np.ones((len(matching_points2), 1))], axis=-1)    # homography coordinate
    trans_matching_points2 = np.dot(homography, _matching_points2.T).T
    trans_matching_points2 /= trans_matching_points2[:, [-1]]    # normalize
    trans_matching_points2 = trans_matching_points2[:, :2]
    
    distances = np.sqrt((trans_matching_points2 - matching_points1) ** 2).sum(axis=-1)
    plt.plot(distances,'.')
    plt.show()
    inliers1 = matching_points1[distances < 1000 * 0.05]
    inliers2 = matching_points2[distances < 1000 * 0.05]
    inlier_indices1 = list()
    inlier_indices2 = list()
    if len(inliers1) > len(inlier_indices1):
        inlier_indices1 = inliers1
        inlier_indices2 = inliers2
        result_homography = homography

result = cv2.warpPerspective(image_list[1], result_homography, (2000, 1600))
result[0:800,0:1000] = image_list[0]
plt.imshow(result)
  
height, width = image_list[0].shape[:2]
concat_imgs = np.concatenate([image_list[1], image_list[0]], axis=1)

_, ax = plt.subplots(dpi=200)

plt.axis("off")
plt.imshow(concat_imgs, cmap="gray")

# Draw corners
for _i in range(len(inlier_indices1)):
    y_i, x_i = inlier_indices1[_i]
    y_j, x_j = inlier_indices2[_i]
    ax.add_patch(
        mpl.patches.Circle(
            (x_i, y_i),                   # (x, y)
            5,
            fill=True,
    ))
    ax.add_patch(
        mpl.patches.Circle(
            (x_j + width, y_j),                   # (x, y)
            5,
            fill=True,
    ))
    ax.add_patch(
        mpl.patches.Arrow(
            x_i, y_i,
            x_j + width - x_i, y_j - y_i,
            width=5,
        )
    )

def warpImages(imgs, homographies):
    result_image = []
    for i in range(len(imgs) - 1):
        homography = homographies[i]
        for j in range(i + 1, len(imgs) - 1):
            homography = np.dot(homographies[j], homography)
        # transfomed_img = cv2.warpPerspective(imgs[i], homography, (imgs[i].shape[1], imgs[i].shape[0]))
        # transfomed_img = cv2.warpPerspective(imgs[i], homographies[i], (imgs[i].shape[1], imgs[i].shape[0]))
        result = cv2.warpPerspective(imgs[i+1], homographies[i],
            (imgs[i].shape[1] + imgs[i+1].shape[1], imgs[i].shape[0]))
        # result[0:imgs[i+1].shape[0], 0:imgs[i+1].shape[1]] = imgs[i]

        # print(transfomed_img.shape, imgs[i].shape)
        # plt.imshow(transfomed_img, cmap="gray")
        # plt.imshow(np.concatenate((imgs[i], transfomed_img, imgs[i+1]), axis=1), cmap="gray")
        plt.imshow(result, cmap="gray")
        plt.show()
        
warpImages([image_list[1], image_list[0]], [result_homography])