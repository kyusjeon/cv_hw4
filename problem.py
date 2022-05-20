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

def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
  

def load_image(root_path, image_path_list, focal):
    K = np.array([[focal,0,1250/2],[0,focal,1000/2],[0,0,1]]) 
    image_list = list()
    for _i in image_path_list:
        _img = cv2.imread(os.path.join(root_path, _i))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _img = cv2.resize(_img, dsize=(1250,1000))
        _img = cylindricalWarp(_img, K)[...,0]
        image_list.append(_img[100:900, 125:1125])
    
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

def match2images(descript_points1, descript_points2, feature_points1 ,feature_points2 ,ratio = 0.9, type=0):
    if type == 0:
        dist = np.sum(((descript_points1[:, np.newaxis, ...] - descript_points2[np.newaxis, ...])**2)**(1/2), axis=2)
    elif type == 1:
        mean1 = np.mean(descript_points1, axis=1)
        mean2 = np.mean(descript_points2, axis=1)
        std1 = np.mean(descript_points1, axis=1)
        std2 = np.mean(descript_points2, axis=1)
        descript_points1_bar = descript_points1 - mean1[:, np.newaxis]
        descript_points2_bar = descript_points2 - mean2[:, np.newaxis]
        sim = np.mean(descript_points1_bar[:, np.newaxis, ...] * descript_points2_bar[np.newaxis, ...] , axis=-1)/ (std1[:, np.newaxis] * std2[np.newaxis, :])
        dist = (sim - 1.) / 2
        
    top_ind = np.argsort(dist, axis=1)
    
    matching_points1 = list()
    matching_points2 = list()
    ind1, _ = top_ind.shape
    for _i in range(ind1):
        if dist[_i, top_ind[_i, 0]] / dist[_i, top_ind[_i, 1]] <= ratio:
            matching_points1.append(feature_points1[_i])
            matching_points2.append(feature_points2[top_ind[_i, 0]])

    return np.asarray(matching_points1)[:,::-1], np.asarray(matching_points2)[:,::-1]

def refine_matchees(matching_points1, matching_points2, iter = 1000, sample_points = 4):
    inlier_indices1 = list()
    for i in range(iter):
        random_indices = np.random.choice(np.arange(len(matching_points1)), sample_points)
        
        _matching_points1 = matching_points1[random_indices]
        _matching_points2 = matching_points2[random_indices]

        matrix_affine = list()
        for (x1, y1), (x2, y2) in zip(_matching_points2, _matching_points1):
            matrix_affine.append([x1, y1, 1, 0,  0,  0, -x2 * x1, -x2 * y1, -x2])
            matrix_affine.append([0,  0,  0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        matrix_affine = np.stack(matrix_affine)

        _, _, eigenvector = np.linalg.svd(matrix_affine)    # Singular Value Decomposition (SVD)

        _homography = np.reshape(eigenvector[-1], (3, 3))
        _homography /= _homography[2, 2] 
        
        _matching_points2 = np.concatenate([matching_points2, np.ones((len(matching_points2), 1))], axis=-1)    # homography coordinate
        trans_matching_points2 = np.dot(_homography, _matching_points2.T).T
        trans_matching_points2 /= trans_matching_points2[:, [-1]]    # normalize
        trans_matching_points2 = trans_matching_points2[:, :2]
        
        distances = np.sqrt((trans_matching_points2 - matching_points1) ** 2).sum(axis=-1)
        
        inliers1 = matching_points1[distances < 50]
        inliers2 = matching_points2[distances < 50]
        if len(inliers1) > len(inlier_indices1):
            inlier_indices1 = inliers1
            inlier_indices2 = inliers2
            result_homography = _homography
    
    return result_homography

def warp_images(image1, image2, homography):
    h, w = image1.shape
    result = cv2.warpPerspective(image2, homography, (int(w + 400), 1000))
    result[0:h,0:w] = np.where(image1 !=0,image1, result[0:h,0:w])
    
    return result

root_path = './png_folder'
image_path_list = sorted(os.listdir(root_path))
image_path_sample = list()
for _i, _j in enumerate(image_path_list):
    image_path_sample.append(_j)

image_list = load_image(root_path, image_path_sample, focal=700)
for _i in image_list:
    pass

descript_points_list = list()
feature_points_list = list()
for i in range(len(image_list)):
    filtered_x, filtered_y = sobel_filter(image_list[i])
    feature_points, feature_map = get_feature_point(filtered_x, filtered_y, win_size=15, thr = 0.01)
    matrix_ori = get_orientation(filtered_x, filtered_y)
    descript_points, feature_points  = get_feature_descriptors(matrix_ori, feature_points)
    descript_points_list.append(descript_points)
    feature_points_list.append(feature_points)

homo_list = list()
_image = image_list[0]
for _i in range(len(image_list)-1):
    matching_points1, matching_points2 = match2images(descript_points_list[_i], descript_points_list[_i+1], feature_points_list[_i], feature_points_list[_i+1], 
                                                    ratio=0.7,
                                                    type=0)
    homo = refine_matchees(matching_points1, matching_points2)
    if _i != 0:
        homo_list.append(np.matmul(homo_list[-1], homo))
    else:
        homo_list.append(homo)
    _image = warp_images(_image, image_list[_i +1], homo_list[-1])

plt.imshow(_image)
plt.imshow()
###########################################
H, status = cv2.findHomography(matching_points2, matching_points1, cv2.RANSAC, 4.0)
result = cv2.warpPerspective(image_list[1], H, (1500, 1000))
result[0:800,0:1000] = image_list[0]
plt.imshow(result)

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
    y1, x1 = matching_points1[_i]
    plt.scatter(x1, y1, color='r', marker='.')
    y2, x2 = matching_points2[_i]
    plt.scatter(x2+1000, y2, color='r', marker='.')
    plt.plot([x1, x2+1000], [y1, y2], color="blue", linewidth=1)
plt.show()

a = np.abs(matching_points1 - matching_points2)
plt.plot(a[:,1], a[:,0], '.')


#################
matching_points1, matching_points2 = match2images(descript_points_list[0], 
                                                  descript_points_list[1], 
                                                  feature_points_list[0], 
                                                  feature_points_list[1], 
                                                  ratio=0.8,
                                                  type=0)
print(len(matching_points1))
iter = 10000
sample_points = 4
for i in range(iter):
    random_indices = np.random.choice(np.arange(len(matching_points1)), sample_points)
    
    _matching_points1 = matching_points1[random_indices]
    _matching_points2 = matching_points2[random_indices]

    matrix_affine = list()
    for (x1, y1), (x2, y2) in zip(_matching_points2, _matching_points1):
        matrix_affine.append([x1, y1, 1, 0,  0,  0, -x2 * x1, -x2 * y1, -x2])
        matrix_affine.append([0,  0,  0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    matrix_affine = np.stack(matrix_affine)

    _, _, eigenvector = np.linalg.svd(matrix_affine)    # Singular Value Decomposition (SVD)

    _homography = np.reshape(eigenvector[-1], (3, 3))
    _homography /= _homography[2, 2] 
    
    _matching_points2 = np.concatenate([matching_points2, np.ones((len(matching_points2), 1))], axis=-1)    # homography coordinate
    trans_matching_points2 = np.dot(_homography, _matching_points2.T).T
    trans_matching_points2 /= trans_matching_points2[:, [-1]]    # normalize
    trans_matching_points2 = trans_matching_points2[:, :2]
    
    distances = np.sqrt((trans_matching_points2 - matching_points1) ** 2).sum(axis=-1)
    
    inliers1 = matching_points1[distances < 50]
    inliers2 = matching_points2[distances < 50]
    inlier_indices1 = list()
    if len(inliers1) > len(inlier_indices1):
        inlier_indices1 = inliers1
        inlier_indices2 = inliers2
        result_homography = _homography
result_homography = refine_matchees(matching_points1, matching_points2)
result = cv2.warpPerspective(image_list[1], result_homography, (2000, 800))
result[0:800,0:1000] = np.where(image_list[0]!=0,image_list[0], result[0:800,0:1000])
warp_images(image_list[0], image_list[1], result_homography)
plt.imshow(warp_images(image_list[0], image_list[1], result_homography))



# image_list[0] = cylindricalWarp(image_list[0], K)[...,0]
# image_list[1] = cylindricalWarp(image_list[1], K)[...,0]

# height, width = image_list[0].shape[:2]
# concat_imgs = np.concatenate([image_list[1], image_list[0]], axis=1)

# _, ax = plt.subplots(dpi=200)

# plt.axis("off")
# plt.imshow(concat_imgs, cmap="gray")

# # Draw corners
# for _i in range(len(inlier_indices1)):
#     y_i, x_i = inlier_indices1[_i]
#     y_j, x_j = inlier_indices2[_i]
#     ax.add_patch(
#         mpl.patches.Circle(
#             (x_i, y_i),                   # (x, y)
#             5,
#             fill=True,
#     ))
#     ax.add_patch(
#         mpl.patches.Circle(
#             (x_j + width, y_j),                   # (x, y)
#             5,
#             fill=True,
#     ))
#     ax.add_patch(
#         mpl.patches.Arrow(
#             x_i, y_i,
#             x_j + width - x_i, y_j - y_i,
#             width=5,
#         )
#     )

# def warpImages(imgs, homographies):
#     result_image = []
#     for i in range(len(imgs) - 1):
#         homography = homographies[i]
#         for j in range(i + 1, len(imgs) - 1):
#             homography = np.dot(homographies[j], homography)
#         # transfomed_img = cv2.warpPerspective(imgs[i], homography, (imgs[i].shape[1], imgs[i].shape[0]))
#         # transfomed_img = cv2.warpPerspective(imgs[i], homographies[i], (imgs[i].shape[1], imgs[i].shape[0]))
#         result = cv2.warpPerspective(imgs[i+1], homographies[i],
#             (imgs[i].shape[1] + imgs[i+1].shape[1], imgs[i].shape[0]))
#         # result[0:imgs[i+1].shape[0], 0:imgs[i+1].shape[1]] = imgs[i]

#         # print(transfomed_img.shape, imgs[i].shape)
#         # plt.imshow(transfomed_img, cmap="gray")
#         # plt.imshow(np.concatenate((imgs[i], transfomed_img, imgs[i+1]), axis=1), cmap="gray")
#         plt.imshow(result, cmap="gray")
#         plt.show()
        
# warpImages([image_list[1], image_list[0]], [result_homography])