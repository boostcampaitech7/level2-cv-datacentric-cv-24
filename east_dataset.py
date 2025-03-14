import math

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from numba import njit
import albumentations as A
@njit
def nb_meshgrid(x, y):
    # Get the number of rows and columns
    rows = len(y)
    cols = len(x)

    # Pre-allocate empty arrays
    X = np.empty((rows, cols), dtype=x.dtype)
    Y = np.empty((rows, cols), dtype=y.dtype)

    # Fill the X array: repeat x along rows
    for i in range(rows):
        for j in range(cols):
            X[i, j] = x[j]

    # Fill the Y array: repeat y along columns
    for i in range(rows):
        for j in range(cols):
            Y[i, j] = y[i]

    return X, Y

@njit
def nb_amin(arr, axis=0):
    # Check the shape of the input array
    rows, cols = arr.shape
    
    # If axis=0, find the minimum along columns (i.e., compare across rows)
    if axis == 0:
        # Initialize the result array to store the minimum values
        min_vals = np.empty(cols, dtype=arr.dtype)

        # Loop through each column
        for j in range(cols):
            # Set the first element of the column as the initial minimum
            min_val = arr[0, j]

            # Compare each element in the column with the current minimum
            for i in range(1, rows):
                if arr[i, j] < min_val:
                    min_val = arr[i, j]

            # Store the minimum value for this column
            min_vals[j] = min_val

        return min_vals

@njit
def nb_amax(arr, axis=0):
    # Check the shape of the input array
    rows, cols = arr.shape

    # If axis=0, find the maximum along columns (i.e., compare across rows)
    if axis == 0:
        # Initialize the result array to store the maximum values
        max_vals = np.empty(cols, dtype=arr.dtype)

        # Loop through each column
        for j in range(cols):
            # Set the first element of the column as the initial maximum
            max_val = arr[0, j]

            # Compare each element in the column with the current maximum
            for i in range(1, rows):
                if arr[i, j] > max_val:
                    max_val = arr[i, j]

            # Store the maximum value for this column
            max_vals[j] = max_val

        return max_vals
    
@njit
def nb_norm(arr,axis=0):
    rows, cols = arr.shape
    norms = np.zeros(cols, dtype=np.float64)
    for j in range(cols):
        col_vector = arr[:, j]
        norms[j] = np.linalg.norm(col_vector)
    return norms

@njit
def get_rotated_coords(h, w, theta, anchor):
    anchor = anchor.reshape(2, 1)
    rotate_mat = get_rotate_mat(theta)
    x, y = nb_meshgrid(np.arange(w), np.arange(h))
    # x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - anchor) + anchor
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y

@njit
def shrink_bbox(bbox, coef=0.3, inplace=False):
    lens = [np.linalg.norm(bbox[i] - bbox[(i + 1) % 4], ord=2) for i in range(4)]
    r = [min(lens[(i - 1) % 4], lens[i]) for i in range(4)]

    if not inplace:
        bbox = bbox.copy()

    offset = 0 if lens[0] + lens[2] > lens[1] + lens[3] else 1
    for idx in [0, 2, 1, 3]:
        p1_idx, p2_idx = (idx + offset) % 4, (idx + 1 + offset) % 4
        p1p2 = bbox[p2_idx] - bbox[p1_idx]
        dist = np.linalg.norm(p1p2)
        if dist <= 1:
            continue
        bbox[p1_idx] += p1p2 / dist * r[p1_idx] * coef
        bbox[p2_idx] -= p1p2 / dist * r[p2_idx] * coef
    return bbox

@njit
def get_rotated_coords_(h, w, theta, anchor):
    anchor = anchor.reshape(2, 1)
    rotate_mat = get_rotate_mat(theta)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - anchor) + anchor
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y

@njit
def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

@njit
def calc_error_from_rect(bbox):
    '''
    Calculate the difference between the vertices orientation and default orientation. Default
    orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    '''
    x_min, y_min = nb_amin(bbox, axis=0)
    x_max, y_max = nb_amax(bbox, axis=0)
    rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32)
    return nb_norm(bbox - rect, axis=0).sum()

@njit
def rotate_bbox(bbox, theta, anchor=None):
    points = bbox.T
    if anchor is None:
        anchor = points[:, :1]
    rotated_points = np.dot(get_rotate_mat(theta), points - anchor) + anchor
    return rotated_points.T


def find_min_rect_angle(bbox, rank_num=10):
    '''Find the best angle to rotate poly and obtain min rectangle
    '''
    areas = []
    angles = np.arange(-90, 90) / 180 * math.pi
    for theta in angles:
        rotated_bbox = rotate_bbox(bbox, theta)
        x_min, y_min = nb_amin(rotated_bbox, axis=0)
        x_max, y_max = nb_amax(rotated_bbox, axis=0)
        areas.append((x_max - x_min) * (y_max - y_min))

    best_angle, min_error = -1, float('inf')
    for idx in np.argsort(areas)[:rank_num]:
        rotated_bbox = rotate_bbox(bbox, angles[idx])
        error = calc_error_from_rect(rotated_bbox)
        if error < min_error:
            best_angle, min_error = angles[idx], error

    return best_angle


def generate_score_geo_maps(image, word_bboxes, map_scale=0.5):
    img_h, img_w = image.shape[:2]
    map_h, map_w = int(img_h * map_scale), int(img_w * map_scale)
    inv_scale = int(1 / map_scale)

    score_map = np.zeros((map_h, map_w, 1), np.float32)
    geo_map = np.zeros((map_h, map_w, 5), np.float32)

    word_polys = []

    for bbox in word_bboxes:
        poly = np.around(map_scale * shrink_bbox(bbox)).astype(np.int32)
        word_polys.append(poly)

        center_mask = np.zeros((map_h, map_w), np.float32)
        cv2.fillPoly(center_mask, [poly], 1)

        theta = find_min_rect_angle(bbox)
        rotated_bbox = rotate_bbox(bbox, theta) * map_scale
        x_min, y_min = nb_amin(rotated_bbox, axis=0)
        x_max, y_max = nb_amax(rotated_bbox, axis=0)

        anchor = bbox[0] * map_scale
        rotated_x, rotated_y = get_rotated_coords(map_h, map_w, theta, anchor)

        d1, d2 = rotated_y - y_min, y_max - rotated_y
        d1[d1 < 0] = 0
        d2[d2 < 0] = 0
        d3, d4 = rotated_x - x_min, x_max - rotated_x
        d3[d3 < 0] = 0
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1 * center_mask * inv_scale
        geo_map[:, :, 1] += d2 * center_mask * inv_scale
        geo_map[:, :, 2] += d3 * center_mask * inv_scale
        geo_map[:, :, 3] += d4 * center_mask * inv_scale
        geo_map[:, :, 4] += theta * center_mask

    cv2.fillPoly(score_map, word_polys, 1)

    return score_map, geo_map

def center_crop_horizontal(image, crop_width):
    height, width, _ = image.shape
    left = (width - crop_width) // 2
    right = left + crop_width
    return image[:, left:right] 

class EASTDataset(Dataset):
    def __init__(self, dataset, map_scale=0.5, to_tensor=True):
        self.dataset = dataset
        self.map_scale = map_scale
        self.to_tensor = to_tensor
        self.crop_width = 512

    def __getitem__(self, idx):
        image, word_bboxes, roi_mask = self.dataset[idx]

        image = center_crop_horizontal(image, crop_width=self.crop_width)

        score_map, geo_map = generate_score_geo_maps(image, word_bboxes, map_scale=self.map_scale)

        mask_size = int(image.shape[0] * self.map_scale), int(image.shape[1] * self.map_scale)
        roi_mask = cv2.resize(roi_mask, dsize=mask_size)
        if roi_mask.ndim == 2:
            roi_mask = np.expand_dims(roi_mask, axis=2)

        # 텐서 변환
        if self.to_tensor:
            image = torch.Tensor(image).permute(2, 0, 1)
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        return len(self.dataset)
