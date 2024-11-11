import os
import os.path as osp
import random
import numpy as np
import torch

from detect import detect
from tqdm import tqdm
import cv2
import json

class AverageMeter:
    """
    평균과 현재 값을 계산하고 저장하는 클래스
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = val * n
        self.count += n
        self.avg = self.sum / self.count

def get_image_path(data_dir, image_name, split='train'):
    """
    이미지의 실제 경로 찾기
    """
    lang_indicator = image_name.split('.')[1]
    if lang_indicator == 'zh':
        lang = 'chinese'
    elif lang_indicator == 'ja':
        lang = 'japanese'
    elif lang_indicator == 'th':
        lang = 'thai'
    elif lang_indicator == 'vi':
        lang = 'vietnamese'
    else:
        raise ValueError(f"Unknown language indicator: {lang_indicator}")
    
    # 실제 경로 생성
    image_path = osp.join(data_dir, f'{lang}_receipt/img/{split}/{image_name}')
    return image_path

def get_pred_bboxes(model, data_dir, valid_images, input_size, batch_size):
    image_fnames, by_sample_bboxes = [], []
    images = []

    for valid_image in tqdm(valid_images):
        # 파일명에서 언어 확인
        lang_indicator = valid_image.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        
        # 언어별 경로로 수정
        image_fpath = osp.join(data_dir, f'{lang}_receipt/img/train/{valid_image}')
        
        if not osp.exists(image_fpath):
            print(f"Warning: Image not found at {image_fpath}")
            continue
            
        image = cv2.imread(image_fpath)
        if image is None:
            print(f"Warning: Failed to load image: {image_fpath}")
            continue
            
        image_fnames.append(valid_image)
        images.append(image[:, :, ::-1])
        
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    pred_bboxes = dict()
    for idx in range(len(image_fnames)):
        image_fname = image_fnames[idx]
        sample_bboxes = by_sample_bboxes[idx]
        pred_bboxes[image_fname] = sample_bboxes

    return pred_bboxes

def get_gt_bboxes(data_dir, valid_images):
    gt_bboxes = dict()
    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
    
    # 각 언어별 UFO 파일 로드
    ufo_files = {}
    for lang in lang_list:
        ufo_path = osp.join(data_dir, f'{lang}_receipt/ufo/train.json')
        with open(ufo_path, 'r', encoding='utf-8') as f:
            ufo_files[lang] = json.load(f)['images']

    for valid_image in tqdm(valid_images):
        # 파일명에서 언어 확인
        lang_indicator = valid_image.split('.')[1]
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        
        # 해당 언어의 UFO 파일에서 bbox 정보 가져오기
        if valid_image not in ufo_files[lang]:
            print(f"Warning: {valid_image} not found in UFO file")
            continue
            
        gt_bboxes[valid_image] = []
        for idx in ufo_files[lang][valid_image]['words'].keys():
            gt_bboxes[valid_image].append(ufo_files[lang][valid_image]['words'][idx]['points'])

    return gt_bboxes