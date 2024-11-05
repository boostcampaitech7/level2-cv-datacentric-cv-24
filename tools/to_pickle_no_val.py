import os
import os.path as osp
import sys
sys.path.append('/data/ephemeral/home/code')
import json
import pickle
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import filter_vertices, resize_img, adjust_height, rotate_img, crop_img, generate_roi_mask
from east_dataset import generate_score_geo_maps  # EAST 관련 함수 import

def create_stratified_split(total_anno, train_ratio=0.8):
    """언어별로 균형잡힌 train/validation 분할 생성"""
    # 언어별로 이미지 분류
    lang_dict = {
        'zh': [], 'ja': [], 'th': [], 'vi': []
    }
    
    # 각 이미지를 언어별로 분류
    for image_fname in total_anno['images'].keys():
        lang_indicator = image_fname.split('.')[1]
        if lang_indicator in lang_dict:
            lang_dict[lang_indicator].append(image_fname)
    
    # 각 언어별로 train/val 분할
    train_images, val_images = [], []
    
    for lang, images in lang_dict.items():
        np.random.shuffle(images)
        n_train = int(len(images) * train_ratio)
        train_images.extend(images[:n_train])
        val_images.extend(images[n_train:])
    
    return train_images, val_images

def preprocessing(
    root_dir,
    split="train",
    num=0,
    image_size=2048, 
    crop_size=1024,
    ignore_under_threshold=10,
    drop_under_threshold=1,
    map_scale=0.5,
    train_ratio=0.8
):
    if crop_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
    
    os.makedirs(osp.join(root_dir, 'pickles'), exist_ok=True)

    total_anno = dict(images=dict())
    
    # json 파일을 읽는 부분
    json_name = 'train_2.json'  # 항상 train.json을 읽습니다

    for nation in lang_list:
        json_path = osp.join(root_dir, f'{nation}_receipt/ufo/{json_name}')
        with open(json_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)
        for im in anno['images']:
            # valid_words = {}
            # for word_id, word_info in anno['images'][im]['words'].items():
            #     if word_info.get('transcription') not in [None, ""]:
            #         valid_words[word_id] = word_info
            
            # if valid_words:
            #     total_anno['images'][im] = anno['images'][im]
            #     total_anno['images'][im]['words'] = valid_words
            total_anno['images'][im] = anno['images'][im]

    # train/validation 분할
    train_images, val_images = create_stratified_split(total_anno, train_ratio)
    
    # train과 validation 각각에 대해 처리
    for current_split, image_list in [("train", train_images), ("val", val_images)]:
        if num == 0:
            pkl_dir = osp.join(root_dir, f"pickles/{current_split}.pickle")
        else:
            pkl_dir = osp.join(root_dir, f"pickles/{current_split}{num}.pickle")

        total = dict(
            images=[],
            vertices=[],
            labels=[],
            word_bboxes=[],
            roi_masks=[],
            score_maps=[],
            geo_maps=[]
        )

        print(f"\nProcessing {current_split} data...")
        for idx in tqdm(range(len(image_list))):
            image_fname = image_list[idx]
            
            lang_indicator = image_fname.split('.')[1]
            lang_map = {
                'zh': 'chinese',
                'ja': 'japanese',
                'th': 'thai',
                'vi': 'vietnamese'
            }
            lang = lang_map.get(lang_indicator)
            if not lang:
                continue
                
            image_fpath = osp.join(root_dir, f'{lang}_receipt/img/train', image_fname)

            vertices, labels = [], []
            for word_info in total_anno['images'][image_fname]['words'].values():
                num_pts = np.array(word_info['points']).shape[0]
                if num_pts > 4:
                    continue
                vertices.append(np.array(word_info['points']).flatten())
                labels.append(1)
                
            vertices = np.array(vertices, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            vertices, labels = filter_vertices(
                vertices,
                labels,
                ignore_under=ignore_under_threshold,
                drop_under=drop_under_threshold
            )

            try:
                # 이미지 전처리
                image = Image.open(image_fpath)
                image, vertices = resize_img(image, vertices, image_size)
                image, vertices = adjust_height(image, vertices)
                image, vertices = rotate_img(image, vertices)
                image, vertices = crop_img(image, vertices, labels, crop_size)

                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)

                word_bboxes = np.reshape(vertices, (-1, 4, 2))
                roi_mask = generate_roi_mask(image, vertices, labels)

                score_map, geo_map = generate_score_geo_maps(
                    image, 
                    word_bboxes,
                    map_scale=map_scale
                )

                total["images"].append(image)
                total["vertices"].append(vertices)
                total["labels"].append(labels)
                total["word_bboxes"].append(word_bboxes)
                total["roi_masks"].append(roi_mask)
                total["score_maps"].append(score_map)
                total["geo_maps"].append(geo_map)
                
            except Exception as e:
                print(f"Error processing image {image_fpath}: {str(e)}")
                continue

        print(f"Save path >> {pkl_dir}")
        print(f"Total {current_split} samples: {len(total['images'])}")
        with open(pkl_dir, 'wb') as fw:
            pickle.dump(total, fw)

if __name__ == "__main__":
    preprocessing(
        root_dir='/data/ephemeral/home/level2-cv-datacentric-cv-24/data',
        split="train",  # split 파라미터는 이제 무시됩니다
        num=0,
        image_size=2048,
        crop_size=1024,
        ignore_under_threshold=10,
        drop_under_threshold=1,
        map_scale=0.5,
        train_ratio=0.8  # train:val = 8:2 비율
    )