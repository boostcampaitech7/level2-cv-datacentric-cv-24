import pickle
import sys
import os
import os.path as osp
import numpy as np
import json
import albumentations as A
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import SceneTextDataset
from east_dataset import EASTDataset

def create_east_pickle(data_dir, image_size_list=[2048], crop_size_list=[1024], val_ratio=0.1):
    # 모든 image_size와 crop_size 조합에 대해 반복
    for image_size in image_size_list:
        for crop_size in crop_size_list:
            print(f"\nProcessing image_size={image_size}, crop_size={crop_size}")
            
            # Train용 SceneTextDataset 생성
            train_dataset = SceneTextDataset(
                data_dir,
                split='train_2',
                image_size=image_size,
                crop_size=crop_size,
                custom_augmentation=None,
            )

            val_dataset = SceneTextDataset(
                data_dir,
                split='train_2',
                image_size=image_size,
                crop_size=crop_size,
                custom_augmentation=None,
                color_jitter=False,
            )
            
            # 언어 레이블 생성
            language_labels = []
            for fname in train_dataset.image_fnames:
                lang_indicator = fname.split('.')[1]
                if lang_indicator == 'zh':
                    language_labels.append('chinese')
                elif lang_indicator == 'ja':
                    language_labels.append('japanese')
                elif lang_indicator == 'th':
                    language_labels.append('thai')
                elif lang_indicator == 'vi':
                    language_labels.append('vietnamese')

            indices = np.arange(len(train_dataset))
            
            # Stratified split
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                stratify=language_labels,
                random_state=42
            )

            # Train/Val 데이터셋 생성
            train_data = []
            val_data = []
            train_fnames = []
            val_fnames = []
            
            # Train용 EAST 데이터셋 생성
            print(f"\nProcessing training data for size {image_size}x{crop_size}...")
            train_east_dataset = EASTDataset(train_dataset)
            for idx in tqdm(train_indices, desc="Training data"):
                data = train_east_dataset[idx]
                train_data.append(data)
                train_fnames.append(train_dataset.image_fnames[idx])
            
            # Validation용 EAST 데이터셋 생성
            print(f"\nProcessing validation data for size {image_size}x{crop_size}...")
            val_east_dataset = EASTDataset(val_dataset)
            for idx in tqdm(val_indices, desc="Validation data"):
                data = val_east_dataset[idx]
                val_data.append(data)
                val_fnames.append(val_dataset.image_fnames[idx])

            # pickle 파일 저장
            train_save = {
                'data': train_data,
                'image_fnames': train_fnames
            }
            val_save = {
                'data': val_data,
                'image_fnames': val_fnames
            }

            # 파일명에 크기 정보 포함
            train_path = osp.join(data_dir, f'east_dataset_train_{image_size}_{crop_size}.pkl')
            val_path = osp.join(data_dir, f'east_dataset_val_{image_size}_{crop_size}.pkl')

            print(f"\nSaving pickle files for size {image_size}x{crop_size}...")
            with open(train_path, 'wb') as f:
                pickle.dump(train_save, f)
            with open(val_path, 'wb') as f:
                pickle.dump(val_save, f)
            
            print(f"Train dataset saved to {train_path}")
            print(f"Validation dataset saved to {val_path}")

def verify_pickle_data(data_dir, image_size=2048, crop_size=1024):
    train_path = osp.join(data_dir, f'east_dataset_train_{image_size}_{crop_size}.pkl')
    val_path = osp.join(data_dir, f'east_dataset_val_{image_size}_{crop_size}.pkl')
    
    # 파일 존재 여부 확인
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("pickle 파일이 존재하지 않습니다.")
        return False
    
    try:
        # 데이터 로드
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
            
        # 데이터 구조 확인
        print("\n데이터 구조 검증:")
        print(f"Train 데이터 키: {train_data.keys()}")
        print(f"Train 데이터 길이: {len(train_data['data'])}")
        print(f"Train 파일명 길이: {len(train_data['image_fnames'])}")
        print(f"Val 데이터 길이: {len(val_data['data'])}")
        print(f"Val 파일명 길이: {len(val_data['image_fnames'])}")
        
        # 데이터 샘플 확인
        print("\n데이터 샘플 확인:")
        print(f"Train 데이터 첫 번째 항목 키: {train_data['data'][0].keys()}")
        print(f"Train 첫 번째 이미지 파일명: {train_data['image_fnames'][0]}")
        
        return True
        
    except Exception as e:
        print(f"데이터 검증 중 오류 발생: {e}")
        return False
        
if __name__ == '__main__':
    data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-24/data'
    create_east_pickle(data_dir)

    print("\n저장된 데이터 검증...")
    if verify_pickle_data(data_dir):
        print("데이터 검증 완료: 정상")
    else:
        print("데이터 검증 실패: 오류")