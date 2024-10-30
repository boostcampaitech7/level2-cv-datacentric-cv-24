import pickle
import os
import os.path as osp
import numpy as np
import json
import albumentations as A
from tqdm import tqdm
from dataset import SceneTextDataset
from east_dataset import EASTDataset
from sklearn.model_selection import train_test_split

# image_size_list는 일단 실험용, 내가 더 해보고 여러 이미지 사이즈로 만들어서 train 하게 수정할게
def create_east_pickle(data_dir, image_size_list=[2048], crop_size_list=[1024], val_ratio=0.2):
    """
    여기서 augmentation 넣고 싶은거 넣고 사용하면 돼!
    대신 이거 사용할거면 color_jitter랑 normalize 변경하는 경우에는
    scene_dataset에서 False 설정해서 하면 되는거야!
    """
    custom_augmentation_dict = {
        'ColorJitter': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        'GaussianBlur': A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        'HueSaturation': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        'Normalize': A.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738), p=1.0)
    }

    selected_aug = ['ColorJitter', 'GaussianBlur', 'HueSaturation', 'Normalize']

    custom_augmentation = []
    for s in selected_aug:
        custom_augmentation.append(custom_augmentation_dict[s])

    # 모든 image_size와 crop_size 조합에 대해 반복
    for image_size in image_size_list:
        for crop_size in crop_size_list:
            print(f"\nProcessing image_size={image_size}, crop_size={crop_size}")
            
            # Train용 SceneTextDataset 생성
            """
            만약 dataset에 augmenatation 넣기 싫으면 custom_augmentation 빼고
            color_jitter, normalize도 지우면 알아서 True로 들어갑니다!
            참고 부탁해용
            """
            scene_dataset = SceneTextDataset(
                data_dir,
                split='train',
                image_size=image_size,
                crop_size=crop_size,
                custom_augmentation=A.Compose(custom_augmentation),
                color_jitter=False,
                normalize=False
            )
            
            # 언어 레이블 생성
            language_labels = []
            for fname in scene_dataset.image_fnames:
                lang_indicator = fname.split('.')[1]
                if lang_indicator == 'zh':
                    language_labels.append('chinese')
                elif lang_indicator == 'ja':
                    language_labels.append('japanese')
                elif lang_indicator == 'th':
                    language_labels.append('thai')
                elif lang_indicator == 'vi':
                    language_labels.append('vietnamese')

            indices = np.arange(len(scene_dataset))
            
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
            train_east_dataset = EASTDataset(scene_dataset)
            for idx in tqdm(train_indices, desc="Training data"):
                data = train_east_dataset[idx]
                train_data.append(data)
                train_fnames.append(scene_dataset.image_fnames[idx])
            
            # Validation용 EAST 데이터셋 생성
            print(f"\nProcessing validation data for size {image_size}x{crop_size}...")
            scene_dataset.color_jitter = False
            val_east_dataset = EASTDataset(scene_dataset)
            for idx in tqdm(val_indices, desc="Validation data"):
                data = val_east_dataset[idx]
                val_data.append(data)
                val_fnames.append(scene_dataset.image_fnames[idx])

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
        
if __name__ == '__main__':
    data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-24/data'
    create_east_pickle(data_dir)