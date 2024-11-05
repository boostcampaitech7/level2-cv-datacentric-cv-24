import os
import os.path as osp
import time
import math
import wandb
from datetime import timedelta
from argparse import ArgumentParser

import torch
import json
import numpy as np
import random
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as A

from east_dataset import EASTDataset
from dataset import SceneTextDataset, PickleDataset
from model import EAST
from utils import AverageMeter, get_gt_bboxes, get_pred_bboxes
from deteval import calc_deteval_metrics


def get_train_transforms(config):
    return A.Compose([
        A.CLAHE(
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit,
            contrast_limit=config.constrast_limit,
            p=0.5
        ),
        A.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738), p=1.0)
    ])
    
def define_sweep_config():
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val F1 score',
            'goal': 'maximize'
        },
        'parameters': {
            'clahe_clip_limit': {'min': 2.0, 'max': 4.0},
            'clahe_grid_size': {'min': 4, 'max': 16},
            'brightness_limit': {'min': 0.1, 'max': 0.5},
            'constrast_limit': {'min': 0.1, 'max': 0.5}
        }
    }
    return sweep_config

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='Data-Centric_train')
    parser.add_argument('--wandb_entity', type=str, default='wj3714-naver-ai-boostcamp')
    parser.add_argument('--wandb_name', type=str, default=None, help='여기에 각자 프로젝트의 이름을 지정하세요.')
    parser.add_argument('--use_pickle', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def get_language_from_filename(filename):
    """파일 이름에서 언어 추출"""
    lang_indicator = filename.split('.')[1]
    lang_map = {
        'zh': 'chinese',
        'ja': 'japanese',
        'th': 'thai',
        'vi': 'vietnamese'
    }
    return lang_map.get(lang_indicator)

def create_stratified_split(data_dir, train_ratio=0.8):
    """언어별로 균형잡힌 train/validation 분할 생성"""
    lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
    
    # 언어별로 이미지 분류
    lang_dict = {lang: [] for lang in lang_list}
    
    # 각 언어별 데이터 로드
    for nation in lang_list:
        json_path = osp.join(data_dir, f'{nation}_receipt/ufo/train.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)
        for im in anno['images'].keys():
            lang_dict[nation].append(im)
    
    # 각 언어별로 train/val 분할
    train_images = []
    val_images = []
    
    for lang, images in lang_dict.items():
        n_train = int(len(images) * train_ratio)
        # 랜덤 셔플
        np.random.shuffle(images)
        train_images.extend(images[:n_train])
        val_images.extend(images[n_train:])
    
    return train_images, val_images

def setup_data_loader(data_dir, use_pickle, batch_size, num_workers, image_size, input_size, train_transform):
    """데이터 로더 설정 함수"""
    if use_pickle:
        train_dataset = PickleDataset(
            osp.join(data_dir, 'pickles/train.pickle'),
            color_jitter=True,
            normalize=True,
            map_scale=0.5,
            custom_augmentation=train_transform
        )
        val_dataset = PickleDataset(
            osp.join(data_dir, 'pickles/val.pickle'),
            color_jitter=False,
            normalize=True,
            map_scale=0.5
        )
    else:
        train_images, val_images = create_stratified_split(data_dir)
        train_scene_dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=image_size,
            crop_size=input_size,
            image_list=train_images,
            custom_augmentation=train_transform
        )
        train_dataset = EASTDataset(train_scene_dataset)

        val_scene_dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=image_size,
            crop_size=input_size,
            image_list=val_images,
            color_jitter=False
        )
        val_dataset = EASTDataset(val_scene_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, train_dataset, val_dataset

def train_one_epoch(model, train_loader, val_loader, optimizer, scheduler, train_num_batches, val_num_batches, 
                    val_interval, save_interval, model_dir, use_wandb, epoch, val_loss, best_val_loss, 
                    early_stopping_counter, patience, switch):
    epoch_loss, epoch_start = 0, time.time()
    
    # Training
    model.train()
    with tqdm(total=train_num_batches) as pbar:
        for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
            pbar.set_description('[Epoch {}]'.format(epoch + 1))
            
            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            pbar.update(1)
            val_dict = {
                'Cls loss': extra_info['cls_loss'],
                'Angle loss': extra_info['angle_loss'],
                'IoU loss': extra_info['iou_loss']
            }
            pbar.set_postfix(val_dict)
            
            if use_wandb:
                wandb.log({
                    "batch_loss": loss_val,
                    "cls_loss": extra_info['cls_loss'],
                    "angle_loss": extra_info['angle_loss'],
                    "iou_loss": extra_info['iou_loss']
                })

    if (epoch + 1) % save_interval == 0:
        if switch == False:
            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            if use_wandb:
                wandb.save(ckpt_fpath)
    
    scheduler.step()
    epoch_duration = time.time() - epoch_start
    
    if use_wandb:
        wandb.log({
            "epoch": epoch + 1,
            "mean_loss": epoch_loss / train_num_batches,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    print('Mean loss: {:.4f} | Elapsed time: {}'.format(
        epoch_loss / train_num_batches, timedelta(seconds=epoch_duration)))
    
    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss.reset()
        with torch.no_grad():
            with tqdm(total=val_num_batches) as pbar:
                for i, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(val_loader):
                    if i >= val_num_batches:
                        break
                    
                    pbar.set_description('Evaluate')
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    val_loss.update(loss.item())
                    
                    pbar.update(1)
                    val_dict = {
                        'val Total loss': val_loss.avg,
                        'Val Cls loss': extra_info['cls_loss'],
                        'Val Angle loss': extra_info['angle_loss'],
                        'Val IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(val_dict)
                    
                    if use_wandb:
                        wandb.log(val_dict)
        
        print(f'Validation Loss: {val_loss.avg:.4f}')
        
        if val_loss.avg < best_val_loss and switch == False:
            best_val_loss = val_loss.avg
            print(f"New best model for val loss : {val_loss.avg:.4f}! saving the best model..")
            torch.save(model.state_dict(), osp.join(model_dir, 'best.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
    return best_val_loss, early_stopping_counter, epoch_loss

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, use_wandb, wandb_project, wandb_entity,
                wandb_name, use_pickle, resume):
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "image_size": image_size,
                "input_size": input_size,
                "max_epoch": max_epoch
            }
        )
        
        config = wandb.config
        config = wandb.config
        clahe_clip_limit = config.clahe_clip_limit
        clahe_grid_size = config.clahe_grid_size
        brightness_limit = config.brightness_limit
        constrast_limit = config.constrast_limit
        
        # 추가적인 config 업데이트
        wandb.config.update({
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_grid_size": clahe_grid_size,
            "brightness_limit": brightness_limit,
            "constrast_limit": constrast_limit
        }, allow_val_change=True)
        
        train_transform = get_train_transforms(config)
    else:
        train_transform = None

    # 폴더가 없으면 만들기
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    train_loader, val_loader, train_dataset, val_dataset =setup_data_loader(
        data_dir=data_dir,
        use_pickle=use_pickle,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        input_size=input_size,
        train_transform=train_transform
    )

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    if resume:
        checkpoint = torch.load(osp.join('/data/ephemeral/home/level2-cv-datacentric-cv-24/trained_models', 'best_f1.pth'))
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    val_interval = 5
    best_val_loss = float('inf')
    val_loss = AverageMeter()

    patience = 30
    early_stopping_counter = 0    

    for epoch in range(max_epoch):
        best_val_loss, early_stopping_counter, epoch_loss = train_one_epoch(
            model, train_loader, val_loader, optimizer, scheduler,
            train_num_batches, val_num_batches, val_interval, save_interval, model_dir,
            use_wandb, epoch, val_loss, best_val_loss, early_stopping_counter, patience,
            switch=False
        )
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if use_wandb:
        wandb.finish()

    # 추가 학습
    print("\nStarting final training with swapped datasets...")

    val_loader, train_loader, val_dataset, train_dataset = setup_data_loader(
        data_dir=data_dir,
        use_pickle=use_pickle,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        input_size=input_size
    )

    best_val_loss = float('inf')
    early_stopping_counter = 0
    model.load_state_dict(torch.load(osp.join(model_dir, 'latest.pth')))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1)

    train_dataset, val_dataset = val_dataset, train_dataset

    for epoch in range(20):
        best_val_loss, early_stopping_counter, epoch_loss = train_one_epoch(
            model, train_loader, val_loader, optimizer, scheduler,
            train_num_batches, val_num_batches, val_interval, save_interval, model_dir,
            use_wandb, epoch, val_loss, best_val_loss, early_stopping_counter, patience,
            switch=True
        )
    
        if early_stopping_counter >= patience:
            break

    if use_wandb:
        wandb.finish()

    torch.save(model.state_dict(), osp.join(model_dir, 'final.pth'))

def main(args):
    # do_training(**args.__dict__)
    if args.use_wandb:
        sweep_config = define_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        def train_wrapper():
            do_training(**args.__dict__)
        wandb.agent(sweep_id, function=train_wrapper, count=1)  # count: n번의 실험 실행
    else:
        do_training(**args.__dict__)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)