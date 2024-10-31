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

from east_dataset import EASTDataset
from dataset import SceneTextDataset, PickleEASTDataset
from model import EAST
from utils import AverageMeter, get_gt_bboxes, get_pred_bboxes
from deteval import calc_deteval_metrics


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


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, use_wandb, wandb_project, wandb_entity,
                wandb_name, use_pickle):
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

    # 폴더가 없으면 만들기
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    if use_pickle:
        pickle_train_path = osp.join(data_dir, 'east_dataset_train_2048_1024.pkl')
        pickle_val_path = osp.join(data_dir, 'east_dataset_val_2048_1024.pkl')

        if os.path.exists(pickle_train_path) and os.path.exists(pickle_val_path):
            print("Loading preprocessed datasets from pickle")
            train_dataset = PickleEASTDataset(pickle_train_path)
            val_dataset = PickleEASTDataset(pickle_val_path)
        else:
            raise FileNotFoundError("Pickle files not found. Please create pickle datasets first.")
    else:
        print("Creating startified split datasets...")
        train_images, val_images = create_stratified_split(data_dir)

        train_scene_dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=image_size,
            crop_size=input_size,
            image_list=train_images
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

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    val_interval = 5
    best_val_loss = float('inf')
    val_loss = AverageMeter()
    
    best_f1_score = 0

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
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
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
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

        scheduler.step()

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "mean_loss": epoch_loss / train_num_batches,
                "leraning_rate": optimizer.param_groups[0]['lr']
            })

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))
        
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
                            wandb.log(val_dict, step=epoch)
                
                print(f'Validation Loss: {val_loss.avg:.4f}')

                if val_loss.avg < best_val_loss:
                    best_val_loss = val_loss.avg
                    print(f"New best model for val loss : {val_loss.avg:.4f}! saving the best model..")
                    torch.save(model.state_dict(), osp.join(model_dir, 'best.pth'))

                print("Calculating validation metrics...")
                valid_images = []
                if use_pickle:
                    all_valid_images = val_dataset.image_fnames
                    valid_images = random.sample(all_valid_images, 10)
                else:
                    all_valid_images = val_dataset.scene_text_dataset.image_fnames
                    valid_images = random.sample(all_valid_images, 10)

                pred_bboxes_dict = get_pred_bboxes(model, data_dir, valid_images, input_size, batch_size)
                gt_bboxes_dict = get_gt_bboxes(data_dir, valid_images)

                random.seed(42) # 재현성을 위한 시드 추가

                if pred_bboxes_dict and gt_bboxes_dict:
                    result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
                    total_result = result['total']
                    precision, recall = total_result['precision'], total_result['recall']
                    f1_score = 2*precision*recall/(precision+recall) if precision + recall > 0 else 0

                    print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1 score: {f1_score:.4f}')

                    if use_wandb:
                        wandb.log({
                            'val Precision' : precision,
                            'val Recall': recall,
                            'val F1 score': f1_score
                        }, step=epoch)

                    if best_f1_score < f1_score:
                        print(f"New best model for f1 score : {f1_score}! saving the best model..")
                        best_fpth = osp.join(model_dir, 'best_f1.pth')
                        torch.save(model.state_dict(), best_fpth)
                        best_f1_score = f1_score

            model.train()

        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

            if use_wandb:
                wandb.save(ckpt_fpath)

    if use_wandb:
        wandb.finish()

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)