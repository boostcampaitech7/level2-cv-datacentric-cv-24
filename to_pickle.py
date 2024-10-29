import pickle
import os
import os.path as osp
from tqdm import tqdm
from dataset import SceneTextDataset
from east_dataset import EASTDataset

def create_east_pickle(data_dir, image_size=2048, crop_size=1024):
    # SceneTextDataset 생성
    scene_dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=crop_size
    )
    
    # EASTDataset으로 변환
    east_dataset = EASTDataset(scene_dataset)

    # pickle 파일 저장 경로 설정
    save_path = osp.join(data_dir, 'east_dataset.pkl')
    
    # 전체 데이터를 미리 계산하여 저장
    preprocessed_data = []
    total_items = len(east_dataset)
    
    print("Creating EAST dataset pickle...")
    for i in tqdm(range(total_items), desc="Processing", unit="items"):
        data = east_dataset[i]  # (img, score_map, geo_map, roi_mask)
        preprocessed_data.append(data)
        
        # 10% 단위로 진행상황 출력
        if (i + 1) % (total_items // 10) == 0:
            print(f"\nProgress: {((i + 1) / total_items) * 100:.1f}% completed")
    
    print("\nSaving pickle file...")
    # pickle로 저장
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"Dataset successfully saved to {save_path}")

if __name__ == '__main__':
    data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-24/data'
    create_east_pickle(data_dir)