import os
import json
import cv2
import numpy as np

def points_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min
    return [x_min, y_min, width, height]

def points_to_coco_segmentation(points):
    """points 좌표를 COCO segmentation 형식으로 변환"""
    # COCO segmentation 형식: [[x1,y1,x2,y2,x3,y3,x4,y4]]
    flattened = []
    for point in points:
        flattened.extend([float(point[0]), float(point[1])])
    return [flattened]  # 리스트로 한번 더 감싸야 함

def convert_ufo_to_coco(ufo_data, base_folder):
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'text'}]
    }
    
    image_id = 1
    annotation_id = 1

    for image_name, image_info in ufo_data['images'].items():
        # 이미지 경로 수정
        # UFO의 이미지 경로는 'images/train/xxx.jpg' 형식
        # base_folder는 'data/chinese_receipt' 같은 형식
        img_folder = os.path.join(base_folder, 'img')
        full_image_path = os.path.join(img_folder, 'train', os.path.basename(image_name))
            
        if not os.path.exists(full_image_path):
            print(f"Warning: Image {full_image_path} not found")
            continue
            
        # 이미지 정보 추가
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Warning: Cannot read image {full_image_path}")
            continue
            
        height, width = image.shape[:2]
        
        coco_format['images'].append({
            'id': image_id,
            'file_name': image_name,
            'width': width,
            'height': height
        })
        
        # 어노테이션 정보 추가
        for word_id, word_info in image_info['words'].items():
            bbox = points_to_bbox(word_info['points'])
            segmentation = points_to_coco_segmentation(word_info['points'])
            
            coco_format['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox,
                'segmentation': segmentation,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            annotation_id += 1
        image_id += 1
    
    return coco_format

def process_receipt_folders(base_dir):
    """모든 receipt 폴더 처리"""
    for folder in os.listdir(base_dir):
        if not folder.endswith('_receipt'):
            continue
            
        folder_path = os.path.join(base_dir, folder)
        ufo_path = os.path.join(folder_path, 'ufo')
        
        # train.json 파일 처리
        train_json = os.path.join(ufo_path, 'train.json')
        if os.path.exists(train_json):
            print(f"\n{folder} 처리 중...")
            
            with open(train_json, 'r', encoding='utf-8') as f:
                ufo_data = json.load(f)
            
            # COCO 폴더 생성
            coco_path = os.path.join(folder_path, 'coco')
            os.makedirs(coco_path, exist_ok=True)
            
            # UFO를 COCO로 변환
            coco_data = convert_ufo_to_coco(ufo_data, folder_path)
            
            # 변환된 데이터 저장
            output_file = os.path.join(coco_path, 'train.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)
            
            print(f"변환 완료: {folder}")
            print(f"- 이미지 수: {len(coco_data['images'])}")
            print(f"- 어노테이션 수: {len(coco_data['annotations'])}")
            print(f"- 저장 경로: {output_file}")

def main():
    base_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-24/data'
    process_receipt_folders(base_dir)

if __name__ == '__main__':
    main()