import os
import json
import shutil
import cv2
import numpy as np

def get_language_code(folder_name):
    """폴더명에서 언어 코드 추출"""
    lang_map = {
        'chinese': 'zh',
        'japanese': 'ja',
        'thai': 'th',
        'vietnamese': 'vi'
    }
    
    for lang, code in lang_map.items():
        if lang in folder_name.lower():
            return code
    return None

def bbox_to_points(bbox):
    """COCO bbox를 UFO points 형식으로 변환"""
    x, y, w, h = bbox
    return [
        [x, y],           # top-left
        [x + w, y],       # top-right
        [x + w, y + h],   # bottom-right
        [x, y + h]        # bottom-left
    ]

def convert_coco_to_ufo(coco_data, base_folder, lang_code):
    """COCO 형식을 UFO 형식으로 변환"""
    ufo_format = {
        "images": {}
    }
    
    # 이미지 ID와 파일명 매핑 생성
    image_id_map = {img['id']: img for img in coco_data['images']}
    
    # 이미지별 어노테이션 그룹화
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 각 이미지에 대해 처리
    for image_id, image_info in image_id_map.items():
        file_name = image_info['file_name']
        
        # UFO 형식의 이미지 정보 생성
        ufo_image_info = {
            "words": {}
        }
        
        # 해당 이미지의 어노테이션 처리
        if image_id in image_annotations:
            for idx, ann in enumerate(image_annotations[image_id]):
                word_id = f"word_{idx}"
                points = bbox_to_points(ann['bbox'])
                
                ufo_image_info["words"][word_id] = {
                    "points": points,
                    "transcription": "",
                    "language": lang_code,
                    "illegibility": False,
                    "orientation": "Horizontal"
                }
        
        # 상대 경로로 저장
        ufo_format["images"][file_name] = ufo_image_info
    
    return ufo_format

def process_receipt_folders(base_dir):
    """모든 receipt 폴더 처리"""
    for folder in os.listdir(base_dir):
        if not folder.endswith('_receipt'):
            continue
            
        # 언어 코드 추출
        lang_code = get_language_code(folder)
        if lang_code is None:
            print(f"Warning: Cannot determine language for folder {folder}")
            continue
            
        folder_path = os.path.join(base_dir, folder)
        coco_path = os.path.join(folder_path, 'coco')
        
        # COCO 파일 확인
        coco_file = os.path.join(coco_path, 'train_relabel_no_div.json')
        if not os.path.exists(coco_file):
            print(f"{coco_file} 파일을 찾을 수 없습니다.")
            continue
        
        print(f"\n{folder} 처리 중... (언어: {lang_code})")
        
        # COCO 데이터 로드
        with open(coco_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        ufo_aug_path = os.path.join(folder_path, 'ufo')
        
        # COCO를 UFO로 변환
        ufo_data = convert_coco_to_ufo(coco_data, folder_path, lang_code)
        
        # UFO 데이터 저장
        output_file = os.path.join(ufo_aug_path, 'train.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ufo_data, f, ensure_ascii=False, indent=2)
        
        print(f"변환 완료: {folder}")
        print(f"- 이미지 수: {len(ufo_data['images'])}")
        print(f"- 저장 경로: {output_file}")

def main():
    base_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-24/data'
    process_receipt_folders(base_dir)

if __name__ == '__main__':
    main()