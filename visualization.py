import os
import json
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='Output Visualiation')
    parser.add_argument('--save-dir',  help='save_dir path')
    parser.add_argument('--inference-dir', help='inference_dir path')
    args = parser.parse_args()
    return args

def read_json(filename: str):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

nation_dict = {
    'vi': 'vietnamese_receipt',
    'th': 'thai_receipt',
    'zh': 'chinese_receipt',
    'ja': 'japanese_receipt',
}

def save_vis_to_img(save_dir: str | os.PathLike, inference_dir: str | os.PathLike = 'output.csv') -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)   
    data = read_json(inference_dir)
    for im, points in data['images'].items():
        # change to 'train' for train dataset 
        im_path = Path('data') / nation_dict[im.split('.')[1]] / 'img' / 'test' / im
        img = Image.open(im_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for obj_k, obj_v in points['words'].items():
            # bbox points
            pts = [(int(p[0]), int(p[1])) for p in obj_v['points']]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]

            draw.polygon(pts, outline=(255, 0, 0))                
            draw.text(
                (pt1[0]-3, pt1[1]-12),
                obj_k,
                fill=(0, 0, 0)
            )
        img.save(os.path.join(save_dir, im))

if __name__  == "__main__":
    args = parse_arg()

    save_vis_to_img(save_dir=args.save_dir, inference_dir=args.inference_dir)