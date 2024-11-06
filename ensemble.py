import json
import numpy as np
from shapely.geometry import Polygon
from typing import List, Dict

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def flatten_points(points: List) -> List[float]:
    if isinstance(points[0], list):
        return [coord for point in points for coord in point]
    return points

def polygon_from_points(points: List) -> Polygon:
    flattened_points = flatten_points(points)
    return Polygon([(flattened_points[i], flattened_points[i+1]) for i in range(0, len(flattened_points), 2)])

def calculate_iou(poly1: Polygon, poly2: Polygon) -> float:
    if not poly1.is_valid or not poly2.is_valid:
        return 0
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0
    except:
        return 0

def non_max_suppression(boxes: List[Dict], scores: List[float], iou_threshold: float) -> List[int]:
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        iou_list = [calculate_iou(polygon_from_points(boxes[i]['points']), 
                                  polygon_from_points(boxes[j]['points'])) for j in indices[1:]]
        indices = indices[1:][np.array(iou_list) <= iou_threshold]
    return keep

def weighted_average_boxes(boxes: List[Dict], scores: List[float]) -> Dict:
    points = np.array([flatten_points(box['points']) for box in boxes])
    weights = np.array(scores)[:, np.newaxis]
    weighted_points = np.sum(points * weights, axis=0) / np.sum(weights)
    return {'points': weighted_points.tolist()}

def ensemble_results(result1: Dict, result2: Dict, iou_threshold: float = 0.5, score_threshold: float = 0.5) -> Dict:
    ensembled_result = {'images': {}}

    for image_name in set(result1['images'].keys()) | set(result2['images'].keys()):
        ensembled_result['images'][image_name] = {'words': {}}
        
        words1 = result1['images'].get(image_name, {}).get('words', {})
        words2 = result2['images'].get(image_name, {}).get('words', {})

        all_words = list(words1.values()) + list(words2.values())
        scores = [1] * len(words1) + [1] * len(words2)  # Assuming equal weights for both models

        keep_indices = non_max_suppression(all_words, scores, iou_threshold)

        for idx, keep_idx in enumerate(keep_indices):
            if scores[keep_idx] > score_threshold:
                ensembled_result['images'][image_name]['words'][str(idx)] = all_words[keep_idx]

    return ensembled_result

if __name__ == "__main__":
    result1 = load_json('/data/ephemeral/home/god/predictions/no_line.csv')
    result2 = load_json('/data/ephemeral/home/god/predictions/only_line.csv')

    ensembled_result = ensemble_results(result1, result2)

    with open('ensembled_output.csv', 'w') as f:
        json.dump(ensembled_result, f, indent=4)

print("앙상블 완료. 결과가 'ensembled_output.csv'에 저장되었습니다.")