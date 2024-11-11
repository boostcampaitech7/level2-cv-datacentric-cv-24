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

def weighted_box_fusion(boxes: List[Dict], scores: List[float], iou_threshold: float) -> List[Dict]:
    sorted_indices = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in sorted_indices]
    scores = np.array([scores[i] for i in sorted_indices])
    
    final_boxes = []
    used = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if used[i]:
            continue
        fused_boxes = [boxes[i]]
        fused_scores = [scores[i]]
        used[i] = True

        for j in range(i+1, len(boxes)):
            if used[j]:
                continue
            iou = calculate_iou(polygon_from_points(boxes[i]['points']), polygon_from_points(boxes[j]['points']))
            if iou >= iou_threshold:
                fused_boxes.append(boxes[j])
                fused_scores.append(scores[j])
                used[j] = True
        
        fused_box = weighted_average_boxes(fused_boxes, fused_scores)
        final_boxes.append(fused_box)

    return final_boxes

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

        fused_words = weighted_box_fusion(all_words, scores, iou_threshold)

        for idx, fused_word in enumerate(fused_words):
            if scores[idx] > score_threshold:
                ensembled_result['images'][image_name]['words'][str(idx)] = fused_word

    return ensembled_result

if __name__ == "__main__":
    result1 = load_json('/data/ephemeral/home/god/predictions/no_line.csv')
    result2 = load_json('/data/ephemeral/home/god/predictions/only_line.csv')

    ensembled_result = ensemble_results(result1, result2)

    with open('ensembled_output.csv', 'w') as f:
        json.dump(ensembled_result, f, indent=4)

print("앙상블 완료. 결과가 'ensembled_output.csv'에 저장되었습니다.")
