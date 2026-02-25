import numpy as np
import json

def correct_last_two_coco_points(coco_json_path: str, output_json_path: str):
    """
    For COCO dataset annotations with keypoints length 94, correct points 92 and 93 if needed.
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    annotations = coco_data.get('annotations', [])
    corrected_count = 0

    for ann in annotations:
        keypoints = ann.get('keypoints', [])
        if not keypoints or len(keypoints) != 282:
            continue  # skip if not 94 points

        # Convert to numpy array shape (94, 3)
        kpts = np.array(keypoints).reshape(-1, 3)

        # Apply the same logic
        if kpts[92, 2] > 0 and kpts[93, 2] > 0:
            even_indices = np.arange(0, 92, 2)
            odd_indices = np.arange(1, 92, 2)

            visible_even = even_indices[kpts[even_indices, 2] > 0]
            visible_odd = odd_indices[kpts[odd_indices, 2] > 0]

            if len(visible_even) > 0 and len(visible_odd) > 0:
                mean_x_even = np.mean(kpts[visible_even, 0])
                mean_x_odd = np.mean(kpts[visible_odd, 0])

                x92 = kpts[92, 0]
                x93 = kpts[93, 0]

                if (mean_x_even >= mean_x_odd) != (x92 > x93):
                    # Swap points 92 and 93
                    kpts[[92, 93]] = kpts[[93, 92]]
                    corrected_count += 1

        # Save back to annotation
        ann['keypoints'] = kpts.flatten().tolist()

    # Save corrected json
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)

    print(f"Correction complete. Total annotations processed: {len(annotations)}. Points swapped: {corrected_count}.")


if __name__ == "__main__":
    correct_last_two_coco_points(
        coco_json_path="../data/saggital_lumbar/annotations/train.json",
        output_json_path="../data/saggital_lumbar/annotations/train_corrected.json"
    )