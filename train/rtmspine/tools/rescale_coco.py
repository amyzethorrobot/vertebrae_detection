import os
import json
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
import argparse

def filter_and_rescale_coco_dataset(
    images_folder: str,
    coco_file: str,
    max_size: int,
    output_folder: str,
    output_coco_file: str,
    types_to_remove: Optional[List[str]] = None
):
    """
    Filter out images by types and rescale remaining images and annotations.
    Save updated dataset and rescaled images.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])

    initial_image_count = len(images)
    initial_annotation_count = len(annotations)

    # Step 1: filter by types if requested
    image_ids_to_remove = set()
    if types_to_remove:
        types_to_remove_set = set(types_to_remove)
        image_ids_to_remove = {img['id'] for img in images if img.get('type') in types_to_remove_set}
        images = [img for img in images if img['id'] not in image_ids_to_remove]
        annotations = [ann for ann in annotations if ann['image_id'] not in image_ids_to_remove]

    print(f"Filtering by types={types_to_remove} completed.")
    print(f" - Removed images: {len(image_ids_to_remove)}")
    print(f" - Remaining images: {len(images)}")
    print(f" - Remaining annotations: {len(annotations)}")

    # Build annotation mapping
    image_id_to_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)

    # Step 2: rescale images and update annotations
    changed_images = 0

    for img_info in tqdm(images, desc="Rescaling images"):
        file_name = img_info['file_name']
        image_id = img_info['id']
        input_path = os.path.join(images_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        if not os.path.isfile(input_path):
            print(f"Warning: image {input_path} not found. Skipping.")
            continue

        try:
            with Image.open(input_path) as img:
                width, height = img.size

                if max(width, height) <= max_size:
                    img.save(output_path)
                    scale = 1.0
                else:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    img.save(output_path)

                    img_info['width'] = new_width
                    img_info['height'] = new_height
                    changed_images += 1

                # Update annotations
                if scale != 1.0 and image_id in image_id_to_annotations:
                    for ann in image_id_to_annotations[image_id]:
                        if 'bbox' in ann:
                            x, y, w, h = ann['bbox']
                            ann['bbox'] = [x * scale, y * scale, w * scale, h * scale]
                        if 'keypoints' in ann:
                            keypoints = ann['keypoints']
                            for i in range(0, len(keypoints), 3):
                                keypoints[i] *= scale
                                keypoints[i + 1] *= scale
                            ann['keypoints'] = keypoints

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Step 3: save updated dataset
    coco_data['images'] = images
    coco_data['annotations'] = annotations

    with open(output_coco_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)

    print("Processing complete.")
    print(f" - Total images before: {initial_image_count}")
    print(f" - Total annotations before: {initial_annotation_count}")
    print(f" - Images removed by types: {len(image_ids_to_remove)}")
    print(f" - Images rescaled: {changed_images}")
    print(f" - Final image count: {len(images)}")
    print(f" - Final annotation count: {len(annotations)}")

def main():
    parser = argparse.ArgumentParser(description="Filter COCO dataset by type(s) and rescale large images.")
    parser.add_argument("--images_folder", type=str, required=True, help="Path to original images folder.")
    parser.add_argument("--coco_file", type=str, required=True, help="Path to original COCO annotation file (json).")
    parser.add_argument("--max_size", type=int, required=True, help="Maximum size for the longest image side.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder for resized images.")
    parser.add_argument("--output_coco_file", type=str, required=True, help="Path to save updated COCO annotation file.")
    parser.add_argument("--types_to_remove", nargs="*", type=str, default=None,
                        help="List of image types to remove (e.g. cervical thoracic). Optional.")

    args = parser.parse_args()

    filter_and_rescale_coco_dataset(
        images_folder=args.images_folder,
        coco_file=args.coco_file,
        max_size=args.max_size,
        output_folder=args.output_folder,
        output_coco_file=args.output_coco_file,
        types_to_remove=args.types_to_remove
    )

if __name__ == "__main__":
    main()
