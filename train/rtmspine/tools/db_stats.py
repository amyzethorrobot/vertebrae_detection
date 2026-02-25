import json
import numpy as np

def analyze_coco_images(coco_file_path):
    # Load COCO annotation file
    with open(coco_file_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Check if images exist
    images = coco_data.get('images', [])
    if not images:
        print("No images found in the file.")
        return

    # Extract sizes and calculate aspect ratios
    widths = np.array([img['width'] for img in images])
    heights = np.array([img['height'] for img in images])
    aspect_ratios = widths / heights

    # Compute averages
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    avg_aspect_ratio = np.mean(aspect_ratios)

    # Compute min and max values
    min_width = np.min(widths)
    min_height = np.min(heights)
    max_width = np.max(widths)
    max_height = np.max(heights)
    min_aspect_ratio = np.min(aspect_ratios)
    max_aspect_ratio = np.max(aspect_ratios)

    # Print results
    print(f"Average image size: {avg_width:.2f} x {avg_height:.2f} (width x height)")
    print(f"Average aspect ratio: {avg_aspect_ratio:.4f}")
    print(f"Minimum image size: {min_width} x {min_height} (width x height)")
    print(f"Maximum image size: {max_width} x {max_height} (width x height)")
    print(f"Minimum aspect ratio: {min_aspect_ratio:.4f}")
    print(f"Maximum aspect ratio: {max_aspect_ratio:.4f}")

# Example usage
if __name__ == "__main__":
    file_path = "data/COCO_RTMPose_All_Train.json"
    analyze_coco_images(file_path)