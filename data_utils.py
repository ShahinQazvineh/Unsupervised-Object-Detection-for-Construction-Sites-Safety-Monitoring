import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from roboflow import Roboflow
from config import CONFIG
import glob

def parse_yolo_labels(label_path, image_width, image_height):
    """
    Parses a YOLO format label file.

    Args:
        label_path (str): The path to the label file.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        list: A list of dictionaries, where each dictionary represents a bounding box.
    """
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            boxes.append({'class_id': int(class_id), 'box': [x1, y1, x2, y2]})
    return boxes

def prepare_dataset(data_fraction=1.0, random_state=42):
    """
    Prepares the dataset by loading data and creating a stratified subset if required.

    Args:
        data_fraction (float): The fraction of the data to use.
        random_state (int): The random state for reproducibility.

    Returns:
        tuple: A tuple containing the data (image paths) and labels (bounding boxes).
    """
    if CONFIG['data_source'] == 'roboflow':
        rf = Roboflow(api_key=CONFIG['roboflow']['api_key'])
        project = rf.workspace(CONFIG['roboflow']['workspace']).project(CONFIG['roboflow']['project'])
        dataset = project.version(CONFIG['roboflow']['version']).download("yolov5")
        data_dir = dataset.location
    else:
        data_dir = CONFIG['data_dir_abs']

    all_image_paths = []
    all_label_paths = {}

    for split in ['train', 'valid', 'test']:
        image_dir = os.path.join(data_dir, split, 'images')
        label_dir = os.path.join(data_dir, split, 'labels')

        if not os.path.isdir(image_dir):
            continue

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
            # Key the label paths by their filename without extension
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            all_label_paths[base_name] = label_file

    image_paths = []
    labels = []
    for img_path in all_image_paths:
        # Find the corresponding label file
        img_base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Roboflow sometimes adds suffixes like .rf.HASH
        # We need to strip this to find the matching label
        label_base_name = img_base_name.split('.rf.')[0]

        if label_base_name in all_label_paths:
            label_path = all_label_paths[label_base_name]
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
            boxes = parse_yolo_labels(label_path, width, height)
            labels.append(boxes)
            image_paths.append(img_path)
        else:
            print(f"Warning: Label file not found for image: {img_path}")


    if data_fraction < 1.0:
        print(f"Creating a stratified subset of the data with fraction: {data_fraction}")
        # Stratified splitting with bounding boxes is more complex.
        # For now, we'll just do a random split.
        image_paths, _, labels, _ = train_test_split(
            image_paths, labels, train_size=data_fraction, random_state=random_state
        )

    print(f"Dataset prepared with {len(image_paths)} samples.")
    return image_paths, labels

if __name__ == '__main__':
    # Example usage:
    if CONFIG:
        image_paths, labels = prepare_dataset(data_fraction=0.5)
        print(f"Loaded {len(image_paths)} images and {len(labels)} labels.")
