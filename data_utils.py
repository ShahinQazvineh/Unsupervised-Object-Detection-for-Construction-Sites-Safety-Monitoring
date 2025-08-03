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

def create_stratified_subset(data, labels, fraction, random_state=42):
    """
    Creates a stratified subset of the data.

    Args:
        data (list or np.array): The full dataset.
        labels (list or np.array): The labels for the data.
        fraction (float): The fraction of the data to include in the subset.
        random_state (int): The random state for reproducibility.

    Returns:
        tuple: A tuple containing the subset of data and labels.
    """
    if fraction == 1.0:
        return data, labels

    # Use train_test_split to create a stratified split. We are only interested in the "train" part.
    subset_data, _, subset_labels, _ = train_test_split(
        data, labels, train_size=fraction, stratify=labels, random_state=random_state
    )
    return subset_data, subset_labels

def prepare_dataset(data_fraction=1.0, random_state=42):
    """
    Prepares the dataset by loading data and creating a stratified subset if required.

    Args:
        data_fraction (float): The fraction of the data to use.
        random_state (int): The random state for reproducibility.

    Returns:
        tuple: A tuple containing the data (image paths) and labels (bounding boxes).
        tuple: A tuple containing the data and labels.
    """
    if CONFIG['data_source'] == 'roboflow':
        rf = Roboflow(api_key=CONFIG['roboflow']['api_key'])
        project = rf.workspace(CONFIG['roboflow']['workspace']).project(CONFIG['roboflow']['project'])
        dataset = project.version(CONFIG['roboflow']['version']).download("yolov5")
        data_dir = dataset.location
    else:
        data_dir = CONFIG['data_dir_abs']

    image_paths = glob.glob(os.path.join(data_dir, 'train/images/*.jpg'))
    labels = []
    for img_path in image_paths:
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        # We need the image dimensions to convert YOLO format to pixel coordinates.
        # We'll get this from the image itself.
        from PIL import Image
        with Image.open(img_path) as img:
            width, height = img.size
        boxes = parse_yolo_labels(label_path, width, height)
        labels.append(boxes)


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
        data_dir = CONFIG['data_dir']


    # This is a placeholder. In a real scenario, you would load your images and labels here.
    # For demonstration purposes, we'll create some dummy data.
    print("Loading and preparing data...")
    # Let's assume we have 1000 data points and 3 classes
    num_samples = 1000
    data = [os.path.join(data_dir, f'image_{i}.jpg') for i in range(num_samples)]
    labels = np.random.randint(1, 4, size=num_samples) # Labels are 1, 2, 3

    if data_fraction < 1.0:
        print(f"Creating a stratified subset of the data with fraction: {data_fraction}")
        data, labels = create_stratified_subset(data, labels, data_fraction, random_state)

    print(f"Dataset prepared with {len(data)} samples.")
    return data, labels

def generate_temp_data_yaml(subset_data, yaml_path='temp_data.yaml'):
    """
    Generates a temporary YAML file for the data subset.
    This is a placeholder and should be adapted to the actual data loader's needs.
    """
    # The structure of this YAML will depend on the data loader of your training framework.
    # For example, it might need paths to training and validation sets.
    data_config = {
        'train': subset_data,
        'val': subset_data  # In a real scenario, you'd have a separate validation set
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    print(f"Temporary data YAML created at: {yaml_path}")


if __name__ == '__main__':
    # Example usage:
    from config import CONFIG
    if CONFIG:
        data, labels = prepare_dataset(CONFIG, data_fraction=0.5)
        generate_temp_data_yaml(data)
