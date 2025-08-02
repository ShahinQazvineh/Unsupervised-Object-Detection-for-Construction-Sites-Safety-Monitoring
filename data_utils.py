import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from roboflow import Roboflow


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


def prepare_dataset(config, data_fraction=1.0, random_state=42):
    """
    Prepares the dataset by loading data and creating a stratified subset if required.

    Args:
        config (dict): The configuration dictionary.

def prepare_dataset(data_dir, data_fraction=1.0, random_state=42):
    """
    Prepares the dataset by loading data and creating a stratified subset if required.
    This is a placeholder function and needs to be adapted to the actual data format.

    Args:
        data_dir (str): The directory where the data is located.

        data_fraction (float): The fraction of the data to use.
        random_state (int): The random state for reproducibility.

    Returns:
        tuple: A tuple containing the data and labels.
    """

    if config['data_source'] == 'roboflow':
        rf = Roboflow(api_key=config['roboflow']['api_key'])
        project = rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])
        dataset = project.version(config['roboflow']['version']).download("yolov5")
        data_dir = dataset.location
    else:
        data_dir = config['data_dir']


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
    from config import load_config
    config = load_config()
    if config:
        data, labels = prepare_dataset(config, data_fraction=0.5)
        generate_temp_data_yaml(data)

    data, labels = prepare_dataset('data/', data_fraction=0.5)
    generate_temp_data_yaml(data)

