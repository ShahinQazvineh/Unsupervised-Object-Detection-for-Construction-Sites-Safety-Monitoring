import yaml
from pathlib import Path

# --- CONFIGURATION ---
# All paths, hyperparameters, and settings are managed here.
CONFIG_YAML = """
# --- Workflow and Path Configuration ---
project_root_path: "/content/drive/MyDrive/PPE_Violation_Detection_Project"

# --- Data Source Configuration ---
data_source: "roboflow"
roboflow:
  api_key: "YOUR_ROBOFLOW_API_KEY"
  workspace: "skcet-g4h72"
  project: "construction-ppe-rdhzo"
  version: 3

# -- Project Paths --
data_dir: 'data/'
output_dir: 'output/'
checkpoint_dir: 'output/checkpoints/'

# -- Model Configuration --
model:
  name: 'vit_small_patch14_dinov2'
  pretrained: True
  frozen_layers: 10 # Number of initial transformer blocks to freeze

# -- Training Configuration --
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: 'adam'
  resume_training: True # Flag to enable/disable resumable training
  data_fraction: 1.0 # Fraction of data to use for training (1.0 for all data)

# -- Discovery Configuration --
discovery:
  class_map:
    1: 'person'
    2: 'helmet'
    3: 'vest'

# -- Violation Detection --
violation:
  tracker: 'botsort'
"""

# Load the YAML configuration into a dictionary
try:
    CONFIG = yaml.safe_load(CONFIG_YAML)
except yaml.YAMLError as e:
    print(f"Error loading YAML configuration: {e}")
    CONFIG = {}

# --- Dynamic Path Construction ---
# After loading, construct the absolute paths based on the project_root_path.
if CONFIG:
    try:
        root = Path(CONFIG['project_root_path'])
        out_base_abs = root / CONFIG['output_dir']

        CONFIG['data_dir_abs'] = root / CONFIG['data_dir']
        CONFIG['output_dir_abs'] = out_base_abs
        CONFIG['checkpoint_dir_abs'] = out_base_abs / CONFIG['checkpoint_dir']

    except KeyError as e:
        print(f"ERROR: A required key is missing from the base YAML config for path construction: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during dynamic path construction: {e}")

if __name__ == '__main__':
    if CONFIG:
        print("--- Example Config Values ---")
        print(f"Project Root: {CONFIG.get('project_root_path')}")
        print(f"Data Source: {CONFIG.get('data_source')}")
        print(f"Roboflow Workspace: {CONFIG.get('roboflow', {}).get('workspace')}")
        print(f"Roboflow Project: {CONFIG.get('roboflow', {}).get('project')}")
        print(f"Absolute Output Dir: {CONFIG.get('output_dir_abs')}")
    else:
        print("CONFIG dictionary is empty due to a loading error.")


def load_config(config_path='config.yaml'):
    """
    Loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    config = load_config()
    if config:
        print("Configuration loaded successfully:")
        print(config)
