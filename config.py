import yaml
from pathlib import Path

# --- CONFIGURATION ---
# All paths, hyperparameters, and settings are managed here.
CONFIG_YAML = """
# --- Workflow and Path Configuration ---
# project_root_path is now set dynamically below.

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
  # out_dim is now set dynamically in the trainer
  frozen_layers: 10 # Number of initial transformer blocks to freeze

# -- Training Configuration --
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: 'adam'
  resume_training: True # Flag to enable/disable resumable training
  data_fraction: 1.0 # Fraction of data to use for training (1.0 for all data)

# -- DINO Configuration --
dino:
  n_crops: 10 # 2 global, 8 local
  warmup_teacher_temp: 0.04
  teacher_temp: 0.07
  warmup_teacher_temp_epochs: 5
  momentum_teacher: 0.9995

# -- Discovery Configuration --
discovery:
  class_map:
    1: 'person'
    2: 'helmet'
    3: 'vest'

# -- Violation Detection --
violation:
  tracker: 'botsort'
  bot_sort_reid_weights: 'osnet_x0_25_msmt17.pt' # Default weights for BoTSORT
"""

# Load the YAML configuration into a dictionary
try:
    CONFIG = yaml.safe_load(CONFIG_YAML)
except yaml.YAMLError as e:
    print(f"Error loading YAML configuration: {e}")
    CONFIG = {}

# --- Dynamic Path Construction ---
# Set the project root path dynamically to the current working directory.
# This makes the project more portable.
if CONFIG:
    # Set and create the project root path
    project_root = Path.cwd()
    CONFIG['project_root_path'] = str(project_root)

    # Create absolute paths for other directories
    output_dir = project_root / CONFIG['output_dir']
    checkpoint_dir = output_dir / 'checkpoints'

    CONFIG['data_dir_abs'] = project_root / CONFIG['data_dir']
    CONFIG['output_dir_abs'] = output_dir
    CONFIG['checkpoint_dir_abs'] = checkpoint_dir

    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (project_root / CONFIG['data_dir']).mkdir(parents=True, exist_ok=True)

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
