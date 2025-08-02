# Unsupervised PPE Detection

This project is a system for detecting Personal Protective Equipment (PPE) violations in images or video streams. It uses an unsupervised object discovery approach with a Vision Transformer (ViT) model to identify people, helmets, and vests. The system then uses a tracker to determine if people are wearing the necessary PPE and flags any violations.

## Features

- **Unsupervised Object Discovery:** Uses a pre-trained DINOv2 model to discover objects in an image without requiring bounding box annotations for training.
- **PPE Violation Detection:** Identifies instances where a person is not wearing a helmet or a vest.
- **Object Tracking:** Employs BoTSORT for robust object tracking across video frames.
- **Configurable:** Project settings, including model parameters, training configurations, and paths, can be easily modified through a `config.yaml` file.
- **Resumable Training:** Training can be paused and resumed, saving time and computational resources.
- **Data Subsetting:** Allows for training on a fraction of the data for faster experimentation.

## Technology Stack

- **Backend:** Python
- **Deep Learning Framework:** PyTorch
- **Model:** DINOv2 (Vision Transformer) via `timm`
- **Object Tracking:** BoTSORT via `boxmot`
- **Configuration:** PyYAML
- **Data Handling:** NumPy, scikit-learn

## Project Structure

```
.
├── .gitignore
├── config.py             # Loads the configuration from config.yaml
├── config.yaml           # Configuration file for the project
├── data_utils.py         # Utilities for data loading and preparation
├── discovery_processor.py  # Handles object discovery using the DINOv2 model
├── main_colab.ipynb      # Jupyter notebook for running the project in Google Colab
├── requirements.txt      # Python dependencies
├── setup_colab.py        # Setup script for Google Colab
└── violation_processor.py  # Processes discovered objects to detect PPE violations
```

## Setup and Usage

### 1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure the project:

Modify the `config.yaml` file to set up your project paths, model parameters, and training settings.

#### Key Configuration Options:

- `data_dir`: Directory containing your dataset.
- `output_dir`: Directory to save outputs, such as model checkpoints.
- `model`:
    - `name`: The name of the Vision Transformer model to use from `timm`.
    - `frozen_layers`: Number of initial transformer blocks to freeze during training.
- `training`:
    - `epochs`: Number of training epochs.
    - `batch_size`: Batch size for training.
    - `learning_rate`: The learning rate for the optimizer.
    - `resume_training`: Set to `true` to resume training from the latest checkpoint.
    - `data_fraction`: Fraction of the dataset to use for training (e.g., `0.5` for 50%).
- `discovery`:
    - `class_map`: A mapping of cluster IDs to class names (e.g., `1: 'person'`).
- `violation`:
    - `tracker`: The object tracker to use (e.g., `'botsort'`).

### 4. Running the Project:

The `main_colab.ipynb` notebook provides a step-by-step guide to running the entire pipeline, from data preparation to violation detection. Open it in a Jupyter environment or Google Colab to get started.

## How It Works

1.  **Object Discovery:** The `DiscoveryProcessor` loads a pre-trained DINOv2 model. Given an image, it extracts attention maps from the model's transformer blocks. These attention maps highlight salient regions, which correspond to objects. While this example uses a simplified approach to generate masks, a full implementation would involve more sophisticated clustering of patch features to group objects.

2.  **Class Mapping:** The discovered objects (initially just clusters of pixels) are assigned semantic labels based on a manually defined `class_map` in the `config.yaml` file.

3.  **Violation Detection:** The `ViolationProcessor` takes the list of discovered and labeled objects. It uses the BoTSORT tracker to track objects across frames. The logic then checks if each tracked "person" object is associated with "helmet" and "vest" objects. If not, it flags a violation.

## Future Improvements

- Implement a more advanced object discovery mechanism using clustering of patch features.
- Integrate with a live video stream for real-time violation detection.
- Add support for more types of PPE.
- Develop a user interface for visualizing the results.
