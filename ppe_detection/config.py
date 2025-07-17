import yaml

def load_config(config_path='ppe_detection/config.yaml'):
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
