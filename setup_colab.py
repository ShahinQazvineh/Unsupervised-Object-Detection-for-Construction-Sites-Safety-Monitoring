import os
import subprocess
import sys

def mount_drive(drive_path='/content/drive'):
    """
    Mounts Google Drive in the Colab environment.
    """
    try:
        from google.colab import drive
        drive.mount(drive_path)
    except ImportError:
        print("This function is intended to be used in a Google Colab environment.")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")

def install_dependencies(requirements_path='requirements.txt'):
    """
    Installs dependencies from a requirements.txt file.
    """
    if not os.path.exists(requirements_path):
        print(f"Error: {requirements_path} not found. Please create a requirements file.")
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

def setup_environment(drive_path='/content/drive', requirements_path='requirements.txt'):
    """
    Sets up the Colab environment by mounting Google Drive and installing dependencies.
    """
    print("Setting up the Colab environment...")
    mount_drive(drive_path)
    install_dependencies(requirements_path)
    print("Environment setup complete.")

if __name__ == '__main__':
    # This block will not run in a Colab notebook directly,
    # but it's good practice to have it.
    # In the notebook, you would import and call the functions.
    print("To set up your Colab environment, import these functions into your notebook and call them.")
    print("Example:")
    print("from setup_colab import setup_environment")
    print("setup_environment()")
