import subprocess
import time

def train_cnn():
    while True:
        try:
            # Run your CNN training command using subprocess
            # cmd = ['runas', '/env', '/user:fabio', 'pytest test_training.py::test_train_v2']
            subprocess.run(['pytest', 'test_training.py::test_train_v2'], check=True)
        except subprocess.CalledProcessError as e:
            # If an error occurs (e.g., memory error), print the error and restart the training
            print(f"Error occurred: {e}. Restarting training...")
            time.sleep(60)  # Wait for some time before restarting to avoid constant restarts

if __name__ == "__main__":
    train_cnn()