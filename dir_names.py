import os
user_name = os.environ['USER']
BASE_DIR = f"/data/user_data/{user_name}/neural-lifting"
DATASET_DIR = f"{BASE_DIR}/datasets/"
RESULTS_DIR = f"{BASE_DIR}/results/"
LOG_DIR = f"{BASE_DIR}/logs/"


# Create directories in main() functions
for dir_path in [DATASET_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)