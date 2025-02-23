# config.py
import os
from pathlib import Path

class ClusterConfig:
    def __init__(self):
        self.USER = os.getenv('USER', 'mswaroop')
        self.BASE_DIR = Path(f"/data/user_data/{self.USER}")
        
        # Project structure
        self.PROJECT_DIR = self.BASE_DIR / "project_name"
        self.CODE_DIR = self.PROJECT_DIR / "code"
        self.DATASET_DIR = self.PROJECT_DIR / "datasets"
        self.RESULTS_DIR = self.PROJECT_DIR / "results"
        self.LOG_DIR = self.PROJECT_DIR / "logs"
        
        # Create directories
        for dir_path in [self.CODE_DIR, self.DATASET_DIR, 
                        self.RESULTS_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_job_name(self, prefix):
        """Generate unique job name"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}"
    
    def setup_gpu(self):
        """Configure GPU settings"""
        import torch
        if torch.cuda.is_available():
            # Set device
            device = torch.device('cuda')
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            # Print GPU info
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
        else:
            device = torch.device('cpu')
            print("WARNING: No GPU available")
        return device

# Usage in main scripts:
config = ClusterConfig()
device = config.setup_gpu()