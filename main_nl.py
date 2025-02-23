# main.py: main script to run experiment for training lifted models on CIFAR-10 dataset
import numpy as np
import os
import argparse
import torch
import json
from datetime import datetime
from models import *
from pathlib import Path
from typing import Dict, Any
from aux_functions import get_unique_results_directory
from algo_2 import ModelTrainer
from dir_names import *

class ExperimentManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._setup_device()
        self.configs = self._load_configs()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = self._setup_results_directory()
        
    def _setup_device(self):
        """Configure device(s) for training."""
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"Found {n_gpus} CUDA GPU(s)")
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, {props.total_memory/1e9:.2f}GB memory")
            
            device = torch.device('cuda:0')  # Primary GPU
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
            print("No GPU available, using CPU")
        
        return device
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load and return configuration from JSON file."""
        with open(self.args.config_path, 'r') as f:
            configs = json.load(f)
        
        # Ensure required config sections exist
        if 'model_configs' not in configs:
            configs['model_configs'] = {}
        
        # For each model, ensure it has required config sections
        for model in self.args.models:
            if model not in configs['model_configs']:
                if model == 'LeNetLifted' or model == 'lenetlifted':
                    configs['model_configs'][model] = LeNetLifted()._get_default_config()
                    
        # Ensure training params exist
        if 'training_params' not in configs:
            configs['training_params'] = {}
            
        # # Ensure odd number of cycles # this was for when cycles meant each phase
        # if configs.get('training_params', {}).get('cycles', 0) % 2 == 0:
        #     configs['training_params']['cycles'] += 1
            
        return configs
    
    def _setup_results_directory(self) -> str:
        """Create and return path to results directory with unique naming."""
        # Get unique directory path
        results_dir = get_unique_results_directory(self.args.results_directory)
        
        # Ensure results directory is an absolute path and starts with BASE_DIR
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(BASE_DIR, results_dir)
        elif not results_dir.startswith(BASE_DIR):
            results_dir = os.path.join(BASE_DIR, os.path.relpath(results_dir, '/'))
        # Create subdirectories
        for subdir in ['plots', 'models']:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        print(f"Results will be saved to: {results_dir}")
        return results_dir
    
    def load_and_prepare_data(self):
        """Load and prepare dataset for training."""
        print("Loading dataset...")
        dataset_path = Path(self.args.dataset)
        X = np.load(dataset_path / 'X.npy')
        Y = np.load(dataset_path / 'Y.npy')
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        
        # Subsample if necessary
        if len(X) > self.args.dataset_size:
            print(f"Subsampling dataset to {self.args.dataset_size} samples")
            indices = np.random.choice(len(X), self.args.dataset_size, replace=False)
            X = X[indices]
            Y = Y[indices]
            
        print(f"Dataset size: {len(X)}")
        label_distribution = self._log_label_distribution(Y)
        
        return X, Y, label_distribution
    
    def _log_label_distribution(self, Y: torch.Tensor) -> Dict:
        """Log distribution of labels in dataset."""
        unique, counts = np.unique(Y.numpy(), return_counts=True)
        label_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"Label distribution: {label_distribution}")
        return label_distribution
    
    def save_experiment_config(self, X: torch.Tensor, Y: torch.Tensor, label_distribution: Dict):
        """Save experiment configuration to JSON."""
        config = {
            'dataset_info': {
                'size': len(X),
                'label_counts': label_distribution,
                'path': self.args.dataset
            },
            'training_params': self.configs.get('training_params', {}),
            'model_configs': {model: self.configs.get(f'{model}', {}) for model in self.args.models},
            'models_trained': self.args.models
        }
        
        with open(os.path.join(self.results_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    def train_models(self, X: torch.Tensor, Y: torch.Tensor):
        """Train all specified models."""
        for model_type in self.args.models:
            print(f"\nTraining Model {model_type}")
            
            # Prepare model configuration
            model_config = self.configs.get('model_configs', {}).get(model_type, LeNetLifted()._get_default_config())
            model_name = f'{model_type}_{self.timestamp}'                
            
            # Create trainer configuration
            trainer_config = {
                **self.configs.get('training_params', {}),
                'model_type': model_type,
                'model_config': model_config,
                'model_name': model_name,
                'seed': self.args.seed
            }
            
            # Initialize and run trainer
            trainer = ModelTrainer(trainer_config)
            # Split into train and test if needed
            if trainer_config.get('n_folds', 3) == 1:
                # For CIFAR-10, test set is fixed
                test_size = self.configs.get('training_params', {}).get('test_size', 10000)
                X_train = X[:-test_size]
                y_train = Y[:-test_size]
                X_test = X[-test_size:]
                y_test = Y[-test_size:]
                print("Training without validation set...")
                model, metrics = trainer.train_with_fixed_val(X_train, y_train, X_test, y_test, self.device)
            else:
                model, metrics = trainer.train(X, Y, self.device)
            
            # Save results
            self._save_training_results(model, metrics, model_name)
    
    def _save_training_results(self, model: torch.nn.Module, metrics: Dict, model_name: str):
        """Save model weights and plot training metrics."""
        # Save model weights
        torch.save(
            model.state_dict(), 
            os.path.join(self.results_dir, 'models', f'{model_name}.pt')
        )

        # Plot metrics
        if self.args.with_val:
            print(f"Validation metrics for {model_name}:")
            for fold, fold_metrics in metrics.items():
                plot_metrics_v5(
                    metrics=fold_metrics,
                    save_path=os.path.join(self.results_dir, 'plots', f'metrics_{model_name}_fold_{fold}.png'),
                    show=False,
                    log_scale=False
                )
        else:
            print(f"Validation metrics for {model_name}:")
            for fold, fold_metrics in metrics.items():
                plot_metrics_v5(
                    metrics=fold_metrics,
                    save_path=os.path.join(self.results_dir, 'plots', f'metrics_{model_name}_fold_{fold}.png'),
                    show=False,
                    log_scale=False
                )

        # saving the metrics to a json file
        with open(os.path.join(self.results_dir, 'plots', f'metrics_{model_name}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # simply print out the model to a .txt file
        with open(os.path.join(self.results_dir, 'models', f'{model_name}.txt'), 'w') as f:
            f.write(str(model))
        


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=f'{BASE_DIR}/datasets/lambdas/dataset_0.47/')
    parser.add_argument('--results_directory', type=str, default=f'{BASE_DIR}/results/run_val/run_1/')
    parser.add_argument('--config_path', type=str, default=f'{BASE_DIR}/configs/model_configs.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--with_val', action='store_true')
    parser.add_argument('--dataset_size', type=int, default=60000)
    parser.add_argument('--models', nargs='+', default=['LeNetLifted'])
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize experiment manager
    experiment = ExperimentManager(args)
    
    # Load and prepare data
    X, Y, label_distribution = experiment.load_and_prepare_data()
    
    # Save experiment configuration
    experiment.save_experiment_config(X, Y, label_distribution)

    # Save all the arguments
    with open(os.path.join(experiment.results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train models
    experiment.train_models(X, Y)
    
    print(f"\nExperiment completed. Results saved to: {experiment.results_dir}")

if __name__ == '__main__':
    main()