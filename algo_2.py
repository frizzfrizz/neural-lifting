# algo_2.py: Implementation of the algorithm 2: Neural Lifting with Interpolated Loss
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
import json
from models import LeNetLifted
from tqdm import tqdm
from dir_names import *
from visualisers import plot_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Training phases enum
from enum import Enum
class TrainingPhase(Enum):
    SHORTCUT = 'shortcut'
    LIFTED = 'lifted'
    MATCHING = 'matching'

def train_test_split(X, y, test_size=10000, random_state=None):
    """Split data into train and test sets for the CIFAR-10 dataset."""
    total_samples = len(X)
    train_val_indices = slice(0, total_samples - test_size)
    test_indices = slice(total_samples - test_size, total_samples) # CIFAR benchmark test set

    X_trainval, X_test = X[train_val_indices], X[test_indices]
    y_trainval, y_test = y[train_val_indices], y[test_indices]

    return X_trainval, X_test, y_trainval, y_test

def setup_model(model, device):
    """Set up model for single/multi-GPU training."""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    return model.to(device)

def setup_device():
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

class LiftedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, latent_dim, device):
        self.X = X.to(device)
        self.y = F.one_hot(y, num_classes=10).float().to(device)
        self.target_z = torch.zeros(len(X), latent_dim).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.target_z[idx]
    
    def update_targets(self, model):
        model.eval()
        print("Updating targets... | model.lift = ", model.lift)
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(self, batch_size=128, shuffle=False)
            all_targets = []
            for x_batch, _, _ in loader:
                _, z = model(x_batch)
                all_targets.append(z)
            self.target_z = torch.cat(all_targets, dim=0)
        model.train()
    
    def reset_targets(self):
        self.target_z = torch.zeros_like(self.target_z)

class TrainingCycle:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_phase = TrainingPhase.SHORTCUT
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.matching_steps = 0
        self.phase_epoch_counter = 0 
        # self.stored_trunk_params = None 
        self.has_completed_first_cycle = False
        self.stored_targets = None
        self.lifter_config = config['model_config']['params'].get('lifter_config', {}) if 'params' in config['model_config'] else config['model_config'].get('lifter_config', {})
        print("using alpha: ", self.config.get('alpha', 0.5))
    
    def get_model_module(self):
        """Get model module regardless of DataParallel wrapping."""
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
    
    def store_trunk_params(self, model= None):
        if model is None:
            self.stored_trunk_params = {k: v.clone().detach() for k, v in self.model.trunk.named_parameters()}
        else:
            self.stored_trunk_params = {k: v.clone().detach() for k, v in model.trunk.named_parameters()}
    
    def should_switch_phase(self, val_loss, train_test = False):
        self.phase_epoch_counter += 1
        """Determine if we should switch to next phase based on validation loss."""
        if train_test:
            max_epochs_per_phase = self.config.get('max_epochs_per_phase', 50)
            if self.phase_epoch_counter >= max_epochs_per_phase:
                print(f"\nReached max epochs ({max_epochs_per_phase}) for phase {self.current_phase}")
                return True
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        if self.current_phase == TrainingPhase.MATCHING:
            return self.patience_counter >= self.config['patience']
        else:     
            return self.patience_counter >= self.config['patience']

    def switch_to_next_phase(self):
        """Switch to next training phase."""
        self.phase_epoch_counter = 0
        old_phase = self.current_phase
        model_module = self.get_model_module()
        device = next(self.model.parameters()).device
        
        if self.current_phase == TrainingPhase.SHORTCUT:
            # with torch.no_grad():
            #     self.store_trunk_params(model_module)
            self.current_phase = TrainingPhase.LIFTED
            model_module.lift = True
            model_module._initialize_lifter(
                init_type=self.lifter_config.get('init_type', 'kaiming'),
                sparsity=self.lifter_config.get('sparsity', 0.0),
                scale=self.lifter_config.get('scale', None)
            )
            model_module.lifter = model_module.lifter.to(device)
            
            for param in self.model.parameters():
                param.requires_grad = True
            
            for param in self.model.lifter.parameters():
                param.requires_grad = True
            for param in self.model.head.parameters():
                param.requires_grad = True
            for param in self.model.trunk.parameters():
                param.requires_grad = True
                
        elif self.current_phase == TrainingPhase.LIFTED:
            self.current_phase = TrainingPhase.MATCHING
            model_module.lift = False
            # model_module.trunk.load_state_dict(self.stored_trunk_params)
            
            for param in model_module.lifter.parameters():
                param.requires_grad = False
            for param in model_module.head.parameters():
                param.requires_grad = False
                # param.requires_grad = True
            for param in model_module.trunk.parameters():
                param.requires_grad = True
                
        else:  # MATCHING to LIFTED NOW
            # with torch.no_grad():
            #     self.store_trunk_params(model_module)
            self.current_phase = TrainingPhase.LIFTED
            model_module.lift = True
            model_module._initialize_lifter(
                init_type=self.lifter_config.get('init_type', 'kaiming'),
                sparsity=self.lifter_config.get('sparsity', 0.0),
                scale=self.lifter_config.get('scale', None)
            )
            model_module.lifter = model_module.lifter.to(device)
            
            for param in self.model.parameters():
                param.requires_grad = True
            
            for param in self.model.lifter.parameters():
                param.requires_grad = True
            for param in self.model.head.parameters():
                param.requires_grad = True
            for param in self.model.trunk.parameters():
                param.requires_grad = True

        print(f"\n\tPhase switch: {old_phase} -> {self.current_phase}")
        print(f"\tModel lift enabled: {model_module.lift}")
        print("\n\tParameter training status:")
        print(f"\t\tTrunk trainable: {any(p.requires_grad for p in model_module.trunk.parameters())}")
        print(f"\t\tLifter trainable: {any(p.requires_grad for p in model_module.lifter.parameters())}")
        print(f"\t\tHead trainable: {any(p.requires_grad for p in model_module.head.parameters())}")
        
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.matching_steps = 0
        
    def compute_loss(self, y_pred, y_batch, z, target_z, alpha=0.5):
        """Phase-specific loss computation."""
        if self.current_phase == TrainingPhase.MATCHING:
            match_loss = F.mse_loss(z, target_z)
            # with torch.no_grad():
            #     ce_loss = F.cross_entropy(y_pred, y_batch.argmax(dim=1))
            ce_loss = F.cross_entropy(y_pred, y_batch.argmax(dim=1))
            alpha = self.config.get('alpha', 0.5)
            return alpha * match_loss + (1.0-alpha) * ce_loss, ce_loss, match_loss
            # return match_loss, ce_loss, match_loss
            
        elif self.current_phase in [TrainingPhase.SHORTCUT, TrainingPhase.LIFTED]:
            # bce_loss = F.binary_cross_entropy(y_pred[:, 1], y_batch[:, 1])
            ce_loss = F.cross_entropy(y_pred, y_batch.argmax(dim=1))
            with torch.no_grad():
                match_loss = F.mse_loss(z, target_z)
            # return bce_loss, bce_loss, match_loss
            return ce_loss, ce_loss, match_loss

        raise ValueError(f"Unknown phase: {self.current_phase}")

class MetricsTracker:
    def __init__(self, n_folds):
        self.metrics = {i: self._init_fold_metrics() for i in range(n_folds)}
        self.aux_vars = {i: self._init_aux_vars() for i in range(n_folds)}
    
    def _init_fold_metrics(self):
        return {
            'epoch': [], 'phases': [],
            'train_loss': [], 'train_ce_loss': [], 'train_match_loss': [], 
            'train_acc': [],
            'val_loss': [], 'val_ce_loss': [], 'val_match_loss': [], 
            'val_acc': [], 'val_shortcut_acc': [], 'val_lifted_acc': [],
            'test_loss': [], 'test_ce_loss': [], 'test_match_loss': [],
            'test_acc': [], 'test_shortcut_acc': [], 'test_lifted_acc': [],
            'final_val_loss': [], 'final_val_acc': [],
            'final_test_loss': [], 'final_test_acc': []
        }
    
    def _init_aux_vars(self):
        return {
            'epoch': [], 'model_lift': [], 
            # 'lambda_match': [],
            'patience_counter': [], 'cycle_counter': []
        }
    
    def update_metrics(self, fold, epoch, metrics_dict, extra_info=None):
        for key, value in metrics_dict.items():
            if key not in self.metrics[fold]:
                self.metrics[fold][key] = []
            if isinstance(value, (list, np.ndarray)):
                self.metrics[fold][key].extend(value)
            else:
                self.metrics[fold][key].append(value)

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.metrics_tracker = MetricsTracker(config['n_folds'])
        self.device = setup_device()
        self.model = None
        self.lifter_config = config['model_config']['params'].get('lifter_config', {}) if 'params' in config['model_config'] else config['model_config'].get('lifter_config', {})
    
    def get_model_module(self, model=None):
        """Get model module regardless of DataParallel wrapping."""
        if model is None:
            model = self.model
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
    def _create_data_loaders(self, X, y, device):
        """Create data loaders with proper device management."""
        # Keep data on CPU until batching
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=self.config['seed']
        )
        
        # Create test loader
        test_loader = self._create_loader(
            X_test, y_test, 
            self.config['batch_size'],
            device=device,  # Pass device explicitly
            shuffle=False
        )
        
        # Setup k-fold
        kfold = KFold(
            n_splits=self.config['n_folds'], 
            shuffle=True, 
            random_state=self.config['seed']
        )
        
        return X_trainval, y_trainval, test_loader, kfold
    
    def _create_loader(self, X, y, batch_size, device, shuffle=True, scale = True):
        if len(X.shape) == 5:
            X = X.squeeze(1)
        if scale:
            X = (X - X.mean()) / (2 * X.std())
        """Create data loader with proper device management."""
        latent_dim = self.config['model_config']['params']['trunk_config']['latent_dim'] if 'params' in self.config['model_config'] else self.config['model_config']['trunk_config']['latent_dim']
        dataset = LiftedDataset(X.clone().detach(), y.clone().detach(), latent_dim, device)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # pin_memory=True if device.type == 'cuda' else False,
            num_workers=0,  # set to 0 to avoid CUDA initialization issues
            persistent_workers=False
        )
    
    def _verify_model_device(self, model, device):
        """Verify all model parameters are on the correct device."""
        if isinstance(model, torch.nn.DataParallel):
            # For DataParallel models, parameters should be on cuda:0
            expected_device = torch.device('cuda:0')
        else:
            expected_device = device
            
        for name, param in model.named_parameters():
            if param.device != expected_device:
                print(f"Parameter {name} is on {param.device}, moving to {expected_device}")
                param.data = param.data.to(expected_device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(expected_device)

    def _initialize_model(self, device):
        print(f"Initializing model with device: {device}")
        """Initialize model with proper device handling."""
        if self.config['model_type'] == 'LeNetLifted':
            print(f'self.config: {self.config}')
            # Get the correct config structure
            model_config = self.config['model_config']['params'] if 'params' in self.config['model_config'] else self.config['model_config']
            print(f"Initialising model with Model config: {model_config}")
            # Create model
            model = LeNetLifted(config=model_config)
            # Initialize parameters before moving to device
            print(f'Initializing model with seed: {self.config["seed"]}')
            model._initialize_modules(seed=self.config['seed'], init_type=self.config.get('init_type', 'xavier'))
            # Initialize lifter if specified
            print(f'Lifter init: {self.config.get("lifter_init")}')
            if self.config.get('lifter_init'):
                model._initialize_lifter(init_type=self.lifter_config.get('init_type', 'xavier'),
                                        sparsity=self.lifter_config.get('sparsity', 0.0),
                                        scale=self.lifter_config.get('scale', None))
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            # Move to device first
            model = model.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Apply DataParallel after moving to device
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                
            self.model = model
            return self.model

    def _initialize_optimizer(self, model):
        """Initialize optimizer(s) for the model."""
        if self.config.get('separate_optimizer', False):
            optimizer_shortcut = torch.optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001), weight_decay=self.config.get('weight_decay', 0.0))
            optimizer_shortcut.state = defaultdict(dict)  # reset optimizer state
            optimizer_lifted = torch.optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001), weight_decay=self.config.get('weight_decay', 0.0))
            optimizer_lifted.state = defaultdict(dict)
            return {
                'shortcut': optimizer_shortcut,
                'lifted': optimizer_lifted
            }
        else:
            lr = self.config.get('lr_shortcut', self.config.get('lr', 0.001))
            # optimizer = torch.optim.Adam(model.parameters(), 
            #                              lr=lr,
            #                              weight_decay=self.config.get('weight_decay', 0.0))
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.config.get('weight_decay', 0)
            )
            optimizer.state = defaultdict(dict)  # reset optimizer state
            
            scheduler = None
            if self.config.get('scheduler_type'):
                scheduler = self._get_scheduler(optimizer)
            
            self.optimizer = optimizer
            self.scheduler = scheduler
            return optimizer, scheduler
    
    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler based on config."""
        if self.config['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.config.get('factor', 0.1),
                patience=self.config.get('patience', 5), verbose=True
            )
        elif self.config['scheduler_type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif self.config['scheduler_type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.get('T_max', 100)
            )
        else:
            scheduler = None
        return scheduler
        
    def _train_epoch(self, model, train_loader, optimizer, training_cycle):
        """Train for one epoch."""
        model.train()
        metrics = {
            'loss': 0.0,
            'ce_loss': 0.0,
            'match_loss': 0.0,
            'acc': 0.0
        }
        n_batches = 0
        device = next(model.parameters()).device
        model_module = self.get_model_module(model)

        # print(f"\tOptimizer: {optimizer}")
        if training_cycle.current_phase == TrainingPhase.LIFTED:
            lr = self.config.get('lr_lifted', self.config.get('lr', 0.001))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif training_cycle.current_phase == TrainingPhase.MATCHING:
            lr = self.config.get('lr_matching', self.config.get('lr', 0.001))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.config.get('lr_shortcut', self.config.get('lr', 0.01))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        for i, (x_batch, y_batch, target_z_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            # Get optimizer based on current phase
            if isinstance(optimizer, dict):
                current_optimizer = optimizer['lifted'] if model_module.lift else optimizer['shortcut']
            else:
                current_optimizer = optimizer
            
            current_optimizer.zero_grad()
            
            # Generate target for matching if in matching phase
            with torch.no_grad():
                if training_cycle.current_phase == TrainingPhase.SHORTCUT or training_cycle.current_phase == TrainingPhase.LIFTED:
                    model_module.lift = not model_module.lift
                    _, target_z = model(x_batch)
                    model_module.lift = not model_module.lift
                else:
                    target_z = target_z_batch
            
            # Forward pass
            y_pred, z = model(x_batch)

            total_loss, ce_loss, match_loss = training_cycle.compute_loss(
                    y_pred, y_batch, z, target_z
                )
            
            # Backward pass
            total_loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize() 
            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['gradient_clip']
                )
            
            current_optimizer.step()

            if device.type == 'cuda':
                torch.cuda.synchronize() 
            
            # Compute accuracy
            acc = (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().mean()
            
            # Update metrics
            metrics['loss'] += total_loss.item()
            metrics['ce_loss'] += ce_loss.item()
            # Handle match_loss differently based on phase
            if isinstance(match_loss, torch.Tensor):
                metrics['match_loss'] += match_loss.item()
            else:
                metrics['match_loss'] += match_loss
            metrics['acc'] += acc.item()
            n_batches += 1
        
        # Average metrics
        return {k: v/n_batches for k, v in metrics.items()}

    def _evaluate(self, model, loader, training_cycle):
        """Evaluate model on given loader."""
        model.eval()
        metrics = defaultdict(float)
        n_batches = 0

        model_module = self.get_model_module(model)
        
        with torch.no_grad():
            for i, (x_batch, y_batch, target_z_batch) in enumerate(loader):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                if training_cycle.current_phase in [TrainingPhase.SHORTCUT, TrainingPhase.LIFTED]:
                    model_module.lift = not model_module.lift
                    _, target_z = model(x_batch)
                    model_module.lift = not model_module.lift
                else:  # Matching phase
                    target_z = target_z_batch
                
                # Get predictions for current configuration
                y_pred, z = model(x_batch)
                total_loss, ce_loss, match_loss = training_cycle.compute_loss(
                    y_pred, y_batch, z, target_z
                )
                
                metrics['loss'] += total_loss.item()
                metrics['ce_loss'] += ce_loss.item()
                metrics['match_loss'] += match_loss.item()
                metrics['acc'] += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().mean().item()
                
                # Evaluate both paths
                model_lift = model_module.lift
                
                model_module.lift = True
                y_pred, _ = model(x_batch)
                metrics['lifted_acc'] += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().mean().item()
                
                model_module.lift = False
                y_pred, _ = model(x_batch)
                metrics['shortcut_acc'] += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).float().mean().item()
                
                model_module.lift = model_lift
                n_batches += 1
        
        return {k: v/n_batches for k, v in metrics.items()}
    
    def train(self, X, y, device='cpu'):
        """Main training loop."""
        X_trainval, y_trainval, test_loader, kfold = self._create_data_loaders(X, y, device)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval)):
            print(f"Fold {fold + 1}/{self.config['n_folds']}")
            
            # Create fold-specific data loaders
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
            
            train_loader = self._create_loader(X_train, y_train, self.config['batch_size'], device)
            val_loader = self._create_loader(X_val, y_val, self.config['batch_size'], device, shuffle=False)
            
            # Initialize model and training components
            model = self._initialize_model(device)
            # model = setup_model(model, device)
            training_cycle = TrainingCycle(model, self.config)
            optimizer, scheduler = self._initialize_optimizer(model)
            
            cycle_counter = 0
            force_final_shortcut = False
            
            for epoch in tqdm(range(self.config['epochs']), desc=f"Fold {fold + 1}"):
                # Training phase
                train_metrics = self._train_epoch(
                    model, train_loader, optimizer, training_cycle
                )
                
                # Evaluation phases
                val_metrics = self._evaluate(model, val_loader, training_cycle)
                test_metrics = self._evaluate(model, test_loader, training_cycle)
                
                if self.scheduler is not None:
                    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

                model_module = self.get_model_module(model)
                # Update metrics
                self.metrics_tracker.update_metrics(
                    fold, epoch,
                    {
                        'epoch': epoch,
                        'phases': training_cycle.current_phase.value,  # Instead of [training_cycle.current_phase.value]
                        'train_loss': train_metrics['loss'],
                        'train_ce_loss': train_metrics['ce_loss'],
                        'train_match_loss': train_metrics['match_loss'],
                        'train_acc': train_metrics['acc'],
                        'val_loss': val_metrics['loss'],
                        'val_ce_loss': val_metrics['ce_loss'],
                        'val_match_loss': val_metrics['match_loss'],
                        'val_acc': val_metrics['acc'],
                        'val_shortcut_acc': val_metrics['shortcut_acc'],
                        'val_lifted_acc': val_metrics['lifted_acc'],
                        'test_loss': test_metrics['loss'],
                        'test_ce_loss': test_metrics['ce_loss'],
                        'test_match_loss': test_metrics['match_loss'],
                        'test_acc': test_metrics['acc'],
                        'test_shortcut_acc': test_metrics['shortcut_acc'],
                        'test_lifted_acc': test_metrics['lifted_acc']
                    },
                    {
                        'epoch': epoch,
                        'model_lift': model_module.lift,
                        # 'lambda_match': self.config['lambda_match'],
                        'patience_counter': training_cycle.patience_counter,
                        'cycle_counter': cycle_counter
                    }
                )
                
                # Check for phase transition
                if training_cycle.should_switch_phase(val_metrics['loss']):
                    print(f"Switching phase at epoch {epoch}")
                    print(f"\t Train loss at the end of {training_cycle.current_phase}: {train_metrics['loss']:.4f}")
                    print(f"\t Train accuracy at the end of {training_cycle.current_phase}: {train_metrics['acc']:.4f}")
                    print(f"\t Val loss at the end of {training_cycle.current_phase}: {val_metrics['loss']:.4f}")
                    print(f"\t Val accuracy at the end of {training_cycle.current_phase}: {val_metrics['acc']:.4f}")
                    print(f"\t Test loss at the end of {training_cycle.current_phase}: {test_metrics['loss']:.4f}")
                    print(f"\t Test accuracy at the end of {training_cycle.current_phase}: {test_metrics['acc']:.4f}")
                    
                    if training_cycle.current_phase == TrainingPhase.LIFTED:
                        train_loader.dataset.update_targets(model_module)
                        val_loader.dataset.update_targets(model_module)
                        test_loader.dataset.update_targets(model_module)
                    elif training_cycle.current_phase == TrainingPhase.MATCHING:
                        train_loader.dataset.reset_targets()
                        val_loader.dataset.reset_targets()
                        test_loader.dataset.reset_targets()
                        cycle_counter += 1
                        if cycle_counter >= self.config['cycles']:
                            print(f"Completed {self.config['cycles']} cycles at epoch {epoch}")
                            break
                    training_cycle.switch_to_next_phase()
        
            # Get final test accuracy without lifting
            model_module = self.get_model_module(model)
            # setting training_cycle.current_phase to SHORTCUT to ensure no lifting
            training_cycle.current_phase = TrainingPhase.SHORTCUT
            model_module.lift = False
            print(f"Model.lift: {model.lift}")
            final_test_metrics = self._evaluate(model, test_loader, training_cycle)
            print(f"Final test accuracy (without lifting): {final_test_metrics['acc']:.4f}")
            final_val_metrics = self._evaluate(model, val_loader, training_cycle)

            self.metrics_tracker.update_metrics(
                fold, epoch + 1,  # Using epoch + 1 to indicate this is after training
                {
                    'final_val_loss': final_val_metrics['loss'],
                    'final_val_acc': final_val_metrics['acc'],
                    'final_test_loss': final_test_metrics['loss'],
                    'final_test_acc': final_test_metrics['acc']
                },
                {}
            )

        return model, self.metrics_tracker.metrics

    def train_val_only(self, X, y, device='cpu'):
        """Train and validate model without test evaluation for hyperparameter selection."""
        self.device = device
        
        # Create train and validation splits using KFold
        kfold = KFold(
            n_splits=self.config['n_folds'], 
            shuffle=True, 
            random_state=self.config['seed']
        )
        
        fold_metrics = {}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold + 1}/{self.config['n_folds']}")
            
            # Create fold-specific data loaders
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_loader = self._create_loader(X_train, y_train, self.config['batch_size'], device)
            val_loader = self._create_loader(X_val, y_val, self.config['batch_size'], device, shuffle=False)
            
            # Initialize model and training components
            self.model = self._initialize_model(device)
            training_cycle = TrainingCycle(self.model, self.config)
            optimizer, scheduler = self._initialize_optimizer(self.model)
            
            cycle_counter = 0
            force_final_shortcut = False
            
            for epoch in tqdm(range(self.config['epochs']), desc=f"Fold {fold + 1}"):
                # Training phase
                train_metrics = self._train_epoch(
                    self.model, train_loader, optimizer, training_cycle
                )
                
                # Validation phase only
                val_metrics = self._evaluate(self.model, val_loader, training_cycle)
                
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

                model_module = self.get_model_module()
                
                # Update metrics (simplified to only include train/val metrics)
                self.metrics_tracker.update_metrics(
                    fold, epoch,
                    {
                        'epoch': epoch,
                        'phases': training_cycle.current_phase.value,
                        'train_loss': train_metrics['loss'],
                        'train_ce_loss': train_metrics['ce_loss'],
                        'train_match_loss': train_metrics['match_loss'],
                        'train_acc': train_metrics['acc'],
                        'val_loss': val_metrics['loss'],
                        'val_ce_loss': val_metrics['ce_loss'],
                        'val_match_loss': val_metrics['match_loss'],
                        'val_acc': val_metrics['acc'],
                        'val_shortcut_acc': val_metrics['shortcut_acc'],
                        'val_lifted_acc': val_metrics['lifted_acc']
                    },
                    {
                        'epoch': epoch,
                        'model_lift': model_module.lift,
                        'patience_counter': training_cycle.patience_counter,
                        'cycle_counter': cycle_counter
                    }
                )
                
                about_to_end = (epoch == self.config['epochs'] - 1) or \
                              (cycle_counter >= self.config['cycles'] - 1 and 
                               training_cycle.current_phase == TrainingPhase.MATCHING)
                               
                if about_to_end and training_cycle.current_phase != TrainingPhase.SHORTCUT:
                    force_final_shortcut = True
                    
                # Check for phase transition
                if training_cycle.should_switch_phase(val_metrics['loss']):
                    print(f"Switching phase at epoch {epoch}")
                    print(f"\t Train loss at the end of {training_cycle.current_phase}: {train_metrics['loss']:.4f}")
                    print(f"\t Train accuracy at the end of {training_cycle.current_phase}: {train_metrics['acc']:.4f}")
                    print(f"\t Val loss at the end of {training_cycle.current_phase}: {val_metrics['loss']:.4f}")
                    print(f"\t Val accuracy at the end of {training_cycle.current_phase}: {val_metrics['acc']:.4f}")
                    # print(f"\t Test loss at the end of {training_cycle.current_phase}: {test_metrics['loss']:.4f}")
                    # print(f"\t Test accuracy at the end of {training_cycle.current_phase}: {test_metrics['acc']:.4f}")
                    
                    if training_cycle.current_phase == TrainingPhase.LIFTED:
                        train_loader.dataset.update_targets(model_module)
                        val_loader.dataset.update_targets(model_module)
                    elif training_cycle.current_phase == TrainingPhase.MATCHING:
                        train_loader.dataset.reset_targets()
                        val_loader.dataset.reset_targets()
                        cycle_counter += 1
                        if cycle_counter >= self.config['cycles']:
                            print(f"Completed {self.config['cycles']} cycles at epoch {epoch}")
                            break
                    training_cycle.switch_to_next_phase()

            # Get final validation metrics
            training_cycle.current_phase = TrainingPhase.SHORTCUT
            model_module.lift = False
            final_val_metrics = self._evaluate(self.model, val_loader, training_cycle)
            
            fold_metrics[fold] = final_val_metrics['acc']

        return self.model, fold_metrics
    
    def train_without_val(self, X_train, y_train, X_test, y_test, device='cpu'):
        """Train model on full training set without validation, evaluate on test set.
        Specifically designed for CIFAR where test set is fixed."""
        
        self.device = device
        self.metrics_tracker = MetricsTracker(n_folds=1)
        
        # Create data loaders
        train_loader = self._create_loader(X_train, y_train, self.config['batch_size'], device)
        test_loader = self._create_loader(X_test, y_test, self.config['batch_size'], device, shuffle=False)
        
        # Initialize model and training components
        self.model = self._initialize_model(device)
        training_cycle = TrainingCycle(self.model, self.config)
        optimizer, scheduler = self._initialize_optimizer(self.model)
        
        cycle_counter = 0
        force_final_shortcut = False
        fold = 0  # Single fold
        
        for epoch in tqdm(range(self.config['epochs']), desc="Training"):
            # Training phase
            train_metrics = self._train_epoch(
                self.model, train_loader, optimizer, training_cycle
            )
            
            # Test evaluation
            test_metrics = self._evaluate(self.model, test_loader, training_cycle)
            
            if self.scheduler is not None:
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_metrics['loss'])
                else:
                    self.scheduler.step()

            model_module = self.get_model_module()
            
            # Update metrics
            self.metrics_tracker.update_metrics(
                fold, epoch,
                {
                    'epoch': epoch,
                    'phases': training_cycle.current_phase.value,
                    'train_loss': train_metrics['loss'],
                    'train_ce_loss': train_metrics['ce_loss'],
                    'train_match_loss': train_metrics['match_loss'],
                    'train_acc': train_metrics['acc'],
                    'test_loss': test_metrics['loss'],
                    'test_ce_loss': test_metrics['ce_loss'],
                    'test_match_loss': test_metrics['match_loss'],
                    'test_acc': test_metrics['acc'],
                    'test_shortcut_acc': test_metrics['shortcut_acc'],
                    'test_lifted_acc': test_metrics['lifted_acc']
                },
                {
                    'epoch': epoch,
                    'model_lift': model_module.lift,
                    'patience_counter': training_cycle.patience_counter,
                    'cycle_counter': cycle_counter
                }
            )
            
            if training_cycle.should_switch_phase(train_metrics['loss'], train_test=True):
                print(f"Switching phase at epoch {epoch}")
                print(f"\t Train loss at the end of {training_cycle.current_phase}: {train_metrics['loss']:.4f}")
                print(f"\t Train accuracy at the end of {training_cycle.current_phase}: {train_metrics['acc']:.4f}")
                # print(f"\t Val loss at the end of {training_cycle.current_phase}: {val_metrics['loss']:.4f}")
                # print(f"\t Val accuracy at the end of {training_cycle.current_phase}: {val_metrics['acc']:.4f}")
                print(f"\t Test loss at the end of {training_cycle.current_phase}: {test_metrics['loss']:.4f}")
                print(f"\t Test accuracy at the end of {training_cycle.current_phase}: {test_metrics['acc']:.4f}")

                if training_cycle.current_phase == TrainingPhase.LIFTED:
                    train_loader.dataset.update_targets(model_module)
                    test_loader.dataset.update_targets(model_module)
                elif training_cycle.current_phase == TrainingPhase.MATCHING:
                    train_loader.dataset.reset_targets()
                    test_loader.dataset.reset_targets()
                    cycle_counter += 1
                    if cycle_counter >= self.config['cycles']:
                        print(f"Completed {self.config['cycles']} cycles at epoch {epoch}")
                        break
                training_cycle.switch_to_next_phase()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Get final test metrics
        training_cycle.current_phase = TrainingPhase.SHORTCUT
        model_module.lift = False
        final_test_metrics = self._evaluate(self.model, test_loader, training_cycle)

        return self.model, self.metrics_tracker.metrics
    
    def train_with_fixed_val(self, X_train, y_train, X_test, y_test, device='cpu', val_split=0.2):
        """Train model with a fixed validation split and test set.
        Uses 80-20 train-val split by default, with CIFAR's fixed test set."""
        
        self.device = device
        self.metrics_tracker = MetricsTracker(n_folds=1)
        
        # Create train-val split (80-20)
        val_size = int(len(X_train) * val_split)
        np.random.seed(self.config.get('seed', 42))
        indices = np.random.permutation(len(X_train))
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]
        
        
        X_train_final = X_train[train_indices]
        y_train_final = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # printing sizes of all datasets
        print("Trainer v7: Train, Val, Test sizes")
        print(f"X_train_final: {X_train_final.shape}, y_train_final: {y_train_final.shape}")
        # Create data loaders
        train_loader = self._create_loader(X_train_final, y_train_final, self.config['batch_size'], device)
        val_loader = self._create_loader(X_val, y_val, self.config['batch_size'], device, shuffle=False)
        test_loader = self._create_loader(X_test, y_test, self.config['batch_size'], device, shuffle=False)
        
        # Initialize model and training components
        self.model = self._initialize_model(device)
        training_cycle = TrainingCycle(self.model, self.config)
        optimizer, scheduler = self._initialize_optimizer(self.model)
        
        cycle_counter = 0
        fold = 0  # Single fold
        
        for epoch in tqdm(range(self.config['epochs']), desc="Training"):
            # Training phase
            train_metrics = self._train_epoch(
                self.model, train_loader, optimizer, training_cycle
            )
            
            # Evaluation phases
            val_metrics = self._evaluate(self.model, val_loader, training_cycle)
            test_metrics = self._evaluate(self.model, test_loader, training_cycle)
            
            if self.scheduler is not None:
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])  # Use validation loss for scheduler
                else:
                    self.scheduler.step()

            model_module = self.get_model_module()
            
            # Update metrics
            self.metrics_tracker.update_metrics(
                fold, epoch,
                {
                    'epoch': epoch,
                    'phases': training_cycle.current_phase.value,
                    'train_loss': train_metrics['loss'],
                    'train_ce_loss': train_metrics['ce_loss'],
                    'train_match_loss': train_metrics['match_loss'],
                    'train_acc': train_metrics['acc'],
                    'val_loss': val_metrics['loss'],
                    'val_ce_loss': val_metrics['ce_loss'],
                    'val_match_loss': val_metrics['match_loss'],
                    'val_acc': val_metrics['acc'],
                    'val_shortcut_acc': val_metrics['shortcut_acc'],
                    'val_lifted_acc': val_metrics['lifted_acc'],
                    'test_loss': test_metrics['loss'],
                    'test_ce_loss': test_metrics['ce_loss'],
                    'test_match_loss': test_metrics['match_loss'],
                    'test_acc': test_metrics['acc'],
                    'test_shortcut_acc': test_metrics['shortcut_acc'],
                    'test_lifted_acc': test_metrics['lifted_acc']
                },
                {
                    'epoch': epoch,
                    'model_lift': model_module.lift,
                    'patience_counter': training_cycle.patience_counter,
                    'cycle_counter': cycle_counter
                }
            )
                
            # Check for phase transition using validation loss
            if training_cycle.should_switch_phase(val_metrics['loss']):
                print(f"Switching phase at epoch {epoch}")
                print(f"\t Train loss at the end of {training_cycle.current_phase}: {train_metrics['loss']:.4f}")
                print(f"\t Train accuracy at the end of {training_cycle.current_phase}: {train_metrics['acc']:.4f}")
                print(f"\t Val loss at the end of {training_cycle.current_phase}: {val_metrics['loss']:.4f}")
                print(f"\t Val accuracy at the end of {training_cycle.current_phase}: {val_metrics['acc']:.4f}")
                print(f"\t Test loss at the end of {training_cycle.current_phase}: {test_metrics['loss']:.4f}")
                print(f"\t Test accuracy at the end of {training_cycle.current_phase}: {test_metrics['acc']:.4f}")
                
                if training_cycle.current_phase == TrainingPhase.LIFTED:
                    train_loader.dataset.update_targets(model_module)
                    test_loader.dataset.update_targets(model_module)
                    val_loader.dataset.update_targets(model_module)
                elif training_cycle.current_phase == TrainingPhase.MATCHING:
                    train_loader.dataset.reset_targets()
                    test_loader.dataset.reset_targets()
                    val_loader.dataset.reset_targets()
                    cycle_counter += 1
                    if cycle_counter >= self.config['cycles']:
                        print(f"Completed {self.config['cycles']} cycles at epoch {epoch}")
                        break
                training_cycle.switch_to_next_phase()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Get final metrics
        training_cycle.current_phase = TrainingPhase.SHORTCUT
        model_module.lift = False
        final_val_metrics = self._evaluate(self.model, val_loader, training_cycle)
        final_test_metrics = self._evaluate(self.model, test_loader, training_cycle)
        
        self.metrics_tracker.update_metrics(
            fold, epoch + 1,
            {
                'final_val_loss': final_val_metrics['loss'],
                'final_val_acc': final_val_metrics['acc'],
                'final_test_loss': final_test_metrics['loss'],
                'final_test_acc': final_test_metrics['acc']
            },
            {}
        )

        return self.model, self.metrics_tracker.metrics

    def save_results(self, model, save_path):
        """Save model, metrics, and configuration."""
        if not save_path:
            return

        # check save path to ensure absolute path and BASE_DIR
        if not os.path.isabs(save_path):
            save_path = os.path.join(BASE_DIR, save_path)
            
        save_dir = os.path.join(save_path, self.config['model_name'])
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics
        self._save_metrics(save_dir)
        
        # Save configuration
        self._save_config(save_dir)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, f"{self.config['model_name']}.pth"))
        
        # Save model summary
        self._save_model_summary(model, save_dir)