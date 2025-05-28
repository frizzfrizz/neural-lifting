import os
import random
import math

def get_unique_results_directory(results_directory):
    if results_directory[-1] == '/':
        base_directory = results_directory[:results_directory[:-1].rfind('/')+1]
    else:
        base_directory = results_directory[:results_directory.rfind('/')+1]
    if not os.path.exists(results_directory):
        return results_directory
    else:
        # find the last run number
        run_numbers = [int(f.split('_')[-1]) for f in os.listdir(base_directory) if f.startswith('run_')]
        if len(run_numbers) == 0:
            return base_directory + 'run_1/'
        else:
            return base_directory + 'run_' + str(max(run_numbers)+1) + '/'
        
def get_unique_dataset_directory(dataset_directory):
    if dataset_directory[-1] == '/':
        base_directory = dataset_directory[:dataset_directory[:-1].rfind('/')+1]
    else:
        base_directory = dataset_directory[:dataset_directory.rfind('/')+1]
    if not os.path.exists(dataset_directory):
        return dataset_directory
    else:
        # find the last run number
        run_numbers = [int(f.split('_')[-1]) for f in os.listdir(base_directory) if f.startswith('run_')]
        if len(run_numbers) == 0:
            return base_directory + 'run_1/'
        else:
            return base_directory + 'run_' + str(max(run_numbers)+1) + '/'
        
def generate_symbolic_schedule(schedule_type, early_freq, total_epochs):
    """Generate symbolic schedule with last 5 epochs using shortcut"""    
    if schedule_type == 'frequent_early':
        # More symbolic epochs early, then reducing
        if early_freq is None:
            early_freq = random.randint(1, 5)
        schedule = list(range(0, total_epochs - 5, early_freq))
    elif schedule_type == 'periodic':
        # Regular intervals except last 5 epochs
        period = random.randint(1, 5)
        schedule = list(range(0, total_epochs - 5, period))
    else:  # decreasing_frequency
        # Start with high frequency, then decrease
        schedule = []
        current = 0
        while current < total_epochs - 5:
            schedule.append(current)
            increment = random.randint(1, 5)
            current += increment
    
    return sorted(list(set(schedule)))  # Ensure unique and sorted

def create_lambda2_scheduler(schedule_type='linear_warmup', **kwargs):
    """
    Create lambda2 scheduler with specified behavior.
    
    Args:
        schedule_type: Type of schedule ('linear_warmup', 'cosine', 'plateau')
        **kwargs: Schedule-specific parameters
    """
    if schedule_type == 'linear_warmup':
        warmup_epochs = kwargs.get('warmup_epochs', 10)
        max_value = kwargs.get('max_value', 0.1)
        min_value = kwargs.get('min_value', 0.001)
        
        def scheduler(epoch):
            if epoch < warmup_epochs:
                return min_value + (max_value - min_value) * (epoch / warmup_epochs)
            return max_value
            
    elif schedule_type == 'cosine':
        total_epochs = kwargs.get('total_epochs', 100)
        max_value = kwargs.get('max_value', 0.1)
        min_value = kwargs.get('min_value', 0.001)
        
        def scheduler(epoch):
            progress = epoch / total_epochs
            return min_value + 0.5 * (max_value - min_value) * (1 + math.cos(math.pi * progress))
            
    elif schedule_type == 'plateau':
        warmup_epochs = kwargs.get('warmup_epochs', 10)
        plateau_epochs = kwargs.get('plateau_epochs', 50)
        cooldown_epochs = kwargs.get('cooldown_epochs', 40)
        max_value = kwargs.get('max_value', 0.1)
        min_value = kwargs.get('min_value', 0.001)
        
        def scheduler(epoch):
            if epoch < warmup_epochs:  # Warmup phase
                return min_value + (max_value - min_value) * (epoch / warmup_epochs)
            elif epoch < warmup_epochs + plateau_epochs:  # Plateau phase
                return max_value
            else:  # Cooldown phase
                remaining = max(0, cooldown_epochs - (epoch - (warmup_epochs + plateau_epochs)))
                return min_value + (max_value - min_value) * (remaining / cooldown_epochs)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
        
    return scheduler


def create_lambda_match_scheduler(schedule_type='linear_warmup', **kwargs):
    """
    Create lambda2 scheduler with specified behavior.
    
    Args:
        schedule_type: Type of schedule ('linear_warmup', 'cosine', 'plateau')
        **kwargs: Schedule-specific parameters
    """
    if schedule_type == 'linear_warmup':
        warmup_steps = kwargs.get('warmup_steps', 1000)
        max_value = kwargs.get('max_value', 0.1)
        min_value = kwargs.get('min_value', 0.001)
        
        def scheduler(step):
            if step < warmup_steps:
                return min_value + (max_value - min_value) * (step / warmup_steps)
            return max_value
            
    elif schedule_type == 'cosine':
        total_steps = kwargs.get('total_steps', 10000)
        max_value = kwargs.get('max_value', 0.1)
        min_value = kwargs.get('min_value', 0.001)
        
        def scheduler(step):
            progress = step / total_steps
            return min_value + 0.5 * (max_value - min_value) * (1 + math.cos(math.pi * progress))
            
    elif schedule_type == 'plateau':
        warmup_steps = kwargs.get('warmup_steps', 1000)
        plateau_steps = kwargs.get('plateau_steps', 5000)
        cooldown_steps = kwargs.get('cooldown_steps', 4000)
        max_value = kwargs.get('max_value', 0.1)
        min_value = kwargs.get('min_value', 0.001)
        
        def scheduler(step):
            if step < warmup_steps:  # Warmup phase
                return min_value + (max_value - min_value) * (step / warmup_steps)
            elif step < warmup_steps + plateau_steps:  # Plateau phase
                return max_value
            else:  # Cooldown phase
                remaining = max(0, cooldown_steps - (step - (warmup_steps + plateau_steps)))
                return min_value + (max_value - min_value) * (remaining / cooldown_steps)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
        
    return scheduler