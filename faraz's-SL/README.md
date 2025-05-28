# neural-lifting
This repository contains code for implementing Neural Lifting on the CIFAR-10 dataset.

## Directory Setup

First, define the main directories based on user settings in `dir_names.py`

## Folder Structure

- `datasets/`: Contains data files
- `results/`: Directory to store the results for all the runs.

## Data Generation

To generate the dataset, run `gen_data.sh`

## Running Experiments
Two sets of experiments - those on the custom lifted model called `LeNetLifted`, and the baselines - `LeNet` and `ResNet18`

All model classes are defined in `models.py`.

### LeNetLifted
To run an experiment on the lifted model, use the following command:

```bash
./train_nl_model.sh
```
For the default arguments, 
    The experiment is initiated in the `main_nl.py` file.
    The results will be stored in `results/run_val`. Look for the most recent run.

    The main function initialises the ExperimentManager.

#### Experiment Manager
The ExperimentManager class is responsible for setting up and managing the entire experiment lifecycle, including device setup, configuration loading, results directory creation, and model training.

#### Training Function
Finally, the ModelTrainer class in `algo_2.py` handles the actual training of the LeNetLifted model.
The training is done in cycles, with each cycle comprising three phases defined in TrainingPhase: SHORTCUT, LIFTED, and MATCHING.

The TrainingCycle class manages the transition across different phases and cycles.

Training is done with K-fold cross validation.

All the training parameters are stored in the input config file. 

## Visualization

Plotting functions are available in `visualisers.py`.