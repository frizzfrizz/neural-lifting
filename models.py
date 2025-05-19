# models.py
import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
from visualisers import plot_metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import copy
from aux_functions import create_lambda_match_scheduler
from collections import defaultdict
from dir_names import BASE_DIR
import math

def train_test_split(X, y, test_size=10000, random_state=None):
    """Split data into train and test sets for the CIFAR-10 dataset."""
    total_samples = len(X)
    train_val_indices = slice(0, total_samples - test_size)
    test_indices = slice(total_samples - test_size, total_samples)

    X_trainval, X_test = X[train_val_indices], X[test_indices]
    y_trainval, y_test = y[train_val_indices], y[test_indices]

    return X_trainval, X_test, y_trainval, y_test

## Baseline models
class BaselineModel(nn.Module, ABC):
    """Abstract base class for baseline models"""
    def __init__(self, config=None):
        super(BaselineModel, self).__init__()
        self.config = self._get_default_config() if config is None else config
        self._build_model()
    
    @abstractmethod
    def _get_default_config(self):
        """Return default configuration dictionary"""
        pass
    
    @abstractmethod
    def _build_model(self):
        """Build model architecture"""
        pass

class ResidualBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 dropout_rate=0.0, use_batch_norm=True, activation='relu'):
        super(ResidualBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Downsample if needed
        self.downsample = downsample
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class ResNet18(BaselineModel):
    """Customizable ResNet18 implementation"""
    def _get_default_config(self):
        return {
            'in_channels': 3,
            'num_classes': 10,
            'base_channels': 64,
            'channel_multiplier': 1.0,
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'activation': 'relu',
            'initial_kernel': 7,
            'initial_stride': 2,
            'initial_pooling': True,
            'pooling_type': 'max',
            'block_config': [2, 2, 2, 2],  # number of blocks in each layer
            'weight_init': 'kaiming'
        }

    def _build_model(self):
        # Initial layer
        c0 = int(self.config['base_channels'] * self.config['channel_multiplier'])
        self.in_planes = c0
        
        self.conv1 = nn.Conv2d(self.config['in_channels'], c0,
                              kernel_size=self.config['initial_kernel'],
                              stride=self.config['initial_stride'],
                              padding=self.config['initial_kernel']//2,
                              bias=False)
        
        self.bn1 = nn.BatchNorm2d(c0) if self.config['use_batch_norm'] else nn.Identity()
        
        if self.config['activation'] == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.config['activation'] == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        if self.config['pooling_type'] == 'max':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.config['initial_pooling'] else nn.Identity()
        elif self.config['pooling_type'] == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if self.config['initial_pooling'] else nn.Identity()
        
        # Residual layers
        self.layer1 = self._make_layer(c0, self.config['block_config'][0])
        self.layer2 = self._make_layer(c0*2, self.config['block_config'][1], stride=2)
        self.layer3 = self._make_layer(c0*4, self.config['block_config'][2], stride=2)
        self.layer4 = self._make_layer(c0*8, self.config['block_config'][3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        self.fc = nn.Linear(c0*8, self.config['num_classes'])
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels) if self.config['use_batch_norm'] else nn.Identity()
            )

        layers = []
        layers.append(ResidualBlock(self.in_planes, channels, stride, downsample,
                               self.config['dropout_rate'],
                               self.config['use_batch_norm'],
                               self.config['activation']))
        
        self.in_planes = channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_planes, channels,
                                   dropout_rate=self.config['dropout_rate'],
                                   use_batch_norm=self.config['use_batch_norm'],
                                   activation=self.config['activation']))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.config['weight_init'] == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.config['weight_init'] == 'xavier':
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
        # return F.softmax(x, dim=1)

# Using the model from https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation/blob/master/resnet_cifar.py
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet_CIFAR(nn.Module):
    """Using the model from https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation/blob/master/resnet_cifar.py"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvNet_CIFAR(BaselineModel):
    """Configurable ConvNet for CIFAR-10"""
    
    # Predefined architectures
    ARCHITECTURES = {
        '2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
        '4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
        '6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 
              'Conv64', 'Conv64', 'MaxPool', 'FC10'],
        '8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 
              'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool', 
              'FC64', 'FC10'],
        '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
               'Conv128', 'Conv128', 'MaxPool', 'Conv256', 'Conv256', 'Conv256',
               'Conv256', 'MaxPool', 'FC128', 'FC10']
    }

    def _get_default_config(self):
        return {
            'in_channels': 3,
            'num_classes': 10,
            'architecture': '2',
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'activation': 'relu',
            'weight_init': 'xavier'
        }

    def _build_model(self):
        layers = self.ARCHITECTURES[self.config['architecture']]
        
        # Track dimensions
        h, w = 32, 32
        in_channels = self.config['in_channels']
        
        # Build layers
        conv_layers = []
        fc_layers = []
        
        for layer in layers:
            if layer.startswith('Conv'):
                out_channels = int(layer[4:])
                conv_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels) if self.config['use_batch_norm'] else nn.Identity(),
                    nn.ReLU(inplace=True) if self.config['activation'] == 'relu' else nn.LeakyReLU(inplace=True),
                    nn.Dropout(self.config['dropout_rate'])
                ])
                in_channels = out_channels
                
            elif layer.startswith('MaxPool'):
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                h, w = h//2, w//2
                
            elif layer.startswith('FC'):
                out_features = int(layer[2:])
                if not fc_layers:  # First FC layer
                    in_features = in_channels * h * w
                else:
                    in_features = fc_layers[-1].out_features
                
                fc_layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True) if layer != layers[-1] else nn.Identity()
                ])
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Sequential(*fc_layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.config['weight_init'] == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.config['weight_init'] == 'xavier':
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class LeNet(BaselineModel):
    """Customizable LeNet implementation"""
    def __init__(self, grayscale=False, input_size=32, **kwargs):
        self.grayscale = grayscale
        self.input_size = input_size
        super().__init__(**kwargs)

    def _get_default_config(self):
        in_channels = 1 if self.grayscale else 3
        base_conv = [6, 16]  # Base conv channels
        base_fc = [120, 84]  # Base fc sizes
        
        return {
            'in_channels': in_channels,
            'num_classes': 10,
            'conv_channels': [c * in_channels for c in base_conv],
            'conv_kernel_sizes': [5, 5],
            'conv_strides': [1, 1],
            'conv_paddings': [2, 2],
            'fc_layers': [c * in_channels for c in base_fc],
            'pooling_size': 2,
            'pooling_type': 'max',
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'activation': 'relu',
            'weight_init': 'kaiming'
        }

    def _build_model(self):
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.config['in_channels']
        
        for i, out_channels in enumerate(self.config['conv_channels']):
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=self.config['conv_kernel_sizes'][i],
                         stride=self.config['conv_strides'][i],
                         padding=self.config['conv_paddings'][i]),
                nn.BatchNorm2d(out_channels) if self.config['use_batch_norm'] else nn.Identity(),
                nn.ReLU(inplace=True) if self.config['activation'] == 'relu' else nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(self.config['pooling_size']) if self.config['pooling_type'] == 'max' 
                else nn.AvgPool2d(self.config['pooling_size']),
                nn.Dropout(self.config['dropout_rate'])
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels

        # Calculate size for first FC layer
        with torch.no_grad():
            x = torch.zeros(1, self.config['in_channels'], self.input_size, self.input_size)
            for conv_block in self.conv_layers:
                x = conv_block(x)
            self.fc_input_size = x.view(1, -1).size(1)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_sizes = [self.fc_input_size] + self.config['fc_layers']
        
        for i in range(len(fc_sizes)-1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_sizes[i], fc_sizes[i+1]),
                nn.ReLU(inplace=True) if self.config['activation'] == 'relu' else nn.LeakyReLU(inplace=True),
                nn.Dropout(self.config['dropout_rate'])
            ))

        # Final classification layer
        self.classifier = nn.Linear(fc_sizes[-1], self.config['num_classes'])
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self, seed=42):
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if self.config['weight_init'] == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.config['weight_init'] == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Convolutional layers
        for conv_block in self.conv_layers:
            x = conv_block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc_block in self.fc_layers:
            x = fc_block(x)
        
        # Classification layer
        x = self.classifier(x)
        return x
        # return F.softmax(x, dim=1)

## Auxiliary models
# MLP class
class MLP(nn.Module):
    def __init__(self, num_layers: int, layer_sizes: list, activations: list, 
                 seed: int=17)->None:
        super(MLP, self).__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        self.activations = activations

        for i in range(num_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(self.get_activation(activations[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_activation(self, activation: str)->nn.Module:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        else:
            return nn.ReLU()

# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_shape, conv_params, fc_layers, latent_dim, pooling = True, dropout = True, pooling_size = 2, dropout_rate = 0.1):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        conv_filters, conv_kernels, conv_strides, paddings = conv_params

        layers = []
        in_channels = input_shape[0]
        for out_channels, kernel_size, stride, padding in zip(conv_filters, conv_kernels, conv_strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
            if pooling:
                layers.append(nn.MaxPool2d(pooling_size))
            if dropout:
                layers.append(nn.Dropout(dropout_rate))

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        conv_output_shape = self._get_conv_output_shape(input_shape)
        fc_input_dim = conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2]

        fc_layers = [fc_input_dim] + fc_layers
        fc_layers.append(latent_dim)

        fc_layers_list = []
        for i in range(len(fc_layers) - 1):
            fc_layers_list.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i <= len(fc_layers) - 2:
                # print(f'here in fc_layers activation{fc_layers[i], fc_layers[i + 1]}')
                fc_layers_list.append(nn.ReLU())

        self.fc = nn.Sequential(*fc_layers_list)

    def _get_conv_output_shape(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)
        return x.shape[1:]

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ResidualSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        
        # Find input and output dimensions from the first and last Linear layers
        input_dim = None
        output_dim = None
        for layer in layers:
            if isinstance(layer, nn.Linear):
                if input_dim is None:
                    input_dim = layer.in_features
                output_dim = layer.out_features
        
        if input_dim is None or output_dim is None:
            raise ValueError("ResidualSequential must contain at least one Linear layer")
        
        # If dimensions don't match, add adjustment layer at the end
        layer_list = list(layers)
        if input_dim != output_dim:
            layer_list.append(nn.Linear(output_dim, input_dim))
            layer_list.append(nn.ReLU())
            
        self.layers = nn.Sequential(*layer_list)
        
    def forward(self, x):
        # return self.layers(x) + x
        return F.relu(self.layers(x) + x)
    
## Proposed Lifting Models
class NeuralLiftNet(nn.Module, ABC):
    def __init__(self, trunk = None, head = None, lifter = None, config = None):
        super(NeuralLiftNet, self).__init__()
        self.config = config if config else self._get_default_config()
        self.trunk = trunk if trunk else self._create_trunk()
        self.head = head if head else self._create_head()
        self.lifter = lifter if lifter else self._create_lifter() # see this
        self.lift = False

    @abstractmethod
    def _get_default_config(self):
        pass

    @abstractmethod
    def _create_trunk(self):
        pass

    @abstractmethod
    def _create_head(self):
        pass

    @abstractmethod
    def _create_lifter(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def enable_lift(self):
        self.lift = True

    def disable_lift(self):
        self.lift = False

class RFLiftNet(NeuralLiftNet):
    def __init__(self, trunk = None, head = None, lifter = None, config = None):
        super(RFLiftNet, self).__init__(None, None, None, config)
        # self.trunk = trunk if trunk else self._create_trunk()
        # self.head = None
        # self.lifter = lifter if lifter else self._create_lifter() # see this
        # self.lift = False


    def _get_default_config(self):
        return {
            'trunk_config': {
                'in_channels': 3,
                'conv_channels': [6, 16],
                'conv_kernels': [3, 3],
                'strides': [1, 1],
                'paddings': [1, 1],
                'fc_layers': [60],
                'input_shape': (3, 32, 32),
                'latent_dim': 84,
                'activations': ['relu', 'relu'],
                'dropout': True,
                'pooling': True,
                'pooling_size': 2,
                'dropout_rate': 0.1
            },
            'lifter_config': {
                'max_depth': 10,
                'rand_state': 0,
            }
        }
    
    def _create_trunk(self):
        trunk_config = self.config['trunk_config'] if 'trunk_config' in self.config else self.config['params']['trunk_config']
        return Encoder(
            trunk_config['input_shape'],
            [trunk_config['conv_channels'], trunk_config['conv_kernels'], trunk_config['strides'], trunk_config['paddings']],
            trunk_config['fc_layers'],
            trunk_config['latent_dim'],
            pooling=trunk_config['pooling'],
            dropout=trunk_config['dropout'],
            pooling_size=trunk_config['pooling_size'],
            dropout_rate=trunk_config['dropout_rate']
        )

    def _create_head(self):
        return None

    def _create_lifter(self):
        return RandomForestRegressor(max_depth=self.config['lifter_config']['max_depth'], 
                                     random_state=self.config['lifter_config']['rand_state'])

    def forward(self, x):
        z = self.trunk(x)
        if self.lift:
            # print('lifting')
            y = self.lifter(z) + z
            return y, z
        
        # print('not lifting')
        y = z
        return y, z


    def enable_lift(self):
        self.lift = True

    def disable_lift(self):
        self.lift = False

class NearNeiLiftNet(NeuralLiftNet):
    def __init__(self, trunk = None, head = None, lifter = None, config = None):
        super(NeuralLiftNet, self).__init__()
        self.config = self._get_default_config() if config is None else config
        self.trunk = trunk if trunk else self._create_trunk()
        # self.head = head if head else self._create_head()
        self.lifter = lifter if lifter else self._create_lifter() # see this
        self.lift = False

    def _get_default_config(self):
        return {
            'trunk_config': {
                'in_channels': 3,
                'conv_channels': [6, 16],
                'conv_kernels': [3, 3],
                'strides': [1, 1],
                'paddings': [1, 1],
                'fc_layers': [60],
                'input_shape': (3, 32, 32),
                'latent_dim': 84,
                'activations': ['relu', 'relu'],
                'dropout': True,
                'pooling': True,
                'pooling_size': 2,
                'dropout_rate': 0.1
            },
            'lifter_config': {
                'n_neighbors': 5,
                'weights': 'uniform',
            }
        }
    
    def _create_trunk(self):
        trunk_config = self.config['trunk_config'] if 'trunk_config' in self.config else self.config['params']['trunk_config']
        return Encoder(
            trunk_config['input_shape'],
            [trunk_config['conv_channels'], trunk_config['conv_kernels'], trunk_config['strides'], trunk_config['paddings']],
            trunk_config['fc_layers'],
            trunk_config['latent_dim'],
            pooling=trunk_config['pooling'],
            dropout=trunk_config['dropout'],
            pooling_size=trunk_config['pooling_size'],
            dropout_rate=trunk_config['dropout_rate']
        )

    def _create_head(self):
        pass

    def _create_lifter(self):
        return KNeighborsRegressor(n_neighbors=self.config['lifter_config']['n_neighbors'], 
                                     weights=self.config['lifter_config']['weights'])

    def forward(self, x):
        if self.lift:
            # print('lifting')
            z = self.trunk(x)
            y = self.lifter(z) + z
            return y
        
        # print('not lifting')
        y = self.trunk(x)
        return y


    def enable_lift(self):
        self.lift = True

    def disable_lift(self):
        self.lift = False

class LeNetUnlifted(NeuralLiftNet):
    def __init__(self, trunk = None, head = None, lifter = None, 
                 config = None):
        self.config = self._get_default_config() if config is None else config
        super(LeNetUnlifted, self).__init__(
            trunk, head, lifter
        )
    
    def _get_default_config(self):
        return {
            'trunk_config': {
                'in_channels': 3,
                'conv_channels': [18, 48],
                'conv_kernels': [3, 3],
                'strides': [1, 1],
                'paddings': [2, 2],
                'fc_layers': [360],
                'input_shape': (3, 32, 32),
                'latent_dim': 252,
                'activations': ['relu', 'relu'],
                'dropout': True,
                'pooling': True,
                'pooling_size': 2,
                'dropout_rate': 0.1
            },
            'head_config': {
                'fc_layers': [],
                'activations': [],
                'final_activation': ['softmax'],
                'num_classes': 10,
                'dropout': False,
                'pooling': False
            }
        }
    def _create_trunk(self):
        trunk_config = self.config['trunk_config'] if 'trunk_config' in self.config else self.config['params']['trunk_config']
        return Encoder(
            trunk_config['input_shape'],
            [trunk_config['conv_channels'], trunk_config['conv_kernels'], trunk_config['strides'], trunk_config['paddings']],
            trunk_config['fc_layers'],
            trunk_config['latent_dim'],
            pooling=trunk_config['pooling'],
            dropout=trunk_config['dropout'],
            pooling_size=trunk_config['pooling_size'],
            dropout_rate=trunk_config['dropout_rate']
        )

    def _create_head(self):
        head_config = self.config['head_config'] if 'head_config' in self.config else self.config['params']['head_config']
        input_dim = self.config['trunk_config']['latent_dim'] if 'trunk_config' in self.config else self.config['params']['trunk_config']['latent_dim']
        fc_layers = [input_dim] + head_config['fc_layers'] + [head_config['num_classes']]
        layers = []
        for i in range(len(fc_layers) - 1):
            layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:
                layers.append(self.get_activation(head_config['activations'][i]))
        # layers.append(self.get_activation(head_config['final_activation'][0]))
        return nn.Sequential(*layers)

    def _create_lifter(self):
        return nn.Identity()
    
    def _initialize_modules(self, seed = 42, init_type = 'xavier'):
        print(f"Initializing modules with seed={seed}, init_type={init_type}")
        torch.manual_seed(seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.trunk.apply(init_weights)
        self.head.apply(init_weights)

    def forward(self, x):
        z = self.trunk(x)
        y = self.head(z)
        return y
    
    def get_activation(self, activation: str)->nn.Module:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        else:
            return nn.ReLU()


class LeNetLifted(NeuralLiftNet):
    def __init__(self, trunk = None, head = None, lifter = None, 
                 config = None):
        self.config = self._get_default_config() if config is None else config
        super(LeNetLifted, self).__init__(
            trunk, head, lifter
        )
    
    def _get_default_config(self):
        return {
            'trunk_config': {
                'in_channels': 3,
                'conv_channels': [6, 16],
                'conv_kernels': [3, 3],
                'strides': [1, 1],
                'paddings': [1, 1],
                'fc_layers': [60],
                'input_shape': (1, 64, 64),
                'latent_dim': 84,
                'activations': ['relu', 'relu'],
                'dropout': True,
                'pooling': True,
                'pooling_size': 2,
                'dropout_rate': 0.1
            },
            'head_config': {
                'fc_layers': [],
                'activations': [],
                'final_activation': ['softmax'],
                'num_classes': 10,
                'dropout': False,
                'pooling': False,
                'is_identity': False
            },
            'lifter_config': {
                'lifter_type': 'Residual',
                'fc_layers': [500],
                'activations': ['relu'],
                'init_type': 'kaiming',
                'sparsity': 0.9,
                'scale': 1.0,
                'is_final_layer': False
            }
        }
    def _create_trunk(self):
        trunk_config = self.config['trunk_config'] if 'trunk_config' in self.config else self.config['params']['trunk_config']
        return Encoder(
            trunk_config['input_shape'],
            [trunk_config['conv_channels'], trunk_config['conv_kernels'], trunk_config['strides'], trunk_config['paddings']],
            trunk_config['fc_layers'],
            trunk_config['latent_dim'],
            pooling=trunk_config['pooling'],
            dropout=trunk_config['dropout'],
            pooling_size=trunk_config['pooling_size'],
            dropout_rate=trunk_config['dropout_rate']
        )

    def _create_head(self):
        head_config = self.config['head_config'] if 'head_config' in self.config else self.config['params']['head_config']
        input_dim = self.config['trunk_config']['latent_dim'] if 'trunk_config' in self.config else self.config['params']['trunk_config']['latent_dim']
        fc_layers = [input_dim] + head_config['fc_layers'] + [head_config['num_classes']]
        layers = []
        for i in range(len(fc_layers) - 1):
            layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:
                layers.append(self.get_activation(head_config['activations'][i]))
        # layers.append(self.get_activation(head_config['final_activation'][0]))
        return nn.Sequential(*layers)

    def _create_lifter(self):
        lifter_config = self.config['lifter_config'] if 'lifter_config' in self.config else self.config['params']['lifter_config']
        input_dim = self.config['trunk_config']['latent_dim'] if 'trunk_config' in self.config else self.config['params']['trunk_config']['latent_dim']
        
        if lifter_config['lifter_type'] == 'MLP':
            lifting_layers = [input_dim] + lifter_config['fc_layers'] + [input_dim]
            layers = []
            for i in range(len(lifting_layers) - 1):
                layers.append(nn.Linear(lifting_layers[i], lifting_layers[i + 1]))
                if i < len(lifting_layers) - 1:
                    layers.append(self.get_activation(lifter_config['activations'][i]))
            return nn.Sequential(*layers)
        
        elif lifter_config['lifter_type'] == 'Residual':
            # Create layers that go through all specified dimensions and back to input_dim
            layers = []
            fc_dims = [input_dim] + lifter_config['fc_layers'] + [input_dim]
            
            # Create layers through all dimensions
            for i in range(len(fc_dims) - 1):
                layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))
                if i < len(fc_dims) - 2:  # Don't add ReLU after the last linear layer
                    layers.append(nn.ReLU())
            
            return ResidualSequential(*layers)
        
        else:
            raise ValueError(f"Unknown lifter type: {lifter_config['lifter_type']}")
    
    def _initialize_modules(self, seed = 42, init_type = 'xavier'):
        print(f"Initializing modules with seed={seed}, init_type={init_type}")
        torch.manual_seed(seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.trunk.apply(init_weights)
        self.head.apply(init_weights)
        self.lifter.apply(init_weights)

    def _initialize_lifter(self, init_type='kaiming', sparsity=None, scale=None):
        """
        Initialize lifter weights with optional sparsity.
        Args:
            init_type: str, base initialization ('kaiming', 'xavier', 'normal', 'zeros')
            sparsity: float or None, if not None, fraction of weights to be set to zero
            scale: float, scaling factor for the weights after initialization
        """
        if sparsity is not None and not (0 <= sparsity <= 1):
            raise ValueError("sparsity must be between 0 and 1")
        print(f"Initializing lifter with {init_type} initialization, sparsity={sparsity}, scale={scale}")
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Step 1: Base initialization
                # print(f"\tInitializing layer with init_type={init_type}")
                if init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=1.0)
                elif init_type == 'normal':
                    init.normal_(m.weight, mean=0, std=0.01)
                elif init_type == 'zeros':
                    init.zeros_(m.weight)
                else:
                    raise ValueError(f"Unknown initialization type: {init_type}")
                
                # Apply scaling
                if scale is not None and scale is not False:
                    # print(f"\t\tScaling layer by {scale}")
                    m.weight.data *= scale
                
                # Step 2: Apply sparsity if requested
                if sparsity is not None:
                    # print(f"\t\tApplying sparsity of {sparsity}")
                    # Create binary mask
                    mask = (torch.rand_like(m.weight) > sparsity)
                    # Apply mask and rescale to maintain variance
                    if mask.sum() > 0:  # Prevent division by zero
                        m.weight.data *= mask / math.sqrt(1 - sparsity)
                
                # Initialize bias to zero if it exists
                if m.bias is not None:
                    init.zeros_(m.bias)
        
        self.lifter.apply(init_weights)
    
    def get_lifter_stats(self):
        """
        Calculate statistics of the lifter weights
        Returns:
            dict: statistics for each layer
        """
        stats = {}
        for name, module in self.lifter.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                total = weights.numel()
                zeros = (weights == 0).sum().item()
                stats[name] = {
                    'sparsity': zeros / total,
                    'mean': weights.mean().item(),
                    'std': weights.std().item(),
                    'min': weights.min().item(),
                    'max': weights.max().item(),
                    'total_weights': total,
                    'zero_weights': zeros
                }
        return stats


    def forward(self, x):
        # print('passing through forward')
        z = self.trunk(x)
        if self.lift:
            # print('lifting')
            z = self.lifter(z)
        # else:
        #     # print('not lifting')
        y = self.head(z)
        return y, z
    
    def get_activation(self, activation: str)->nn.Module:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        else:
            return nn.ReLU()