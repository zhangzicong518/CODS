import torch
import torch.nn as nn
import json
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class CapabilityLabelProcessor:
    CAPABILITIES = [
        "Action Recognition",           
        "Attribute Recognition",        
        "Captioning",                   
        "Function Reasoning",           
        "Future Prediction",            
        "Identity Reasoning",           
        "Logical Reasoning",            
        "Object Localization",          
        "OCR",                         
        "Physical Property",           
        "Relation Reasoning",         
        "Spatial Reasoning",           
        "Structural Understanding"     
    ]
    
    def __init__(self):
        self.num_capabilities = len(self.CAPABILITIES)
        self.capability_to_idx = {cap: idx for idx, cap in enumerate(self.CAPABILITIES)}
        self.idx_to_capability = {idx: cap for idx, cap in enumerate(self.CAPABILITIES)}
        
    def process_categories(self, categories: List[str]) -> torch.Tensor:
        """
        Convert categories to binary label vector
        """
        label_vector = torch.zeros(self.num_capabilities, dtype=torch.float32)
        
        for category in categories:
            if category in self.capability_to_idx:
                idx = self.capability_to_idx[category]
                label_vector[idx] = 1.0
        
        return label_vector
    
    def get_capability_names(self, label_vector: torch.Tensor) -> List[str]:
        """
        Get capability names from label vector
        """
        capabilities = []
        for idx, value in enumerate(label_vector):
            if value > 0.5:  # threshold 0.5
                capabilities.append(self.CAPABILITIES[idx])
        return capabilities
    
    def print_capability_stats(self, all_labels: List[torch.Tensor]):
        if not all_labels:
            return
            
        # Stack all labels
        label_matrix = torch.stack(all_labels)  # [num_samples, num_capabilities]
        capability_counts = label_matrix.sum(dim=0)  # count for each capability
        
        print("\nCapability distribution statistics:")
        print("-" * 60)
        for idx, (capability, count) in enumerate(zip(self.CAPABILITIES, capability_counts)):
            percentage = count.item() / len(all_labels) * 100
            print(f"{idx:2d}. {capability:30s}: {count:5.0f} ({percentage:5.1f}%)")
        
        # Multi-label statistics
        labels_per_sample = label_matrix.sum(dim=1)
        print(f"\nAverage labels per sample: {labels_per_sample.mean().item():.2f}")
        print(f"Max labels: {labels_per_sample.max().item():.0f}")
        print(f"Min labels: {labels_per_sample.min().item():.0f}")


class EnhancedMultiModalDataset(Dataset):
    
    def __init__(self, 
                 gradients_file: str,
                 routing_weights_file: str,
                 meta_file: Optional[str] = None,
                 transform_gradients: bool = True,
                 transform_weights: bool = True,
                 max_samples: Optional[int] = None,
                 data_augmentation: bool = True,
                 noise_std: float = 0.01):
        self.transform_gradients = transform_gradients
        self.transform_weights = transform_weights
        self.data_augmentation = data_augmentation
        self.noise_std = noise_std
        self.training = True
        
        self.label_processor = CapabilityLabelProcessor()
        self.num_classes = self.label_processor.num_capabilities
        
        print("Loading data...")
        
        print(f"Loading gradients: {gradients_file}")
        self.gradients = torch.load(gradients_file, map_location='cpu')
        
        print(f"Loading routing weights: {routing_weights_file}")
        if routing_weights_file.endswith('.pkl'):
            with open(routing_weights_file, 'rb') as f:
                self.routing_data = pickle.load(f)
        else:
            self.routing_data = torch.load(routing_weights_file, map_location='cpu')
        
        # Process routing weights format
        self._process_routing_weights_format()
        
        # Load metadata and labels
        self.meta_data = None
        self.processed_labels = None
        if meta_file and os.path.exists(meta_file):
            print(f"Loading metadata: {meta_file}")
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.meta_data = json.load(f)
        else:
            print("No metadata file provided, using zero labels")
        
        # Determine sample count
        num_gradients = self.gradients.shape[0]
        num_routing = len(self.routing_weights)
        num_meta = len(self.meta_data) if self.meta_data else 0
        
        if self.meta_data:
            self.num_samples = min(num_gradients, num_routing, num_meta)
        else:
            self.num_samples = min(num_gradients, num_routing)
        
        # Limit sample count if specified
        if max_samples is not None and max_samples < self.num_samples:
            self.num_samples = max_samples
        
        print(f"Using {self.num_samples} samples")
        
        # Process labels after num_samples is determined
        if self.meta_data:
            self._process_labels()
        
        self.gradient_dim = self.gradients.shape[1]
        self.routing_weight_dim = self.routing_weights.shape[1]
        
        # Preprocess data
        if self.transform_gradients:
            self._preprocess_gradients()
        
        if self.transform_weights:
            self._preprocess_routing_weights()
        
        print("Data loading complete!")
    
    def _process_routing_weights_format(self):
        if isinstance(self.routing_data, list):
            if len(self.routing_data) > 0:
                first_item = self.routing_data[0]
                if isinstance(first_item, dict) and 'weight' in first_item:
                    # Format: [{'id': ..., 'weight': tensor}, ...]
                    weights = []
                    for item in self.routing_data:
                        weight = item['weight']
                        if weight.dim() == 2 and weight.shape[0] == 1:
                            weight = weight.squeeze(0)
                        weights.append(weight)
                    self.routing_weights = torch.stack(weights)
                elif torch.is_tensor(first_item):
                    # Format: [tensor, tensor, ...]
                    weights = []
                    for weight in self.routing_data:
                        if weight.dim() == 2 and weight.shape[0] == 1:
                            weight = weight.squeeze(0)
                        weights.append(weight)
                    self.routing_weights = torch.stack(weights)
                else:
                    raise ValueError(f"Unsupported routing weights list element type: {type(first_item)}")
            else:
                raise ValueError("Routing weights list is empty")
        
        elif torch.is_tensor(self.routing_data):
            # If single tensor, use directly
            self.routing_weights = self.routing_data
        
        else:
            raise ValueError(f"Unsupported routing weights data type: {type(self.routing_data)}")
    
    def _process_labels(self):
        print("Processing labels...")
        
        self.processed_labels = []
        label_stats = []
        
        for i, meta_item in enumerate(self.meta_data):
            if i >= self.num_samples:
                break
                
            # Get categories field
            categories = meta_item.get('categories', [])
            if not isinstance(categories, list):
                categories = [categories] if categories else []
            
            # Process labels
            label_vector = self.label_processor.process_categories(categories)
            self.processed_labels.append(label_vector)
            label_stats.append(label_vector)
        
        # Print label statistics
        if label_stats:
            self.label_processor.print_capability_stats(label_stats)
    
    def _preprocess_gradients(self):
        gradients_subset = self.gradients[:self.num_samples]
        
        zero_grad_mask = torch.all(gradients_subset == 0, dim=1)
        
        non_zero_gradients = gradients_subset[~zero_grad_mask]
        if len(non_zero_gradients) > 0:
            self.grad_mean = non_zero_gradients.mean(dim=0)
            self.grad_std = non_zero_gradients.std(dim=0) + 1e-8  # avoid division by zero
        else:
            self.grad_mean = torch.zeros(self.gradient_dim)
            self.grad_std = torch.ones(self.gradient_dim)
    
    def _preprocess_routing_weights(self):
        weights_subset = self.routing_weights[:self.num_samples]
        
        # Calculate statistics
        self.weight_mean = weights_subset.mean(dim=0)
        self.weight_std = weights_subset.std(dim=0) + 1e-8
    
    def _add_noise(self, tensor, std=None):
        """Add Gaussian noise for data augmentation"""
        if std is None:
            std = self.noise_std
        noise = torch.randn_like(tensor) * std
        return tensor + noise
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            gradients: Preprocessed gradients [gradient_dim]
            routing_weights: Preprocessed routing weights [routing_weight_dim]
            labels: Multi-label binary vector [num_capabilities]
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        gradient = self.gradients[idx].clone()
        
        if self.transform_gradients and hasattr(self, 'grad_mean'):
            gradient = (gradient - self.grad_mean) / self.grad_std
        
        # Data augmentation
        if self.data_augmentation and self.training:
            gradient = self._add_noise(gradient, std=self.noise_std)
        
        routing_weight = self.routing_weights[idx].clone()
        
        if self.transform_weights and hasattr(self, 'weight_mean'):
            routing_weight = (routing_weight - self.weight_mean) / self.weight_std
        
        if self.data_augmentation and self.training:
            routing_weight = self._add_noise(routing_weight, std=self.noise_std * 0.5)
        
        if self.processed_labels is not None and idx < len(self.processed_labels):
            label = self.processed_labels[idx].clone()
        else:
            # If no labels, return zero vector
            label = torch.zeros(self.num_classes, dtype=torch.float32)
        
        return gradient, routing_weight, label
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
    
    def get_data_info(self):
        info = {
            'num_samples': self.num_samples,
            'gradient_dim': self.gradient_dim,
            'routing_weight_dim': self.routing_weight_dim,
            'num_classes': self.num_classes,
            'capabilities': self.label_processor.CAPABILITIES,
            'has_labels': self.processed_labels is not None
        }
        
        if hasattr(self, 'grad_mean'):
            info['gradient_stats'] = {
                'mean_range': [self.grad_mean.min().item(), self.grad_mean.max().item()],
                'std_range': [self.grad_std.min().item(), self.grad_std.max().item()]
            }
        
        if hasattr(self, 'weight_mean'):
            info['routing_weight_stats'] = {
                'mean': self.weight_mean.tolist(),
                'std': self.weight_std.tolist()
            }
        
        return info
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range")
        
        gradient, routing_weight, label = self[idx]
        
        info = {
            'index': idx,
            'gradient_norm': torch.norm(gradient).item(),
            'routing_weight': routing_weight.tolist(),
            'label_vector': label.tolist(),
            'active_capabilities': self.label_processor.get_capability_names(label)
        }
        
        if self.meta_data and idx < len(self.meta_data):
            info['original_categories'] = self.meta_data[idx].get('categories', [])
            info['meta_id'] = self.meta_data[idx].get('id', 'unknown')
            info['source'] = self.meta_data[idx].get('source', 'unknown')
            question = self.meta_data[idx].get('question', '')
            info['question'] = question[:100] + '...' if len(question) > 100 else question
        
        return info

MultiModalDataset = EnhancedMultiModalDataset

class DataLoaderFactory:
    
    @staticmethod
    def create_dataloader(dataset: MultiModalDataset,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last
        )
    
    @staticmethod
    def create_train_val_dataloaders(dataset: MultiModalDataset,
                                   train_ratio: float = 0.8,
                                   batch_size: int = 32,
                                   num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        num_samples = len(dataset)
        num_train = int(num_samples * train_ratio)
        num_val = num_samples - num_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_dataset.dataset.train()
        val_dataset.dataset.eval()
        
        train_dataloader = DataLoaderFactory.create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        val_dataloader = DataLoaderFactory.create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_dataloader, val_dataloader