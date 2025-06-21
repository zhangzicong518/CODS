import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from model import ImprovedHyperSphereClassifier, FocalLoss, StableContrastiveLoss
from dataloader import MultiModalDataset, DataLoaderFactory
import os
import numpy as np
import math


class RepulsionLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(RepulsionLoss, self).__init__()
        self.margin = margin
        
    def forward(self, base_vectors):
        normalized_vectors = torch.nn.functional.normalize(base_vectors, p=2, dim=1)
        
        # Calculate cosine similarity between all base vectors
        similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t())
        
        # Remove diagonal elements (self-similarity)
        num_classes = base_vectors.size(0)
        mask = torch.eye(num_classes, device=base_vectors.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # Use ReLU to ensure only similarities above margin contribute to loss
        repulsion_loss = torch.relu(similarity_matrix.abs() - self.margin).sum()
        repulsion_loss = repulsion_loss / (num_classes * (num_classes - 1))
        
        return repulsion_loss


class TwoStageTrainer:
    def __init__(self, model, device, stage1_lr=1e-3, stage2_lr=5e-4):
        self.model = model.to(device)
        self.device = device
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.contrastive_loss = StableContrastiveLoss(temperature=0.1)
        self.repulsion_loss = RepulsionLoss(margin=0.1)
        
        self.best_val_f1 = 0
        self.patience_counter = 0
    
    def freeze_base_vectors(self):
        self.model.base_vectors.requires_grad = False
        
    def unfreeze_base_vectors(self):
        self.model.base_vectors.requires_grad = True
        
    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
    
    def get_stage1_optimizer(self):
        self.freeze_base_vectors()
        self.unfreeze_encoder()
        
        # Only optimize encoder and sphere_projection parameters
        trainable_params = list(self.model.encoder.parameters()) + \
                          list(self.model.sphere_projection.parameters()) + \
                          list(self.model.attention.parameters()) + \
                          list(self.model.classifier.parameters())
        
        return optim.AdamW(trainable_params, lr=self.stage1_lr, weight_decay=1e-4)
    
    def get_stage2_optimizer(self):
        # Freeze encoder, unfreeze base_vectors
        self.freeze_encoder()
        self.unfreeze_base_vectors()
        
        # Only optimize sphere_projection, base_vectors and related components
        trainable_params = list(self.model.sphere_projection.parameters()) + \
                          list(self.model.attention.parameters()) + \
                          list(self.model.classifier.parameters()) + \
                          [self.model.base_vectors, self.model.temperature]
        
        return optim.AdamW(trainable_params, lr=self.stage2_lr, weight_decay=1e-4)
    
    def train_stage1(self, dataloader, epochs=15):
        """
        Stage 1 training: Train encoder and related components, freeze base_vectors
        Mainly use classification loss to learn basic feature representation
        """
        print("=" * 60)
        print("Starting Stage 1 Training")
        print("Training: Encoder + Sphere Projection + Attention + Classifier")
        print("Frozen: Base Vectors")
        print("Primary Loss: Focal Loss (Classification)")
        print("=" * 60)
        
        self.model.train()
        optimizer = self.get_stage1_optimizer()
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.stage1_lr * 0.1)
        
        for epoch in range(epochs):
            total_loss = 0
            total_classification_loss = 0
            successful_batches = 0
            
            for batch_idx, (gradients, weights, labels) in enumerate(dataloader):
                gradients = gradients.to(self.device, dtype=torch.float32)
                weights = weights.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                
                optimizer.zero_grad()
                
                sphere_features, similarities, logits = self.model(weights, gradients)
                classification_loss = self.focal_loss(logits, labels)
                total_loss_batch = classification_loss
                total_loss_batch.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                optimizer.step()
                
                self.model._normalize_base_vectors()
                total_loss += total_loss_batch.item()
                total_classification_loss += classification_loss.item()
                successful_batches += 1
                
                if batch_idx % 200 == 0:
                    print(f'Stage 1 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}')
                    print(f'  Classification Loss: {classification_loss.item():.4f}')
                
            scheduler.step()
            
            if successful_batches == 0:
                print(f"Stage 1 - Epoch {epoch+1}: No successful batches")
                continue
            
            avg_loss = total_loss / successful_batches
            avg_classification_loss = total_classification_loss / successful_batches
            
            print(f'Stage 1 - Epoch {epoch+1}/{epochs} Complete')
            print(f'  Average Classification Loss: {avg_classification_loss:.4f}')
            print(f'  Average Total Loss: {avg_loss:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        print("Stage 1 training complete!")
    
    def train_stage2(self, dataloader, epochs=12, contrastive_weight=0.1, repulsion_weight=0.05):
        """
        Stage 2 training: Fix encoder, train sphere_projection and base_vectors
        Use classification loss + contrastive loss + repulsion loss
        """
        print("=" * 60)
        print("Starting Stage 2 Training")
        print("Training: Sphere Projection + Base Vectors + Temperature")
        print("Frozen: Encoder")
        print(f"Loss Combination: Focal Loss + Contrastive Loss({contrastive_weight}) + Repulsion Loss({repulsion_weight})")
        print("=" * 60)
        
        self.model.train()
        optimizer = self.get_stage2_optimizer()
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.stage2_lr * 0.1)
        
        for epoch in range(epochs):
            total_loss = 0
            total_classification_loss = 0
            total_contrastive_loss = 0
            total_repulsion_loss = 0
            successful_batches = 0
            
            for batch_idx, (gradients, weights, labels) in enumerate(dataloader):
                gradients = gradients.to(self.device, dtype=torch.float32)
                weights = weights.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                
                optimizer.zero_grad()
                
                sphere_features, similarities, logits = self.model(weights, gradients)
                
                # 1. Classification loss
                classification_loss = self.focal_loss(logits, labels)
                
                # 2. Contrastive loss
                contrastive_loss = self.contrastive_loss(sphere_features, labels)
                
                # 3. Base vector repulsion loss
                repulsion_loss = self.repulsion_loss(self.model.base_vectors)
                
                total_loss_batch = (
                    classification_loss + 
                    contrastive_weight * contrastive_loss +
                    repulsion_weight * repulsion_loss
                )
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()               
                self.model._normalize_base_vectors()
                
                # Statistics
                total_loss += total_loss_batch.item()
                total_classification_loss += classification_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_repulsion_loss += repulsion_loss.item()
                successful_batches += 1
                
                if batch_idx % 200 == 0:
                    print(f'Stage 2 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}')
                    print(f'  Classification Loss: {classification_loss.item():.4f}')
                    print(f'  Contrastive Loss: {contrastive_loss.item():.4f}')
                    print(f'  Repulsion Loss: {repulsion_loss.item():.4f}')
                    print(f'  Total Loss: {total_loss_batch.item():.4f}')

            scheduler.step()
            
            if successful_batches == 0:
                print(f"Stage 2 - Epoch {epoch+1}: No successful batches")
                continue
            
            avg_loss = total_loss / successful_batches
            avg_classification_loss = total_classification_loss / successful_batches
            avg_contrastive_loss = total_contrastive_loss / successful_batches
            avg_repulsion_loss = total_repulsion_loss / successful_batches
            
            print(f'Stage 2 - Epoch {epoch+1}/{epochs} Complete')
            print(f'  Average Classification Loss: {avg_classification_loss:.4f}')
            print(f'  Average Contrastive Loss: {avg_contrastive_loss:.4f}')
            print(f'  Average Repulsion Loss: {avg_repulsion_loss:.4f}')
            print(f'  Average Total Loss: {avg_loss:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        print("Stage 2 training complete!")
    
    def validate(self, dataloader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        
        with torch.no_grad():
            for gradients, weights, labels in dataloader:
                gradients = gradients.to(self.device, dtype=torch.float32)
                weights = weights.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                
                sphere_features, similarities, logits = self.model(weights, gradients)

                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                correct = (predictions == labels).all(dim=1).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                
                for i in range(labels.size(0)):
                    pred = predictions[i]
                    true = labels[i]
                    
                    tp = (pred * true).sum().item()
                    fp = (pred * (1 - true)).sum().item()
                    fn = ((1 - pred) * true).sum().item()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
        
        if total_samples == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        return {
            'accuracy': total_correct / total_samples,
            'precision': total_precision / total_samples,
            'recall': total_recall / total_samples,
            'f1': total_f1 / total_samples
        }
    
    def full_training(self, dataloader, val_dataloader=None, 
                     stage1_epochs=15, stage2_epochs=12, 
                     contrastive_weight=0.1, repulsion_weight=0.05):
        print("Starting complete two-stage training process")
        print(f"Stage 1 epochs: {stage1_epochs}")
        print(f"Stage 2 epochs: {stage2_epochs}")
        print(f"Contrastive weight: {contrastive_weight}")
        print(f"Repulsion weight: {repulsion_weight}")
        
        self.train_stage1(dataloader, stage1_epochs)
        
        if val_dataloader is not None:
            print("\nValidation after Stage 1:")
            stage1_metrics = self.validate(val_dataloader)
            print(f"  Validation F1: {stage1_metrics['f1']:.4f}")
            print(f"  Validation Accuracy: {stage1_metrics['accuracy']:.4f}")
        
        self.train_stage2(dataloader, stage2_epochs, contrastive_weight, repulsion_weight)
        
        if val_dataloader is not None:
            print("\nValidation after Stage 2:")
            stage2_metrics = self.validate(val_dataloader)
            print(f"  Validation F1: {stage2_metrics['f1']:.4f}")
            print(f"  Validation Accuracy: {stage2_metrics['accuracy']:.4f}")
        
        print("=" * 60)
        print("Two-stage training complete!")
        print("=" * 60)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    print("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, (gradients, weights, labels) in enumerate(dataloader):
            # Ensure data type consistency
            gradients = gradients.to(device, dtype=torch.float32)
            weights = weights.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
                
            sphere_features, similarities, logits = model(weights, gradients)
            
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            # Calculate exact match accuracy (all labels correct)
            correct = (predictions == labels).all(dim=1).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            for i in range(labels.size(0)):
                pred = predictions[i]
                true = labels[i]
                
                # Calculate precision, recall, f1
                tp = (pred * true).sum().item()
                fp = (pred * (1 - true)).sum().item()
                fn = ((1 - pred) * true).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1             
    
    if total_samples == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    accuracy = total_correct / total_samples
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_f1 = total_f1 / total_samples
    
    print(f"\nModel evaluation results:")
    print(f"Exact match accuracy: {accuracy:.4f}")
    print(f"Average precision: {avg_precision:.4f}")
    print(f"Average recall: {avg_recall:.4f}")
    print(f"Average F1 score: {avg_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }


def save_model_and_info(model, data_info, config, save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(save_dir, 'two_stage_hypersphere_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim_weights': data_info['routing_weight_dim'],
            'input_dim_gradients': data_info['gradient_dim'],
            'num_classes': data_info['num_classes'],
            'projection_dim': config['projection_dim'],
            'hidden_dim': config['hidden_dim'],
            'num_blocks': config['num_blocks'],
            'sphere_dim': config['sphere_dim'],
            'dropout_rate': config['dropout_rate']
        },
        'training_config': config,
        'data_info': data_info
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Save base vector orthogonality information
    with torch.no_grad():
        gram_matrix, orthogonality_error = model.get_base_vector_orthogonality()
        
        orthogonality_path = os.path.join(save_dir, 'base_vectors_analysis.pth')
        torch.save({
            'gram_matrix': gram_matrix,
            'orthogonality_error': orthogonality_error,
            'base_vectors': model.base_vectors.clone(),
            'temperature': model.temperature.clone()
        }, orthogonality_path)
        
        print(f"Base vector analysis saved to: {orthogonality_path}")
        print(f"Base vector orthogonality error: {orthogonality_error:.6f}")
        print(f"Temperature parameter: {model.temperature.item():.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Two-stage training configuration
    config = {
        # Model parameters
        'projection_dim': 256,
        'hidden_dim': 256,
        'num_blocks': 4,
        'sphere_dim': 64,  # Lower sphere dimension for improved stability
        'dropout_rate': 0.2,
        
        # Training parameters
        'stage1_lr': 2e-3,      # Stage 1 learning rate
        'stage2_lr': 5e-4,      # Stage 2 learning rate
        'stage1_epochs': 20,    # Stage 1 epochs
        'stage2_epochs': 6,     # Stage 2 epochs
        'batch_size': 32,
        
        # Loss weights
        'contrastive_weight': 0.1,   # Contrastive loss weight
        'repulsion_weight': 0.05,    # Repulsion loss weight
    }
    
    gradients_file = "your_gradients_file.pt"
    routing_weights_file = "your_routing_weights_file.pt"
    meta_file = "your_meta_file.json"
    
    print("Creating dataset...")
    dataset = MultiModalDataset(
        gradients_file=gradients_file,
        routing_weights_file=routing_weights_file,
        meta_file=meta_file,
        transform_gradients=True,
        transform_weights=True,
        max_samples=None
    )
    
    data_info = dataset.get_data_info()
    gradient_dim = data_info['gradient_dim']
    routing_weight_dim = data_info['routing_weight_dim']
    num_classes = data_info['num_classes']
    
    """
    print(f"\nDataset information:")
    print(f"Number of samples: {data_info['num_samples']}")
    print(f"Gradient dimension: {gradient_dim}")
    print(f"Routing weight dimension: {routing_weight_dim}")
    print(f"Number of classes: {num_classes}")
    """
    
    train_dataloader, val_dataloader = DataLoaderFactory.create_train_val_dataloaders(
        dataset=dataset,
        train_ratio=0.8,
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    model = ImprovedHyperSphereClassifier(
        input_dim_weights=routing_weight_dim,
        input_dim_gradients=gradient_dim,
        num_classes=num_classes,
        projection_dim=config['projection_dim'],
        hidden_dim=config['hidden_dim'],
        num_blocks=config['num_blocks'],
        sphere_dim=config['sphere_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = TwoStageTrainer(
        model, 
        device, 
        stage1_lr=config['stage1_lr'], 
        stage2_lr=config['stage2_lr']
    )
    
    # Execute two-stage training
    print("\nStarting two-stage training...")
    trainer.full_training(
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        stage1_epochs=config['stage1_epochs'],
        stage2_epochs=config['stage2_epochs'],
        contrastive_weight=config['contrastive_weight'],
        repulsion_weight=config['repulsion_weight']
    )
    
    train_metrics = evaluate_model(model, train_dataloader, device)
    val_metrics = evaluate_model(model, val_dataloader, device)
    
    print(f"\nTraining set performance:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nValidation set performance:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    save_model_and_info(model, data_info, config)