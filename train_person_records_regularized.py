#!/usr/bin/env python3
"""
Regularized Training Script for Person Records

This script includes several regularization techniques to prevent overfitting:
- Lower learning rate
- Dropout
- Weight decay  
- Early stopping
- Reduced epochs
- Smaller batch size
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
import json

from ditto_light.dataset import DittoDataset
from torch.utils import data
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class RegularizedDittoModel(nn.Module):
    """A regularized version of DittoModel with dropout and other techniques."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, dropout_prob=0.3):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # Add dropout for regularization
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = torch.nn.Linear(hidden_size, 2)
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x1, x2=None):
        """Encode with regularization"""
        x1 = x1.to(self.device)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]
            enc2 = enc[batch_size:]

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        # Apply dropout
        enc = self.dropout(enc)
        return self.fc(enc)

def evaluate(model, iterator, threshold=None):
    """Evaluate with early stopping capability"""
    all_p = []
    all_y = []
    all_probs = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            loss = criterion(logits, y.to(model.device))
            total_loss += loss.item()
            
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    avg_loss = total_loss / len(iterator)
    
    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1, avg_loss
    else:
        best_th = 0.5
        f1 = 0.0

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th, avg_loss

def train_regularized(config):
    """Train with regularization techniques"""
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load datasets
    trainset = DittoDataset(config['trainset'], 
                           lm=config['lm'], 
                           max_len=config['max_len'],
                           da=config.get('da'))
    validset = DittoDataset(config['validset'], 
                           lm=config['lm'], 
                           max_len=config['max_len'])
    testset = DittoDataset(config['testset'], 
                          lm=config['lm'], 
                          max_len=config['max_len'])

    print(f"Train set size: {len(trainset)}")
    print(f"Valid set size: {len(validset)}")
    print(f"Test set size: {len(testset)}")

    # Create DataLoaders with smaller batch size
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=config['batch_size'],
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=trainset.pad)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=config['batch_size'] * 4,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=validset.pad)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=config['batch_size'] * 4,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=testset.pad)

    # Initialize model with regularization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RegularizedDittoModel(device=device,
                                 lm=config['lm'],
                                 alpha_aug=config.get('alpha_aug', 0.8),
                                 dropout_prob=config.get('dropout_prob', 0.3))
    model = model.to(device)
    
    # Use weight decay for L2 regularization
    optimizer = AdamW(model.parameters(), 
                     lr=config['lr'], 
                     weight_decay=config.get('weight_decay', 0.01))
    
    # Gradient scaler for fp16
    scaler = GradScaler() if config.get('fp16', False) else None
    
    # Learning rate scheduler
    num_steps = (len(trainset) // config['batch_size']) * config['n_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=int(0.1 * num_steps),  # 10% warmup
                                               num_training_steps=num_steps)

    # Early stopping parameters
    best_val_f1 = 0.0
    best_test_f1 = 0.0
    patience = config.get('patience', 5)
    patience_counter = 0
    min_delta = config.get('min_delta', 0.001)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training with regularization...")
    print(f"Learning rate: {config['lr']}")
    print(f"Weight decay: {config.get('weight_decay', 0.01)}")
    print(f"Dropout: {config.get('dropout_prob', 0.3)}")
    print(f"Patience: {patience}")
    
    for epoch in range(1, config['n_epochs'] + 1):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()

            if config.get('fp16', False):
                with autocast():
                    if len(batch) == 2:
                        x, y = batch
                        prediction = model(x)
                    else:
                        x1, x2, y = batch
                        prediction = model(x1, x2)
                    
                    loss = criterion(prediction, y.to(device))
                
                scaler.scale(loss).backward()
                
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                if len(batch) == 2:
                    x, y = batch
                    prediction = model(x)
                else:
                    x1, x2, y = batch
                    prediction = model(x1, x2)

                loss = criterion(prediction, y.to(device))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            
            # Less frequent logging
            if i % 50 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.6f}")

        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_f1, val_th, val_loss = evaluate(model, valid_iter)
        test_f1, test_loss = evaluate(model, test_iter, threshold=val_th)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}, Val F1: {val_f1:.4f}")
        print(f"  Test Loss: {test_loss:.6f}, Test F1: {test_f1:.4f}")
        
        # Early stopping check
        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            best_test_f1 = test_f1
            patience_counter = 0
            
            # Save best model
            if config.get('save_model', False):
                os.makedirs(f"{config['logdir']}/person_records", exist_ok=True)
                ckpt_path = f"{config['logdir']}/person_records/model.pt"
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_f1': val_f1,
                    'test_f1': test_f1
                }
                torch.save(ckpt, ckpt_path)
                print(f"  Saved new best model (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best test F1: {best_test_f1:.4f}")
    
    return best_val_f1, best_test_f1

def main():
    # Configuration with regularization
    config = {
        'trainset': 'data/person_records/train.txt',
        'validset': 'data/person_records/valid.txt', 
        'testset': 'data/person_records/test.txt',
        'lm': 'distilbert',
        'max_len': 128,
        'batch_size': 8,  # Smaller batch size
        'lr': 1e-5,  # Lower learning rate
        'n_epochs': 15,  # Fewer epochs
        'weight_decay': 0.01,  # L2 regularization
        'dropout_prob': 0.4,  # Higher dropout
        'patience': 3,  # Early stopping patience
        'min_delta': 0.001,  # Minimum improvement
        'da': 'del',  # Data augmentation
        'fp16': True,
        'save_model': True,
        'logdir': 'checkpoints'
    }
    
    train_regularized(config)

if __name__ == "__main__":
    main() 