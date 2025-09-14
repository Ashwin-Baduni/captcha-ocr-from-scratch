"""
Training script for CAPTCHA classification
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available, logging will be limited")
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from task1.dataset import create_data_loaders
from task1.model import create_model

class CaptchaTrainer:
    """Trainer class for CAPTCHA classification"""

    def __init__(self,
                 model_type: str = 'lightweight',
                 dataset_type: str = 'easy',
                 num_epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 device: str = 'auto',
                 save_dir: str = None,
                 vocab_size: int = 100):
        """
        Initialize trainer

        Args:
            model_type: Type of model to use
            dataset_type: Dataset to train on ('easy', 'hard', 'bonus')
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            device: Device to use ('cpu', 'cuda', or 'auto')
            save_dir: Directory to save results
            vocab_size: Number of words in vocabulary
        """
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        if save_dir is None:

            current_file = Path(__file__).resolve()
            project_root = current_file

            while project_root.parent != project_root:
                if (project_root / 'README.md').exists() and (project_root / 'code').exists():
                    break
                project_root = project_root.parent
            save_dir = str(project_root / 'results')
        self.save_dir = Path(save_dir)
        self.model_dir = self.save_dir / 'models'
        self.log_dir = self.save_dir / 'logs'
        self.results_dir = self.save_dir / 'reports'

        for dir_path in [self.model_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0
        self.training_time = 0

        self.challenges = []

    def setup(self, data_dir: str):
        """Setup model and data loaders"""
        print(f"\nSetting up {self.model_type} model for {self.dataset_type} dataset...")

        self.train_loader, self.val_loader, self.data_info = create_data_loaders(
            data_dir=data_dir,
            dataset_type=self.dataset_type,
            batch_size=self.batch_size,
            vocab_size=self.vocab_size,
            num_workers=2
        )

        actual_vocab_size = self.data_info['vocab_size']
        print(f"Creating model with {actual_vocab_size} classes (vocabulary size)")
        self.model = create_model(
            model_type=self.model_type,
            num_classes=actual_vocab_size,
            dropout=0.5 if self.dataset_type == 'hard' else 0.3
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)

        if TENSORBOARD_AVAILABLE:
            run_name = f"{self.dataset_type}_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(self.log_dir / run_name)
        else:
            self.writer = None

        print(f"Model: {self.model_type}")
        print(f"Dataset: {self.dataset_type}")
        print(f"Training samples: {self.data_info['train_size']}")
        print(f"Validation samples: {self.data_info['val_size']}")
        print(f"Vocabulary size: {self.data_info['vocab_size']}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        class_correct = {}
        class_total = {}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

                pbar.set_postfix({
                    'loss': running_loss / len(pbar),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        self.class_accuracy = {
            idx: (class_correct.get(idx, 0) / class_total.get(idx, 1)) * 100
            for idx in range(self.data_info['vocab_size'])
            if idx in class_total
        }

        return epoch_loss, epoch_acc

    def train(self, data_dir: str):
        """Full training loop"""
        self.setup(data_dir)

        print("\nStarting training...")
        start_time = time.time()

        for epoch in range(self.num_epochs):

            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            self.scheduler.step(val_acc)

            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
            elif epoch == self.num_epochs - 1:
                print("  Saving final model (no improvement found)...")
                self.save_checkpoint(epoch, val_acc)

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            if self.device.type == 'cuda':
                print(f"  🎮 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}/{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

            if val_acc < 30 and epoch > 5:
                self.challenges.append(f"Low accuracy at epoch {epoch}: {val_acc:.2f}%")
                if self.dataset_type == 'hard':
                    print("  Challenge: Hard dataset proving difficult - may need more data or augmentation")
                elif self.dataset_type == 'bonus':
                    print("  Challenge: Conditional transformations confusing the model")

            if val_acc > 95 and self.dataset_type == 'easy':
                print(f"  Early stopping: Reached {val_acc:.2f}% accuracy on easy set")
                break

        self.training_time = time.time() - start_time
        if self.writer:
            self.writer.close()

        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        self.save_results()

    def save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'model_type': self.model_type,
            'dataset_type': self.dataset_type,
            'vocab_size': self.data_info['vocab_size']
        }

        filename = f"{self.dataset_type}_{self.model_type}_best.pth"
        torch.save(checkpoint, self.model_dir / filename)

    def save_results(self):
        """Save training results and analysis"""
        results = {
            'model_type': self.model_type,
            'dataset_type': self.dataset_type,
            'num_epochs': len(self.train_losses),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'vocab_size': self.data_info['vocab_size'],
            'train_size': self.data_info['train_size'],
            'val_size': self.data_info['val_size'],
            'best_val_acc': self.best_val_acc,
            'final_train_acc': self.train_accs[-1] if self.train_accs else 0,
            'training_time': self.training_time,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'challenges': self.challenges,
            'solutions': self.get_solutions()
        }

        filename = f"{self.dataset_type}_{self.model_type}_results.json"
        with open(self.results_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)

        self.create_plots()

    def get_solutions(self) -> List[str]:
        """Get solutions to challenges encountered"""
        solutions = []

        if self.dataset_type == 'hard' and self.best_val_acc < 50:
            solutions.append("Use data augmentation (rotation, scaling)")
            solutions.append("Increase model capacity or use pre-trained features")
            solutions.append("Implement curriculum learning (train on easy first)")

        if self.dataset_type == 'bonus':
            solutions.append("Use conditional batch normalization for different conditions")
            solutions.append("Multi-task learning with condition prediction")

        if len(self.challenges) > 5:
            solutions.append("Implement early stopping to prevent overfitting")
            solutions.append("Use learning rate warmup")

        return solutions

    def create_plots(self):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Results: {self.dataset_type.capitalize()} Dataset - {self.model_type.capitalize()} Model',
                     fontsize=14, fontweight='bold')

        ax1 = axes[0, 0]
        ax1.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        ax1.plot(self.val_losses, label='Val Loss', color='red', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(self.train_accs, label='Train Acc', color='green', alpha=0.7)
        ax2.plot(self.val_accs, label='Val Acc', color='orange', alpha=0.7)
        ax2.axhline(y=self.best_val_acc, color='red', linestyle='--',
                   label=f'Best Val: {self.best_val_acc:.1f}%', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if hasattr(self, 'class_accuracy'):
            sorted_acc = sorted(self.class_accuracy.items(), key=lambda x: x[1])
            worst_5 = sorted_acc[:5]
            best_5 = sorted_acc[-5:]

            words_worst = [self.data_info['idx_to_word'][idx] for idx, _ in worst_5]
            acc_worst = [acc for _, acc in worst_5]
            words_best = [self.data_info['idx_to_word'][idx] for idx, _ in best_5]
            acc_best = [acc for _, acc in best_5]

            y_pos = np.arange(len(words_worst))
            ax3.barh(y_pos, acc_worst, color='red', alpha=0.6)
            ax3.barh(y_pos + len(words_worst) + 1, acc_best, color='green', alpha=0.6)
            ax3.set_yticks(list(y_pos) + list(y_pos + len(words_worst) + 1))
            ax3.set_yticklabels(words_worst + words_best, fontsize=8)
            ax3.set_xlabel('Accuracy (%)')
            ax3.set_title('Best and Worst Performing Words')

        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Dataset: {self.dataset_type.upper()}
        Model: {self.model_type.capitalize()}

        Best Validation Accuracy: {self.best_val_acc:.2f}%
        Final Training Accuracy: {self.train_accs[-1]:.2f}%

        Training Time: {self.training_time:.1f} seconds
        Epochs Trained: {len(self.train_losses)}

        Vocabulary Size: {self.data_info['vocab_size']} words
        Training Samples: {self.data_info['train_size']}
        Validation Samples: {self.data_info['val_size']}

        Challenges: {len(self.challenges)}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')

        plt.tight_layout()

        filename = f"{self.dataset_type}_{self.model_type}_training_plot.png"
        plt.savefig(self.results_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Results saved to {self.results_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train CAPTCHA Classification Model')
    parser.add_argument('--dataset', type=str, default='easy',
                       choices=['easy', 'hard', 'bonus'],
                       help='Dataset to train on')
    parser.add_argument('--model', type=str, default='lightweight',
                       choices=['lightweight', 'standard', 'improved'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--vocab-size', type=int, default=100,
                       help='Vocabulary size (number of unique words)')
    parser.add_argument('--data-dir', type=str,
                       default='data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Output directory for models and logs')

    args = parser.parse_args()

    trainer = CaptchaTrainer(
        model_type=args.model,
        dataset_type=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        vocab_size=args.vocab_size,
        save_dir=args.output_dir
    )

    trainer.train(args.data_dir)

if __name__ == "__main__":
    main()