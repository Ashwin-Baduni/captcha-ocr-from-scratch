"""
Training script for Task 2: Text Generation/Extraction from CAPTCHAs
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from task2.generation import (
    CaptchaOCR, CTCOCRModel,
    create_char_vocabulary, decode_sequence,
    calculate_cer, calculate_wer
)
from task2.dataset_seq2seq import create_data_loaders

class OCRTrainer:
    """Trainer for CAPTCHA OCR models"""

    def __init__(self,
                 model_type: str = 'seq2seq',
                 dataset_type: str = 'easy',
                 vocab_size: int = 66,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 device: str = None,
                 output_dir: str = None):
        """
        Initialize OCR trainer

        Args:
            model_type: 'seq2seq' or 'ctc'
            dataset_type: 'easy', 'hard', or 'bonus'
            vocab_size: Size of character vocabulary
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use (cuda/cpu)
            output_dir: Directory to save models and results
        """
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if output_dir is None:

            current_file = Path(__file__).resolve()
            project_root = current_file

            while project_root.parent != project_root:
                if (project_root / 'README.md').exists() and (project_root / 'code').exists():
                    break
                project_root = project_root.parent
            output_dir = str(project_root / 'results')
        self.output_dir = Path(output_dir)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model_dir = self.output_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.report_dir = self.output_dir / 'reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)

        if model_type == 'seq2seq':
            self.model = CaptchaOCR(vocab_size).to(self.device)
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.model = CTCOCRModel(vocab_size).to(self.device)
            self.criterion = nn.CTCLoss(blank=3, zero_infinity=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.char_to_idx, self.idx_to_char = create_char_vocabulary()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cer': [],
            'val_cer': [],
            'train_wer': [],
            'val_wer': [],
            'train_exact_match': [],
            'val_exact_match': []
        }

        self.best_val_cer = float('inf')

    def train_epoch_seq2seq(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch with seq2seq model"""
        self.model.train()

        total_loss = 0
        total_cer = 0
        total_wer = 0
        exact_matches = 0
        num_samples = 0

        progress_bar = tqdm(train_loader, desc='Training')

        for batch_idx, (images, sequences, texts) in enumerate(progress_bar):
            images = images.to(self.device)
            sequences = sequences.to(self.device)

            input_seq = sequences[:, :-1]
            target_seq = sequences[:, 1:]

            outputs, _ = self.model(images, input_seq)

            loss = self.criterion(
                outputs.reshape(-1, self.vocab_size),
                target_seq.reshape(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            with torch.no_grad():
                pred_sequences = outputs.argmax(dim=-1)

                for i in range(len(texts)):
                    pred_text = decode_sequence(pred_sequences[i], self.idx_to_char)
                    target_text = texts[i]

                    cer = calculate_cer(pred_text, target_text)
                    wer = calculate_wer(pred_text, target_text)

                    total_cer += cer
                    total_wer += wer

                    if pred_text == target_text:
                        exact_matches += 1

                    num_samples += 1

            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_cer': total_cer / max(num_samples, 1)
            })

        return {
            'loss': total_loss / len(train_loader),
            'cer': total_cer / num_samples,
            'wer': total_wer / num_samples,
            'exact_match': exact_matches / num_samples
        }

    def train_epoch_ctc(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch with CTC model"""
        self.model.train()

        total_loss = 0
        total_cer = 0
        total_wer = 0
        exact_matches = 0
        num_samples = 0

        progress_bar = tqdm(train_loader, desc='Training')

        for batch_idx, (images, targets, target_lengths, texts) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            outputs = self.model(images)

            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

            loss = self.criterion(outputs, targets, input_lengths, target_lengths)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            with torch.no_grad():

                pred_sequences = outputs.argmax(dim=-1).permute(1, 0)

                for i in range(len(texts)):

                    pred_seq = []
                    prev = -1
                    for idx in pred_sequences[i]:
                        if idx != 3 and idx != prev:
                            pred_seq.append(idx.item())
                        prev = idx

                    pred_text = decode_sequence(pred_seq, self.idx_to_char)
                    target_text = texts[i]

                    cer = calculate_cer(pred_text, target_text)
                    wer = calculate_wer(pred_text, target_text)

                    total_cer += cer
                    total_wer += wer

                    if pred_text == target_text:
                        exact_matches += 1

                    num_samples += 1

            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_cer': total_cer / max(num_samples, 1)
            })

        return {
            'loss': total_loss / len(train_loader),
            'cer': total_cer / num_samples,
            'wer': total_wer / num_samples,
            'exact_match': exact_matches / num_samples
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        total_cer = 0
        total_wer = 0
        exact_matches = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                if self.model_type == 'seq2seq':
                    images, sequences, texts = batch
                    images = images.to(self.device)
                    sequences = sequences.to(self.device)

                    outputs, _ = self.model.generate(images)

                    input_seq = sequences[:, :-1]
                    target_seq = sequences[:, 1:]
                    outputs_tf, _ = self.model(images, input_seq)

                    loss = self.criterion(
                        outputs_tf.reshape(-1, self.vocab_size),
                        target_seq.reshape(-1)
                    )

                    pred_sequences = outputs.argmax(dim=-1)

                else:
                    images, targets, target_lengths, texts = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    target_lengths = target_lengths.to(self.device)

                    outputs = self.model(images)
                    input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

                    loss = self.criterion(outputs, targets, input_lengths, target_lengths)

                    pred_sequences = outputs.argmax(dim=-1).permute(1, 0)

                total_loss += loss.item()

                for i in range(len(texts)):
                    if self.model_type == 'seq2seq':
                        pred_text = decode_sequence(pred_sequences[i], self.idx_to_char)
                    else:

                        pred_seq = []
                        prev = -1
                        for idx in pred_sequences[i]:
                            if idx != 3 and idx != prev:
                                pred_seq.append(idx.item())
                            prev = idx
                        pred_text = decode_sequence(pred_seq, self.idx_to_char)

                    target_text = texts[i]

                    cer = calculate_cer(pred_text, target_text)
                    wer = calculate_wer(pred_text, target_text)

                    total_cer += cer
                    total_wer += wer

                    if pred_text == target_text:
                        exact_matches += 1

                    num_samples += 1

        return {
            'loss': total_loss / len(val_loader),
            'cer': total_cer / num_samples,
            'wer': total_wer / num_samples,
            'exact_match': exact_matches / num_samples
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print(f"\nStarting training for {self.model_type} model on {self.dataset_type} dataset")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print('='*50)

            if self.model_type == 'seq2seq':
                train_metrics = self.train_epoch_seq2seq(train_loader)
            else:
                train_metrics = self.train_epoch_ctc(train_loader)

            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics['loss'])

            for key in ['loss', 'cer', 'wer', 'exact_match']:
                self.history[f'train_{key}'].append(train_metrics[key])
                self.history[f'val_{key}'].append(val_metrics[key])

            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"CER: {train_metrics['cer']:.4f}, "
                  f"WER: {train_metrics['wer']:.4f}, "
                  f"Exact: {train_metrics['exact_match']:.2%}")

            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"CER: {val_metrics['cer']:.4f}, "
                  f"WER: {val_metrics['wer']:.4f}, "
                  f"Exact: {val_metrics['exact_match']:.2%}")

            if val_metrics['cer'] < self.best_val_cer:
                self.best_val_cer = val_metrics['cer']
                self.save_checkpoint(epoch, val_metrics)
                print(f"✓ Saved best model (CER: {self.best_val_cer:.4f})")

            if epoch > 10 and val_metrics['cer'] > 0.8:
                print("Warning: Model not learning effectively. Consider adjusting hyperparameters.")

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'dataset_type': self.dataset_type,
            'history': self.history
        }

        model_path = self.model_dir / f'{self.dataset_type}_{self.model_type}_best.pth'
        torch.save(checkpoint, model_path)

        history_path = self.report_dir / f'{self.dataset_type}_{self.model_type}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def generate_report(self):
        """Generate training report"""
        report = {
            'model_type': self.model_type,
            'dataset_type': self.dataset_type,
            'num_epochs': len(self.history['train_loss']),
            'best_val_cer': self.best_val_cer,
            'final_metrics': {
                'train_cer': self.history['train_cer'][-1] if self.history['train_cer'] else 0,
                'val_cer': self.history['val_cer'][-1] if self.history['val_cer'] else 0,
                'train_wer': self.history['train_wer'][-1] if self.history['train_wer'] else 0,
                'val_wer': self.history['val_wer'][-1] if self.history['val_wer'] else 0,
                'train_exact': self.history['train_exact_match'][-1] if self.history['train_exact_match'] else 0,
                'val_exact': self.history['val_exact_match'][-1] if self.history['val_exact_match'] else 0,
            }
        }

        report_path = self.report_dir / f'{self.dataset_type}_{self.model_type}_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nTraining report saved to {report_path}")

        return report

def main():
    parser = argparse.ArgumentParser(description='Train OCR model for CAPTCHA text extraction')
    parser.add_argument('--dataset', type=str, default='easy',
                       choices=['easy', 'hard', 'bonus'],
                       help='Dataset to train on')
    parser.add_argument('--model', type=str, default='seq2seq',
                       choices=['seq2seq', 'ctc'],
                       help='Model type')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, vocab_size = create_data_loaders(
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        use_ctc=(args.model == 'ctc')
    )

    trainer = OCRTrainer(
        model_type=args.model,
        dataset_type=args.dataset,
        vocab_size=vocab_size,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir
    )

    trainer.train(train_loader, val_loader)

    report = trainer.generate_report()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best validation CER: {report['best_val_cer']:.4f}")
    print(f"Final validation exact match: {report['final_metrics']['val_exact']:.2%}")

if __name__ == "__main__":
    main()