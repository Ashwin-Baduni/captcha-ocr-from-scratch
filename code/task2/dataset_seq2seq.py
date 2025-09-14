"""
Dataset loader for sequence-to-sequence CAPTCHA OCR
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

class CaptchaSeq2SeqDataset(Dataset):
    """Dataset for CAPTCHA text extraction (sequence-to-sequence)"""

    def __init__(self,
                 data_dir: str,
                 dataset_type: str = 'easy',
                 transform: Optional[transforms.Compose] = None,
                 train: bool = True,
                 train_split: float = 0.8,
                 max_length: int = 20,
                 seed: int = 42):
        """
        Initialize CAPTCHA sequence-to-sequence dataset

        Args:
            data_dir: Root directory containing data
            dataset_type: 'easy', 'hard', or 'bonus'
            transform: Torchvision transforms to apply
            train: Whether this is training or validation set
            train_split: Proportion of data for training
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.train = train
        self.train_split = train_split
        self.max_length = max_length
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.char_to_idx, self.idx_to_char = self._create_vocabulary()
        self.vocab_size = len(self.char_to_idx)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((60, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.images, self.texts = self._load_dataset()

    def _create_vocabulary(self) -> Tuple[Dict, Dict]:
        """Create character-level vocabulary"""
        char_to_idx = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
        }

        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            char_to_idx[char] = i + 4

        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            char_to_idx[char] = i + 30

        for i, char in enumerate('0123456789'):
            char_to_idx[char] = i + 56

        idx_to_char = {v: k for k, v in char_to_idx.items()}

        return char_to_idx, idx_to_char

    def _load_dataset(self) -> Tuple[List, List]:
        """Load and prepare dataset"""

        if self.data_dir.name in ['easy', 'hard', 'bonus']:
            img_dir = self.data_dir
        else:
            img_dir = self.data_dir / self.dataset_type

        if not img_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {img_dir}")

        if self.data_dir.name in ['easy', 'hard', 'bonus']:
            base_data_dir = self.data_dir.parent
        else:
            base_data_dir = self.data_dir

        metadata_file = base_data_dir / 'metadata' / f'{self.dataset_type}_metadata.json'

        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)

                for img_meta in metadata_dict.get('images', []):
                    filename = img_meta.get('filename', '')
                    metadata[filename] = img_meta

        all_images = []
        all_texts = []

        for img_file in sorted(img_dir.glob('*.png')):

            filename = img_file.name

            if filename in metadata:

                text = metadata[filename].get('original_text') or metadata[filename].get('text')
            else:

                parts = img_file.stem.split('_')
                if self.dataset_type == 'bonus':
                    text = parts[-1] if len(parts) >= 4 else parts[-1]
                else:
                    text = parts[-1] if len(parts) >= 3 else parts[-1]

            text = text.lower()

            all_images.append(str(img_file))
            all_texts.append(text)

        num_samples = len(all_images)
        indices = list(range(num_samples))
        random.shuffle(indices)

        split_idx = int(num_samples * self.train_split)

        if self.train:
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        final_images = [all_images[i] for i in selected_indices]
        final_texts = [all_texts[i] for i in selected_indices]

        print(f"Loaded {len(final_images)} images for {self.dataset_type} {'train' if self.train else 'val'} set")
        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Sample texts: {final_texts[:5]}")

        return final_images, final_texts

    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        sequence = [self.char_to_idx['<START>']]

        for char in text.lower():
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(self.char_to_idx['<UNK>'])

        sequence.append(self.char_to_idx['<END>'])

        if len(sequence) < self.max_length:
            sequence += [self.char_to_idx['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length-1] + [self.char_to_idx['<END>']]

        return sequence

    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text"""
        text = []
        for idx in sequence:
            if idx == self.char_to_idx['<PAD>']:
                break
            if idx == self.char_to_idx['<START>']:
                continue
            if idx == self.char_to_idx['<END>']:
                break

            char = self.idx_to_char.get(idx, '?')
            text.append(char)

        return ''.join(text)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample"""
        img_path = self.images[idx]
        text = self.texts[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sequence = self.text_to_sequence(text)
        sequence_tensor = torch.LongTensor(sequence)

        return image, sequence_tensor, text

def collate_fn(batch):
    """Custom collate function for batching"""
    images, sequences, texts = zip(*batch)

    images = torch.stack(images, dim=0)

    sequences = torch.stack(sequences, dim=0)

    return images, sequences, texts

class CTCDataset(CaptchaSeq2SeqDataset):
    """Dataset for CTC-based OCR training"""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Get a single sample for CTC training"""
        img_path = self.images[idx]
        text = self.texts[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sequence = []
        for char in text.lower():
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(self.char_to_idx['<UNK>'])

        sequence_tensor = torch.LongTensor(sequence)
        sequence_length = len(sequence)

        return image, sequence_tensor, sequence_length, text

def ctc_collate_fn(batch):
    """Custom collate function for CTC training"""
    images, sequences, lengths, texts = zip(*batch)

    images = torch.stack(images, dim=0)

    sequences = torch.cat(sequences, dim=0)
    lengths = torch.LongTensor(lengths)

    return images, sequences, lengths, texts

def create_data_loaders(data_dir: str,
                       dataset_type: str = 'easy',
                       batch_size: int = 32,
                       num_workers: int = 2,
                       use_ctc: bool = False,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for seq2seq OCR

    Args:
        data_dir: Path to data directory
        dataset_type: 'easy', 'hard', or 'bonus'
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        use_ctc: Whether to use CTC dataset/collate
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """

    if use_ctc:

        train_dataset = CTCDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            train=True,
            seed=seed
        )

        val_dataset = CTCDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            train=False,
            seed=seed
        )

        collate = ctc_collate_fn
    else:

        train_dataset = CaptchaSeq2SeqDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            train=True,
            seed=seed
        )

        val_dataset = CaptchaSeq2SeqDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            train=False,
            seed=seed
        )

        collate = collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    if len(train_dataset) < 50:
        print(f"⚠️ WARNING: Only {len(train_dataset)} training samples found for OCR. This may be insufficient for proper training.")
        print(f"  Recommended: Generate at least 100 images per dataset for meaningful results.")

    if len(val_dataset) < 10:
        print(f"⚠️ WARNING: Only {len(val_dataset)} validation samples found for OCR. This may be insufficient for validation.")
        print(f"  Recommended: Generate at least 100 images per dataset for meaningful results.")

    return train_loader, val_loader, train_dataset.vocab_size

if __name__ == "__main__":

    data_dir = str(Path(__file__).parent.parent.parent / "data")

    for dataset_type in ['easy', 'hard', 'bonus']:
        print(f"\n{'='*50}")
        print(f"Testing {dataset_type} seq2seq dataset")
        print('='*50)

        try:

            train_loader, val_loader, vocab_size = create_data_loaders(
                data_dir=data_dir,
                dataset_type=dataset_type,
                batch_size=4,
                use_ctc=False
            )

            print(f"Vocabulary size: {vocab_size}")

            images, sequences, texts = next(iter(train_loader))
            print(f"Batch images shape: {images.shape}")
            print(f"Batch sequences shape: {sequences.shape}")
            print(f"Sample texts: {texts}")

            print(f"\nTesting CTC dataset for {dataset_type}")
            train_loader_ctc, val_loader_ctc, _ = create_data_loaders(
                data_dir=data_dir,
                dataset_type=dataset_type,
                batch_size=4,
                use_ctc=True
            )

            images, sequences, lengths, texts = next(iter(train_loader_ctc))
            print(f"CTC images shape: {images.shape}")
            print(f"CTC sequences shape: {sequences.shape}")
            print(f"CTC lengths: {lengths}")
            print(f"CTC texts: {texts}")

        except Exception as e:
            print(f"Error loading {dataset_type} dataset: {e}")