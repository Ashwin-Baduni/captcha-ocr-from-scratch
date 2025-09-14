"""
Dataset loader for CAPTCHA classification task
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
from torchvision import transforms
from collections import Counter

class CaptchaClassificationDataset(Dataset):
    """Dataset for CAPTCHA classification (100 words)"""

    def __init__(self,
                 data_dir: str,
                 dataset_type: str = 'easy',
                 vocab_size: int = 100,
                 transform: Optional[transforms.Compose] = None,
                 train: bool = True,
                 train_split: float = 0.8,
                 seed: int = 42):
        """
        Initialize CAPTCHA classification dataset

        Args:
            data_dir: Root directory containing data
            dataset_type: 'easy', 'hard', or 'bonus'
            vocab_size: Number of unique words to use (default 100)
            transform: Torchvision transforms to apply
            train: Whether this is training or validation set
            train_split: Proportion of data for training
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.vocab_size = vocab_size
        self.train = train
        self.train_split = train_split
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.images, self.labels, self.word_to_idx, self.idx_to_word = self._load_dataset()

    def _load_dataset(self) -> Tuple[List, List, Dict, Dict]:
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
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'images': []}

        all_images = []
        all_words = []

        for img_file in sorted(img_dir.glob('*.png')):

            parts = img_file.stem.split('_')
            if self.dataset_type == 'bonus':
                word = parts[-1] if len(parts) >= 4 else parts[-1]
            else:
                word = parts[-1] if len(parts) >= 3 else parts[-1]

            all_images.append(str(img_file))
            all_words.append(word)

        word_counts = Counter(all_words)

        vocab_words = list(word_counts.keys())
        print(f"Using all {len(vocab_words)} unique words in dataset")

        word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab_words))}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        filtered_images = []
        filtered_labels = []

        for img_path, word in zip(all_images, all_words):
            if word in word_to_idx:
                filtered_images.append(img_path)
                filtered_labels.append(word_to_idx[word])

        num_samples = len(filtered_images)
        indices = list(range(num_samples))
        random.shuffle(indices)

        split_idx = int(num_samples * self.train_split)

        if self.train:
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        final_images = [filtered_images[i] for i in selected_indices]
        final_labels = [filtered_labels[i] for i in selected_indices]

        print(f"Loaded {len(final_images)} images for {self.dataset_type} {'train' if self.train else 'val'} set")
        print(f"Vocabulary size: {len(word_to_idx)} words")

        return final_images, final_labels, word_to_idx, idx_to_word

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_idx)

    def decode_label(self, label: int) -> str:
        """Convert label index to word"""
        return self.idx_to_word.get(label, 'unknown')

    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced training"""

        class_counts = Counter(self.labels)

        weights = []
        for label in self.labels:
            weights.append(1.0 / class_counts[label])

        return torch.FloatTensor(weights)

def create_data_loaders(data_dir: str,
                       dataset_type: str = 'easy',
                       batch_size: int = 32,
                       vocab_size: int = 100,
                       num_workers: int = 2,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation data loaders

    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """

    train_dataset = CaptchaClassificationDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        vocab_size=vocab_size,
        train=True,
        seed=seed
    )

    val_dataset = CaptchaClassificationDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        vocab_size=vocab_size,
        train=False,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    info = {
        'vocab_size': train_dataset.get_vocab_size(),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'word_to_idx': train_dataset.word_to_idx,
        'idx_to_word': train_dataset.idx_to_word
    }

    if len(train_dataset) < 50:
        print(f"⚠️ WARNING: Only {len(train_dataset)} training samples found. This may be insufficient for proper training.")
        print(f"  Recommended: Generate at least 100 images per dataset for meaningful results.")

    if len(val_dataset) < 10:
        print(f"⚠️ WARNING: Only {len(val_dataset)} validation samples found. This may be insufficient for validation.")
        print(f"  Recommended: Generate at least 100 images per dataset for meaningful results.")

    return train_loader, val_loader, info

if __name__ == "__main__":

    data_dir = str(Path(__file__).parent.parent.parent / "data")

    for dataset_type in ['easy', 'hard', 'bonus']:
        print(f"\n{'='*50}")
        print(f"Testing {dataset_type} dataset")
        print('='*50)

        train_loader, val_loader, info = create_data_loaders(
            data_dir=data_dir,
            dataset_type=dataset_type,
            batch_size=4,
            vocab_size=100
        )

        print(f"Vocabulary size: {info['vocab_size']}")
        print(f"Training samples: {info['train_size']}")
        print(f"Validation samples: {info['val_size']}")

        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        print("Sample words in vocabulary:")
        for i in range(min(5, len(info['idx_to_word']))):
            print(f"  {i}: {info['idx_to_word'][i]}")