"""
Base generator class for CAPTCHA dataset creation
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import hashlib
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import config as settings

class BaseGenerator(ABC):
    """Abstract base class for CAPTCHA image generation"""

    def __init__(self, dataset_type: str, seed: Optional[int] = None):
        """
        Initialize the base generator

        Args:
            dataset_type: Type of dataset ('easy', 'hard', 'bonus')
            seed: Random seed for reproducibility
        """
        self.dataset_type = dataset_type
        self.seed = seed or settings.RANDOM_SEED
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.output_dir = self._get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            'dataset_type': dataset_type,
            'creation_date': datetime.now().isoformat(),
            'seed': self.seed,
            'images': []
        }

        self.stats = {
            'total_generated': 0,
            'word_lengths': [],
            'difficulty_scores': [],
            'generation_times': []
        }

    def _get_output_dir(self) -> Path:
        """Get the output directory based on dataset type"""
        dir_map = {
            'easy': settings.EASY_DIR,
            'hard': settings.HARD_DIR,
            'bonus': settings.BONUS_DIR
        }
        return dir_map.get(self.dataset_type, settings.DATA_DIR / self.dataset_type)

    @abstractmethod
    def generate_image(self, text: str, index: int) -> Tuple[Image.Image, Dict]:
        """
        Generate a single CAPTCHA image

        Args:
            text: The text to render
            index: Index of the image in the dataset

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        pass

    def get_random_font(self) -> str:
        """Get a random font based on dataset type"""
        fonts = settings.FONTS.get(self.dataset_type, settings.FONTS['hard'])
        if isinstance(fonts, list):

            font_paths = []
            for font_name in fonts:

                possible_paths = [
                    f"/usr/share/fonts/truetype/dejavu/{font_name.replace(' ', '')}.ttf",
                    f"/usr/share/fonts/truetype/liberation/{font_name.replace(' ', '')}.ttf",
                    f"/usr/share/fonts/truetype/{font_name.lower().replace(' ', '-')}.ttf",
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        font_paths.append(path)
                        break

            if font_paths:
                return random.choice(font_paths)

        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def get_random_font_size(self) -> int:
        """Get random font size based on dataset type"""
        size_range = settings.FONT_SIZE_RANGE.get(
            self.dataset_type,
            settings.FONT_SIZE_RANGE['hard']
        )
        return random.randint(*size_range)

    def calculate_text_position(self, draw: ImageDraw.Draw, text: str,
                               font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Calculate centered text position"""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (settings.IMAGE_WIDTH - text_width) // 2
        y = (settings.IMAGE_HEIGHT - text_height) // 2

        if self.dataset_type in ['hard', 'bonus']:
            x += random.randint(-10, 10)
            y += random.randint(-5, 5)

        return max(0, x), max(0, y)

    def generate_dataset(self, num_images: Optional[int] = None) -> Dict:
        """
        Generate the complete dataset

        Args:
            num_images: Number of images to generate (defaults to config setting)

        Returns:
            Dictionary containing generation statistics
        """
        num_images = num_images or settings.DATASET_SIZE[self.dataset_type]
        word_list = self.get_word_list(num_images)

        print(f"Generating {num_images} images for {self.dataset_type} set...")

        from tqdm import tqdm
        import time

        for idx in tqdm(range(num_images), desc=f"{self.dataset_type.capitalize()} dataset", unit="img"):
            start_time = time.time()

            word = word_list[idx % len(word_list)]

            image, metadata = self.generate_image(word, idx)

            filename = f"{self.dataset_type}_{idx:04d}_{word}.png"
            filepath = self.output_dir / filename
            image.save(filepath, 'PNG')

            metadata.update({
                'filename': filename,
                'filepath': str(filepath),
                'index': idx,
                'text': word,
                'word_length': len(word)
            })
            self.metadata['images'].append(metadata)

            self.stats['total_generated'] += 1
            self.stats['word_lengths'].append(len(word))
            if 'difficulty_score' in metadata:
                self.stats['difficulty_scores'].append(metadata['difficulty_score'])
            self.stats['generation_times'].append(time.time() - start_time)

        self.save_metadata()

        return self.get_generation_summary()

    def get_word_list(self, num_images: int) -> List[str]:
        """Get list of words for generation with smart repetition for smaller datasets"""
        base_word_list = settings.WORD_LIST.copy()

        if num_images <= 100:

            num_unique_words = min(20, num_images // 5 + 5)
        elif num_images <= 500:

            num_unique_words = min(50, num_images // 10 + 10)
        else:

            num_unique_words = min(len(base_word_list), 110)

        random.shuffle(base_word_list)
        selected_words = base_word_list[:num_unique_words]

        word_list = []
        while len(word_list) < num_images:
            word_list.extend(selected_words)

        random.shuffle(word_list)

        print(f"  Using {num_unique_words} unique words, ~{num_images/num_unique_words:.1f} images per word")

        return word_list[:num_images]

    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_file = settings.METADATA_DIR / f"{self.dataset_type}_metadata.json"
        settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Metadata saved to {metadata_file}")

    def get_generation_summary(self) -> Dict:
        """Get summary statistics of generation"""
        summary = {
            'dataset_type': self.dataset_type,
            'total_images': self.stats['total_generated'],
            'average_word_length': np.mean(self.stats['word_lengths']) if self.stats['word_lengths'] else 0,
            'min_word_length': min(self.stats['word_lengths']) if self.stats['word_lengths'] else 0,
            'max_word_length': max(self.stats['word_lengths']) if self.stats['word_lengths'] else 0,
            'average_generation_time': np.mean(self.stats['generation_times']) if self.stats['generation_times'] else 0,
        }

        if self.stats['difficulty_scores']:
            summary.update({
                'average_difficulty': np.mean(self.stats['difficulty_scores']),
                'min_difficulty': min(self.stats['difficulty_scores']),
                'max_difficulty': max(self.stats['difficulty_scores'])
            })

        return summary

    def calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash of image for duplicate detection"""

        gray = image.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.array(gray).flatten()
        avg = pixels.mean()
        diff = pixels > avg
        return hashlib.md5(diff.tobytes()).hexdigest()

    def validate_contrast(self, image: Image.Image) -> float:
        """Calculate contrast ratio for readability validation"""
        img_array = np.array(image.convert('L'))
        return float(img_array.std())

    def add_metadata_watermark(self, image: Image.Image, metadata: Dict) -> Image.Image:
        """Add invisible metadata watermark for tracking (optional)"""

        return image