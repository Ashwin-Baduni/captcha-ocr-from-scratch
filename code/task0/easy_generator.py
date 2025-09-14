"""
Easy Set Generator - Simple, clean CAPTCHA images
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict
import random
import numpy as np
from task0.base_generator import BaseGenerator
from utils import config as settings

class EasyGenerator(BaseGenerator):
    """Generator for easy CAPTCHA images with minimal complexity"""

    def __init__(self, seed: int = None):
        super().__init__('easy', seed)
        self.font_path = self.get_fixed_font()

    def get_fixed_font(self) -> str:
        """Get the fixed font for easy set"""

        possible_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def generate_image(self, text: str, index: int) -> Tuple[Image.Image, Dict]:
        """
        Generate a simple, clean CAPTCHA image

        Args:
            text: The text to render
            index: Index of the image

        Returns:
            Tuple of (PIL Image, metadata dict)
        """

        image = Image.new('RGB',
                         (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT),
                         color=settings.COLORS['easy']['bg'])

        draw = ImageDraw.Draw(image)

        font_size = random.randint(*settings.FONT_SIZE_RANGE['easy'])

        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:

            font = ImageFont.load_default()

        x, y = self.calculate_text_position(draw, text, font)

        x += random.randint(-2, 2)
        y += random.randint(-1, 1)

        draw.text((x, y), text,
                 fill=settings.COLORS['easy']['text'],
                 font=font)

        image = self.add_subtle_effects(image)

        difficulty_score = self.calculate_difficulty(image, text)

        metadata = {
            'difficulty_score': difficulty_score,
            'font_size': font_size,
            'text_position': (x, y),
            'effects_applied': ['subtle_gradient'],
            'capitalization': 'normal',
            'image_hash': self.calculate_image_hash(image),
            'contrast_ratio': self.validate_contrast(image)
        }

        return image, metadata

    def add_subtle_effects(self, image: Image.Image) -> Image.Image:
        """Add subtle effects to make it slightly more challenging"""

        if random.random() > 0.3:
            image = self.add_subtle_gradient(image)

        if random.random() > 0.4:
            image = self.add_light_noise(image)

        if random.random() > 0.6:
            image = self.add_thin_lines(image)

        if random.random() > 0.5:
            from PIL import ImageFilter
            image = image.filter(ImageFilter.SMOOTH_MORE)

        return image

    def add_subtle_gradient(self, image: Image.Image) -> Image.Image:
        """Add a very subtle gradient to the background"""
        width, height = image.size
        gradient = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(gradient)

        for y in range(height):
            gray_value = 255 - int(10 * (y / height))
            draw.line([(0, y), (width, y)], fill=(gray_value, gray_value, gray_value))

        return Image.blend(gradient, image, 0.90)

    def add_light_noise(self, image: Image.Image) -> Image.Image:
        """Add very light noise to the image"""
        import numpy as np
        img_array = np.array(image)

        noise = np.random.normal(0, 3, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)

    def add_thin_lines(self, image: Image.Image) -> Image.Image:
        """Add thin lines across the image"""
        draw = ImageDraw.Draw(image)
        width, height = image.size

        num_lines = random.randint(1, 2)
        for _ in range(num_lines):
            y = random.randint(10, height - 10)

            draw.line([(0, y), (width, y)], fill=(230, 230, 230), width=1)

        return image

    def calculate_difficulty(self, image: Image.Image, text: str) -> float:
        """
        Calculate difficulty score for easy set (will be low)

        Returns:
            Float between 0 and 1, where easy set scores around 0.1-0.3
        """
        base_score = 0.1

        length_factor = min(len(text) / 15, 0.1)

        contrast = self.validate_contrast(image)
        contrast_factor = max(0, 0.1 - contrast / 1000)

        total_score = base_score + length_factor + contrast_factor

        return min(total_score, 0.3)

def main():
    """Test the Easy Generator"""
    generator = EasyGenerator()

    test_words = ['cat', 'book', 'happy', 'computer', 'rainbow']

    print("Generating test images for Easy Set...")
    for i, word in enumerate(test_words):
        image, metadata = generator.generate_image(word, i)

        test_dir = Path("test_output/easy")
        test_dir.mkdir(parents=True, exist_ok=True)
        image.save(test_dir / f"test_{word}.png")

        print(f"Generated: {word}")
        print(f"  Difficulty: {metadata['difficulty_score']:.3f}")
        print(f"  Contrast: {metadata['contrast_ratio']:.2f}")

    print("\nGenerating full Easy dataset...")
    stats = generator.generate_dataset(num_images=100)

    print("\nGeneration Summary:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()