"""
Hard Set Generator - Complex CAPTCHA images with multiple distortions
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Dict, List
import random
import numpy as np
from task0.base_generator import BaseGenerator
from task0.effects import CaptchaEffects
from utils import config as settings

class HardGenerator(BaseGenerator):
    """Generator for hard CAPTCHA images with advanced distortions"""

    def __init__(self, seed: int = None):
        super().__init__('hard', seed)
        self.effects = CaptchaEffects()

    def generate_image(self, text: str, index: int) -> Tuple[Image.Image, Dict]:
        """
        Generate a complex CAPTCHA image with multiple distortions

        Args:
            text: The text to render
            index: Index of the image

        Returns:
            Tuple of (PIL Image, metadata dict)
        """

        image = self.create_noisy_background()
        draw = ImageDraw.Draw(image)

        rendered_text = self.apply_random_capitalization(text)

        font_path = self.get_random_font()
        font_size = self.get_random_font_size()

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        base_x, base_y = self.calculate_text_position(draw, rendered_text, font)

        image = self.render_complex_text(image, rendered_text, font, base_x, base_y)

        effects_applied = []
        image, effects_applied = self.apply_distortions(image, text)

        if random.random() > 0.7:
            image = self.effects.add_security_lines(image, num_lines=random.randint(1, 2))
            effects_applied.append('security_lines')

        difficulty_score = self.calculate_difficulty(image, text, effects_applied)

        metadata = {
            'difficulty_score': difficulty_score,
            'font_path': font_path,
            'font_size': font_size,
            'text_position': (base_x, base_y),
            'rendered_text': rendered_text,
            'original_text': text,
            'effects_applied': effects_applied,
            'capitalization_pattern': self.get_cap_pattern(rendered_text),
            'image_hash': self.calculate_image_hash(image),
            'contrast_ratio': self.validate_contrast(image),
            'noise_level': self.estimate_noise_level(image)
        }

        return image, metadata

    def create_noisy_background(self) -> Image.Image:
        """Create a background with noise and texture"""
        width, height = settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT

        bg_color = tuple([
            random.randint(*range_val)
            for range_val in settings.COLORS['hard']['bg_range']
        ])

        image = Image.new('RGB', (width, height), color=bg_color)

        texture_types = ['paper', 'canvas', None]
        texture_type = random.choice(texture_types)
        if texture_type:
            image = self.effects.add_texture_background(image, texture_type)

        pattern_choice = random.random()
        if pattern_choice < 0.3:
            image = self.effects.add_dots_pattern(image, density=random.uniform(0.01, 0.03))
        elif pattern_choice < 0.6:
            image = self.effects.add_grid_pattern(image, spacing=random.randint(15, 25))

        return image

    def apply_random_capitalization(self, text: str) -> str:
        """Apply random capitalization patterns"""
        patterns = [
            lambda t: t.upper(),
            lambda t: t.lower(),
            lambda t: t.title(),
            lambda t: ''.join(random.choice([c.upper(), c.lower()]) for c in t),
            lambda t: t[0].upper() + t[1:].lower() if t else t,
            lambda t: ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(t))
        ]

        pattern = random.choice(patterns)
        return pattern(text)

    def render_complex_text(self, image: Image.Image, text: str, font: ImageFont.FreeTypeFont,
                          base_x: int, base_y: int) -> Image.Image:
        """Render text with character-level variations"""
        draw = ImageDraw.Draw(image)
        current_x = base_x

        for i, char in enumerate(text):

            char_color = tuple([
                random.randint(*range_val)
                for range_val in settings.COLORS['hard']['text_range']
            ])

            char_y = base_y + random.randint(-3, 3)

            if random.random() > 0.85:

                char_img = Image.new('RGBA', (50, 50), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_draw.text((10, 10), char, fill=char_color + (255,), font=font)

                angle = random.uniform(-5, 5)
                char_img = char_img.rotate(angle, expand=True)

                image.paste(char_img, (current_x, char_y), char_img)

                bbox = draw.textbbox((0, 0), char, font=font)
                char_width = bbox[2] - bbox[0]
                current_x += char_width - random.randint(0, 3)
            else:

                draw.text((current_x, char_y), char, fill=char_color, font=font)

                bbox = draw.textbbox((0, 0), char, font=font)
                char_width = bbox[2] - bbox[0]
                current_x += char_width + random.randint(-2, 2)

        return image

    def apply_distortions(self, image: Image.Image, text: str) -> Tuple[Image.Image, List[str]]:
        """Apply multiple distortion effects"""
        effects_applied = []

        if random.random() > 0.7:
            alpha = random.uniform(10, 15)
            sigma = random.uniform(2, 3)
            image = self.effects.elastic_distortion(image, alpha, sigma)
            effects_applied.append(f'elastic_α{alpha:.0f}_σ{sigma:.0f}')

        if random.random() > 0.7:
            amplitude = random.uniform(2, 3)
            frequency = random.uniform(0.02, 0.04)
            image = self.effects.add_sine_wave_distortion(image, amplitude, frequency)
            effects_applied.append(f'sine_a{amplitude:.1f}_f{frequency:.2f}')

        if random.random() > 0.8:
            image = self.effects.perspective_transform(image, intensity=random.uniform(0.00005, 0.0001))
            effects_applied.append('perspective')

        if random.random() > 0.6:
            image = self.effects.random_rotation(image, max_angle=random.uniform(1, 3))
            effects_applied.append('rotation')

        if random.random() > 0.8:
            image = self.effects.random_shear(image, max_shear=random.uniform(0.05, 0.1))
            effects_applied.append('shear')

        if random.random() > 0.3:
            noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
            if noise_type == 'gaussian':
                image = self.effects.add_gaussian_noise(image, std=random.uniform(0.005, 0.01))
                effects_applied.append('gaussian_noise')
            elif noise_type == 'salt_pepper':
                image = self.effects.add_salt_pepper_noise(image, amount=random.uniform(0.002, 0.005))
                effects_applied.append('salt_pepper_noise')
            elif noise_type == 'speckle':
                image = self.effects.add_speckle_noise(image, variance=random.uniform(0.005, 0.01))
                effects_applied.append('speckle_noise')

        return image, effects_applied

    def get_cap_pattern(self, text: str) -> str:
        """Identify capitalization pattern"""
        if text.isupper():
            return 'UPPER'
        elif text.islower():
            return 'lower'
        elif text.istitle():
            return 'Title'
        elif all(c.isupper() if i % 2 == 0 else c.islower() for i, c in enumerate(text) if c.isalpha()):
            return 'Alternating'
        else:
            return 'Mixed'

    def estimate_noise_level(self, image: Image.Image) -> float:
        """Estimate the noise level in the image"""
        img_array = np.array(image.convert('L'))

        kernel_size = 3
        padded = np.pad(img_array, kernel_size // 2, mode='edge')
        local_var = np.zeros_like(img_array, dtype=float)

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                local_var[i, j] = np.var(window)

        return float(np.mean(local_var))

    def calculate_difficulty(self, image: Image.Image, text: str, effects: List[str]) -> float:
        """
        Calculate difficulty score for hard set

        Returns:
            Float between 0.5 and 1.0 for hard set
        """
        base_score = 0.5

        effects_factor = min(len(effects) * 0.05, 0.3)

        length_factor = min(len(text) / 10 * 0.1, 0.1)

        noise_factor = min(self.estimate_noise_level(image) / 1000, 0.1)

        contrast = self.validate_contrast(image)
        contrast_factor = max(0, 0.1 - contrast / 100)

        total_score = base_score + effects_factor + length_factor + noise_factor + contrast_factor

        return min(max(total_score, 0.5), 1.0)

def main():
    """Test the Hard Generator"""
    generator = HardGenerator()

    test_words = ['complex', 'DIFFICULT', 'ChAlLeNgE', 'distorted', 'CAPTCHA']

    print("Generating test images for Hard Set...")
    for i, word in enumerate(test_words):
        image, metadata = generator.generate_image(word, i)

        test_dir = Path("test_output/hard")
        test_dir.mkdir(parents=True, exist_ok=True)
        image.save(test_dir / f"test_{word}.png")

        print(f"Generated: {word} -> {metadata['rendered_text']}")
        print(f"  Difficulty: {metadata['difficulty_score']:.3f}")
        print(f"  Effects: {', '.join(metadata['effects_applied'])}")
        print(f"  Capitalization: {metadata['capitalization_pattern']}")

    print("\nGenerating full Hard dataset...")
    stats = generator.generate_dataset(num_images=100)

    print("\nGeneration Summary:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()