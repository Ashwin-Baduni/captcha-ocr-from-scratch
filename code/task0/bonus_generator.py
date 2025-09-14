"""
Bonus Set Generator - Conditional CAPTCHA images with special rules
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, List
import random
import numpy as np
from task0.hard_generator import HardGenerator
from task0.effects import CaptchaEffects
from utils import config as settings

class BonusGenerator(HardGenerator):
    """Generator for bonus CAPTCHA images with conditional transformations"""

    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.dataset_type = 'bonus'
        self.output_dir = settings.BONUS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.conditions = {
            'green': {
                'color': settings.COLORS['bonus']['green_bg'],
                'rule': 'normal',
                'description': 'Text rendered normally'
            },
            'red': {
                'color': settings.COLORS['bonus']['red_bg'],
                'rule': 'reverse',
                'description': 'Text rendered in reverse, but label stays normal'
            }
        }

    def generate_image(self, text: str, index: int) -> Tuple[Image.Image, Dict]:
        """
        Generate a bonus CAPTCHA with conditional logic

        Args:
            text: The text to render (this is the label, not necessarily what's displayed)
            index: Index of the image

        Returns:
            Tuple of (PIL Image, metadata dict)
        """

        condition_name = random.choice(list(self.conditions.keys()))
        condition = self.conditions[condition_name]

        image = self.create_conditional_background(condition['color'])

        display_text = self.apply_condition_rule(text, condition['rule'])

        display_text = self.apply_random_capitalization(display_text)

        font_path = self.get_random_font()
        font_size = self.get_random_font_size()

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(image)
        base_x, base_y = self.calculate_text_position(draw, display_text, font)

        image = self.render_complex_text(image, display_text, font, base_x, base_y)

        effects_applied = [f'condition_{condition_name}']
        image, additional_effects = self.apply_lighter_distortions(image, display_text)
        effects_applied.extend(additional_effects)

        if random.random() > 0.8:
            num_lines = 1 if condition_name == 'red' else 1
            image = self.effects.add_security_lines(image, num_lines=num_lines)
            effects_applied.append('security_lines')

        difficulty_score = self.calculate_bonus_difficulty(image, text, effects_applied, condition_name)

        metadata = {
            'difficulty_score': difficulty_score,
            'font_path': font_path,
            'font_size': font_size,
            'text_position': (base_x, base_y),
            'original_text': text,
            'display_text': display_text,
            'condition': condition_name,
            'condition_rule': condition['rule'],
            'condition_description': condition['description'],
            'background_color': condition['color'],
            'effects_applied': effects_applied,
            'capitalization_pattern': self.get_cap_pattern(display_text),
            'transformation_applied': self.get_transformation_description(text, display_text, condition['rule']),
            'image_hash': self.calculate_image_hash(image),
            'contrast_ratio': self.validate_contrast(image),
            'noise_level': self.estimate_noise_level(image)
        }

        return image, metadata

    def create_conditional_background(self, bg_color: Tuple[int, int, int]) -> Image.Image:
        """Create background with specific color and texture"""
        width, height = settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT

        image = Image.new('RGB', (width, height), color=bg_color)

        texture_overlay = Image.new('RGB', (width, height), color=bg_color)

        img_array = np.array(texture_overlay)
        noise = np.random.normal(0, 5, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        texture_overlay = Image.fromarray(noisy_img)

        image = Image.blend(image, texture_overlay, 0.3)

        if random.random() > 0.4:
            if bg_color == self.conditions['green']['color']:

                image = self.effects.add_dots_pattern(image, density=0.015)
            elif bg_color == self.conditions['red']['color']:

                image = self.effects.add_grid_pattern(image, spacing=25, alpha=0.2)

        return image

    def apply_condition_rule(self, text: str, rule: str) -> str:
        """Apply conditional transformation based on background color

        ONLY TWO RULES:
        - normal: text rendered as-is (green background)
        - reverse: text rendered backwards (red background)
        """
        if rule == 'normal':
            return text
        elif rule == 'reverse':
            return text[::-1]
        else:
            return text

    def get_transformation_description(self, original: str, display: str, rule: str) -> str:
        """Get description of transformation applied"""
        if rule == 'normal':
            return "No transformation (green background)"
        elif rule == 'reverse':
            return f"Reversed (red background): '{original}' -> '{display}'"
        return "Unknown transformation"

    def apply_lighter_distortions(self, image: Image.Image, text: str) -> Tuple[Image.Image, List[str]]:
        """Apply lighter distortions for better readability"""
        effects_applied = []

        if random.random() > 0.85:
            alpha = random.uniform(8, 12)
            sigma = random.uniform(2, 3)
            image = self.effects.elastic_distortion(image, alpha, sigma)
            effects_applied.append(f'elastic_α{alpha:.0f}_σ{sigma:.0f}')

        if random.random() > 0.7:
            amplitude = random.uniform(1, 2)
            frequency = random.uniform(0.02, 0.03)
            image = self.effects.add_sine_wave_distortion(image, amplitude, frequency)
            effects_applied.append(f'sine_a{amplitude:.1f}_f{frequency:.2f}')

        if random.random() > 0.7:
            image = self.effects.random_rotation(image, max_angle=random.uniform(1, 2))
            effects_applied.append('rotation')

        if random.random() > 0.6:
            image = self.effects.add_gaussian_noise(image, std=random.uniform(0.003, 0.008))
            effects_applied.append('gaussian_noise')

        return image, effects_applied

    def calculate_bonus_difficulty(self, image: Image.Image, text: str,
                                  effects: List[str], condition: str) -> float:
        """
        Calculate difficulty score for bonus set

        Returns:
            Float between 0.6 and 1.0 for bonus set (harder than regular hard set)
        """

        base_score = super().calculate_difficulty(image, text, effects)

        condition_bonus = {
            'green': 0.0,
            'red': 0.2
        }

        bonus = condition_bonus.get(condition, 0.05)

        total_score = base_score + bonus

        return min(max(total_score, 0.6), 1.0)

    def generate_balanced_dataset(self, num_images: int = None) -> Dict:
        """
        Generate dataset with balanced distribution of conditions

        Args:
            num_images: Total number of images to generate

        Returns:
            Dictionary containing generation statistics
        """
        num_images = num_images or settings.DATASET_SIZE['bonus']
        word_list = self.get_word_list(num_images)

        conditions_cycle = []
        condition_names = list(self.conditions.keys())
        while len(conditions_cycle) < num_images:
            random.shuffle(condition_names)
            conditions_cycle.extend(condition_names)

        print(f"Generating {num_images} images for bonus set with conditional logic...")

        from tqdm import tqdm
        import time

        condition_counts = {name: 0 for name in condition_names}

        for idx in tqdm(range(num_images)):
            start_time = time.time()

            word = word_list[idx % len(word_list)]

            forced_condition = conditions_cycle[idx]
            original_choice = random.choice
            random.choice = lambda x: forced_condition if x == list(self.conditions.keys()) else original_choice(x)

            image, metadata = self.generate_image(word, idx)

            random.choice = original_choice

            condition_counts[metadata['condition']] += 1

            filename = f"{self.dataset_type}_{idx:04d}_{metadata['condition']}_{word}.png"
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
            self.stats['difficulty_scores'].append(metadata['difficulty_score'])
            self.stats['generation_times'].append(time.time() - start_time)

        self.metadata['condition_distribution'] = condition_counts

        self.save_metadata()

        summary = self.get_generation_summary()
        summary['condition_distribution'] = condition_counts

        return summary

def main():
    """Test the Bonus Generator"""
    generator = BonusGenerator()

    test_words = ['hello', 'world', 'python', 'captcha', 'bonus']

    print("Generating test images for Bonus Set (all conditions)...")

    for condition_name in generator.conditions.keys():
        word = random.choice(test_words)

        original_choice = random.choice
        random.choice = lambda x: condition_name if x == list(generator.conditions.keys()) else original_choice(x)

        image, metadata = generator.generate_image(word, 0)

        random.choice = original_choice

        test_dir = Path("test_output/bonus")
        test_dir.mkdir(parents=True, exist_ok=True)
        image.save(test_dir / f"test_{condition_name}_{word}.png")

        print(f"Generated [{condition_name}]: {word}")
        print(f"  Display: {metadata['display_text']}")
        print(f"  Rule: {metadata['condition_rule']}")
        print(f"  Difficulty: {metadata['difficulty_score']:.3f}")
        print(f"  Transformation: {metadata['transformation_applied']}")
        print()

    print("Generating full Bonus dataset with balanced conditions...")
    stats = generator.generate_balanced_dataset(num_images=100)

    print("\nGeneration Summary:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()