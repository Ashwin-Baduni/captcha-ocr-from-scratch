"""
Main script to generate all CAPTCHA datasets
"""
import sys
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import os

sys.path.append(str(Path(__file__).parent.parent))

from task0.easy_generator import EasyGenerator
from task0.hard_generator import HardGenerator
from task0.bonus_generator import BonusGenerator
from utils import config as settings

def generate_all_datasets(num_images: int = 100):
    """Generate all three dataset types"""

    print("="*60)
    print("CAPTCHA Dataset Generation - Precog Task")
    print("="*60)
    print(f"Generating {num_images} images per set")
    print(f"Seed: {settings.RANDOM_SEED}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    all_stats = {}
    total_start = time.time()

    print("\n[1/3] Generating Easy Set...")
    print("-"*40)
    easy_gen = EasyGenerator(seed=settings.RANDOM_SEED)
    easy_stats = easy_gen.generate_dataset(num_images)
    all_stats['easy'] = easy_stats
    print(f"✓ Easy Set complete: {easy_stats['total_images']} images")

    print("\n[2/3] Generating Hard Set...")
    print("-"*40)
    hard_gen = HardGenerator(seed=settings.RANDOM_SEED)
    hard_stats = hard_gen.generate_dataset(num_images)
    all_stats['hard'] = hard_stats
    print(f"✓ Hard Set complete: {hard_stats['total_images']} images")

    print("\n[3/3] Generating Bonus Set...")
    print("-"*40)
    bonus_gen = BonusGenerator(seed=settings.RANDOM_SEED)
    bonus_stats = bonus_gen.generate_balanced_dataset(num_images)
    all_stats['bonus'] = bonus_stats
    print(f"✓ Bonus Set complete: {bonus_stats['total_images']} images")

    total_time = time.time() - total_start

    overall_stats = {
        'generation_date': datetime.now().isoformat(),
        'total_generation_time': total_time,
        'total_images': sum(s['total_images'] for s in all_stats.values()),
        'seed': settings.RANDOM_SEED,
        'dataset_stats': all_stats
    }

    stats_file = settings.DATA_DIR / 'generation_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)

    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Total images generated: {overall_stats['total_images']}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Statistics saved to: {stats_file}")

    print("\nDifficulty Score Ranges:")
    print("-"*40)
    for dataset_type, stats in all_stats.items():
        if 'average_difficulty' in stats:
            print(f"{dataset_type.capitalize():8s}: {stats.get('min_difficulty', 0):.3f} - {stats.get('max_difficulty', 0):.3f} (avg: {stats['average_difficulty']:.3f})")

    if 'condition_distribution' in all_stats.get('bonus', {}):
        print("\nBonus Set Condition Distribution:")
        print("-"*40)
        for condition, count in all_stats['bonus']['condition_distribution'].items():
            print(f"  {condition:8s}: {count} images")

    print("\nDataset directories:")
    print(f"  Easy:  {settings.EASY_DIR}")
    print(f"  Hard:  {settings.HARD_DIR}")
    print(f"  Bonus: {settings.BONUS_DIR}")

    return overall_stats

def validate_datasets():
    """Validate generated datasets"""
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)

    issues = []

    for dataset_type in ['easy', 'hard', 'bonus']:
        print(f"\nValidating {dataset_type} set...")

        dir_map = {
            'easy': settings.EASY_DIR,
            'hard': settings.HARD_DIR,
            'bonus': settings.BONUS_DIR
        }
        dataset_dir = dir_map[dataset_type]

        if not dataset_dir.exists():
            issues.append(f"{dataset_type}: Directory does not exist")
            continue

        image_files = list(dataset_dir.glob("*.png"))
        print(f"  Found {len(image_files)} images")

        metadata_file = settings.METADATA_DIR / f"{dataset_type}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  Metadata entries: {len(metadata['images'])}")

            if len(metadata['images']) != len(image_files):
                issues.append(f"{dataset_type}: Metadata count mismatch")
        else:
            issues.append(f"{dataset_type}: Metadata file missing")

    if issues:
        print("\n⚠ Validation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All validations passed!")

    return len(issues) == 0

def generate_samples(num_samples: int = 5):
    """Generate sample images for quick testing"""
    print("\n" + "="*60)
    print("GENERATING SAMPLE IMAGES")
    print("="*60)

    sample_dir = settings.DATA_DIR / "samples"
    sample_dir.mkdir(exist_ok=True)

    sample_words = ['test', 'sample', 'demo', 'captcha', 'example']

    generators = {
        'easy': EasyGenerator(),
        'hard': HardGenerator(),
        'bonus': BonusGenerator()
    }

    for dataset_type, generator in generators.items():
        print(f"\nGenerating {dataset_type} samples...")
        type_dir = sample_dir / dataset_type
        type_dir.mkdir(exist_ok=True)

        for i in range(min(num_samples, len(sample_words))):
            word = sample_words[i]
            image, metadata = generator.generate_image(word, i)

            filename = f"sample_{i:02d}_{word}.png"
            image.save(type_dir / filename)

            meta_file = type_dir / f"sample_{i:02d}_{word}.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  Created: {filename} (difficulty: {metadata.get('difficulty_score', 0):.3f})")

    print(f"\nSamples saved to: {sample_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate CAPTCHA datasets')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of images per dataset (default: 100)')
    parser.add_argument('--samples-only', action='store_true',
                       help='Generate only sample images for testing')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing datasets')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of datasets')

    args = parser.parse_args()

    if args.samples_only:
        generate_samples()
    elif args.validate:
        validate_datasets()
    elif args.visualize:
        print("Visualization will be implemented in the visualizer module")

    else:
        stats = generate_all_datasets(args.num_images)
        validate_datasets()

        print("\nGenerating sample images for inspection...")
        generate_samples(3)

if __name__ == "__main__":
    main()