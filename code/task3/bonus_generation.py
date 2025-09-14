"""
Task 3: Bonus Set Text Extraction
Handles the bonus dataset where text may be rendered forward or in reverse based on background color
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from task2.generation import CaptchaOCR, CTCOCRModel, create_char_vocabulary, decode_sequence
from task2.dataset_seq2seq import CaptchaSeq2SeqDataset
from task2.train_generation import OCRTrainer

class BonusOCRDataset(CaptchaSeq2SeqDataset):
    """
    Special dataset for bonus set that handles reversed text
    The key insight: the label is always the correct forward text,
    but the display might be reversed (for red backgrounds)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = 'bonus'

        self.display_metadata = self._load_display_metadata()

    def _load_display_metadata(self) -> Dict:
        """Load metadata about display conditions"""
        metadata = {}

        if self.data_dir.name == 'bonus':
            base_data_dir = self.data_dir.parent
        else:
            base_data_dir = self.data_dir

        metadata_file = base_data_dir / 'metadata' / 'bonus_metadata.json'

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                for img_meta in metadata_dict.get('images', []):
                    filename = img_meta.get('filename', '')
                    metadata[filename] = {
                        'condition': img_meta.get('condition', 'green'),
                        'display_text': img_meta.get('display_text', ''),
                        'original_text': img_meta.get('original_text', ''),
                        'is_reversed': img_meta.get('condition') == 'red'
                    }

        return metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, Dict]:
        """
        Get a single sample with metadata about display condition

        Returns:
            image: The CAPTCHA image
            sequence: The target sequence (always forward text)
            text: The correct text label
            metadata: Information about display condition
        """
        img_path = self.images[idx]
        text = self.texts[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sequence = self.text_to_sequence(text)
        sequence_tensor = torch.LongTensor(sequence)

        filename = Path(img_path).name
        meta = self.display_metadata.get(filename, {
            'condition': 'unknown',
            'is_reversed': False
        })

        return image, sequence_tensor, text, meta

class BonusOCRTrainer(OCRTrainer):
    """
    Specialized trainer for bonus set that handles conditional rendering
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = 'bonus'

        self.condition_metrics = {
            'green': {'correct': 0, 'total': 0},
            'red': {'correct': 0, 'total': 0}
        }

    def validate_with_conditions(self, val_loader) -> Dict:
        """
        Validate model with tracking of performance on different background conditions
        """
        self.model.eval()

        results = {
            'overall': {'cer': 0, 'wer': 0, 'exact_match': 0, 'count': 0},
            'by_condition': {
                'green': {'cer': 0, 'wer': 0, 'exact_match': 0, 'count': 0},
                'red': {'cer': 0, 'wer': 0, 'exact_match': 0, 'count': 0}
            }
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation with conditions'):

                if len(batch) == 3:
                    images, sequences, texts = batch
                    metadata = [{}] * len(texts)
                else:
                    images, sequences, texts, metadata = batch
                images = images.to(self.device)

                if self.model_type == 'seq2seq':
                    outputs, _ = self.model.generate(images)
                    pred_sequences = outputs.argmax(dim=-1)
                else:

                    outputs = self.model(images)
                    pred_sequences = outputs.argmax(dim=-1).permute(1, 0)

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
                    condition = metadata[i].get('condition', 'unknown')

                    from task2.generation import calculate_cer, calculate_wer
                    cer = calculate_cer(pred_text, target_text)
                    wer = calculate_wer(pred_text, target_text)
                    exact_match = 1 if pred_text == target_text else 0

                    results['overall']['cer'] += cer
                    results['overall']['wer'] += wer
                    results['overall']['exact_match'] += exact_match
                    results['overall']['count'] += 1

                    if condition in results['by_condition']:
                        results['by_condition'][condition]['cer'] += cer
                        results['by_condition'][condition]['wer'] += wer
                        results['by_condition'][condition]['exact_match'] += exact_match
                        results['by_condition'][condition]['count'] += 1

        for key in ['overall'] + list(results['by_condition'].keys()):
            if key == 'overall':
                metrics = results['overall']
            else:
                metrics = results['by_condition'][key]

            if metrics['count'] > 0:
                metrics['cer'] /= metrics['count']
                metrics['wer'] /= metrics['count']
                metrics['exact_match'] /= metrics['count']

        return results

    def generate_bonus_report(self, val_loader) -> Dict:
        """
        Generate detailed report for bonus set performance
        """
        print("\nGenerating detailed bonus set analysis...")

        results = self.validate_with_conditions(val_loader)

        report = {
            'model_type': self.model_type,
            'dataset': 'bonus',
            'overall_metrics': {
                'cer': results['overall']['cer'],
                'wer': results['overall']['wer'],
                'exact_match_rate': results['overall']['exact_match']
            },
            'condition_specific': {},
            'insights': []
        }

        for condition, metrics in results['by_condition'].items():
            if metrics['count'] > 0:
                report['condition_specific'][condition] = {
                    'samples': metrics['count'],
                    'cer': metrics['cer'],
                    'wer': metrics['wer'],
                    'exact_match_rate': metrics['exact_match']
                }

        if report['condition_specific'].get('red', {}).get('cer', 1) > \
           report['condition_specific'].get('green', {}).get('cer', 1):
            report['insights'].append(
                "Model struggles more with reversed text (red background) as expected"
            )

        if report['overall_metrics']['cer'] > 0.3:
            report['insights'].append(
                "Conditional rendering (especially reversed text) significantly impacts OCR performance"
            )

        report_path = self.report_dir / f'bonus_{self.model_type}_analysis.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*60)
        print("BONUS SET ANALYSIS")
        print("="*60)

        print(f"\nOverall Performance:")
        print(f"  CER: {report['overall_metrics']['cer']:.4f}")
        print(f"  WER: {report['overall_metrics']['wer']:.4f}")
        print(f"  Exact Match: {report['overall_metrics']['exact_match_rate']:.2%}")

        print(f"\nPerformance by Background Condition:")
        for condition, metrics in report['condition_specific'].items():
            print(f"\n  {condition.upper()} Background:")
            print(f"    Samples: {metrics['samples']}")
            print(f"    CER: {metrics['cer']:.4f}")
            print(f"    Exact Match: {metrics['exact_match_rate']:.2%}")

        if report['insights']:
            print(f"\nKey Insights:")
            for insight in report['insights']:
                print(f"  • {insight}")

        return report

def analyze_bonus_challenges():
    """
    Analyze the specific challenges of the bonus dataset
    """
    print("\n" + "="*60)
    print("ANALYZING BONUS SET CHALLENGES")
    print("="*60)

    challenges = {
        'green': {
            'transformation': 'Normal rendering',
            'difficulty': 'Baseline',
            'expected_performance': 'Best - standard OCR task'
        },
        'red': {
            'transformation': 'Text reversed (e.g., "hello" → "olleh")',
            'difficulty': 'High',
            'expected_performance': 'Model must learn to reverse the visual pattern'
        }
    }

    print("\nConditional Transformations:")
    for color, info in challenges.items():
        print(f"\n{color.upper()} Background:")
        print(f"  Transformation: {info['transformation']}")
        print(f"  Difficulty: {info['difficulty']}")
        print(f"  Challenge: {info['expected_performance']}")

    print("\n" + "-"*60)
    print("Training Strategy for Bonus Set:")
    print("  1. Model should learn that output is always the forward text")
    print("  2. Must handle visual variations based on background color")
    print("  3. Key challenge: Reversing the visual pattern for red backgrounds")
    print("  4. Recommend: Augment training with synthetic reversed examples")

    return challenges

def main():
    """Test bonus set functionality"""
    import argparse

    parser = argparse.ArgumentParser(description='Train OCR model on bonus dataset')
    parser.add_argument('--model', type=str, default='seq2seq',
                       choices=['seq2seq', 'ctc'],
                       help='Model type')
    parser.add_argument('--data-dir', type=str, default='../../data',
                       help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze challenges, don\'t train')

    args = parser.parse_args()

    if args.analyze_only:
        analyze_bonus_challenges()
        return

    from task2.dataset_seq2seq import create_data_loaders

    print("Loading bonus dataset...")
    train_loader, val_loader, vocab_size = create_data_loaders(
        data_dir=args.data_dir,
        dataset_type='bonus',
        batch_size=args.batch_size,
        use_ctc=(args.model == 'ctc')
    )

    trainer = BonusOCRTrainer(
        model_type=args.model,
        dataset_type='bonus',
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )

    print("\nTraining on bonus set with conditional rendering...")
    trainer.train(train_loader, val_loader)

    report = trainer.generate_bonus_report(val_loader)

    print("\n" + "="*60)
    print("BONUS SET TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()