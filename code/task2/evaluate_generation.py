"""
Evaluation script for Task 2: Text Generation/Extraction
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from task2.generation import (
    CaptchaOCR, CTCOCRModel,
    create_char_vocabulary, decode_sequence,
    calculate_cer, calculate_wer
)
from task2.dataset_seq2seq import create_data_loaders

def evaluate_model(model_path: str, dataset_type: str, data_dir: str,
                  num_samples: int = 100, model_type: str = 'seq2seq') -> Dict:
    """
    Evaluate a trained OCR model on text extraction

    Args:
        model_path: Path to saved model checkpoint
        dataset_type: 'easy', 'hard', or 'bonus'
        data_dir: Path to data directory
        num_samples: Number of samples to evaluate
        model_type: 'seq2seq' or 'ctc'

    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    vocab_size = checkpoint.get('vocab_size', 66)
    saved_model_type = checkpoint.get('model_type', model_type)

    if saved_model_type == 'seq2seq':
        model = CaptchaOCR(vocab_size)
    else:
        model = CTCOCRModel(vocab_size)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully ({saved_model_type} model with {vocab_size} vocab)")

    _, val_loader, _ = create_data_loaders(
        data_dir=data_dir,
        dataset_type=dataset_type,
        batch_size=1,
        use_ctc=(saved_model_type == 'ctc')
    )

    char_to_idx, idx_to_char = create_char_vocabulary()

    results = {
        'dataset': dataset_type,
        'model_type': saved_model_type,
        'num_samples': 0,
        'exact_matches': 0,
        'total_cer': 0,
        'total_wer': 0,
        'predictions': [],
        'errors': []
    }

    print(f"\nEvaluating on {dataset_type} dataset...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, total=min(num_samples, len(val_loader)))):
            if idx >= num_samples:
                break

            if saved_model_type == 'seq2seq':
                images, sequences, texts = batch
                images = images.to(device)

                outputs, attention = model.generate(images, max_length=20)
                pred_sequence = outputs.argmax(dim=-1).squeeze(0)
                pred_text = decode_sequence(pred_sequence, idx_to_char)

            else:
                images, targets, lengths, texts = batch
                images = images.to(device)

                outputs = model(images)
                pred_sequence = outputs.argmax(dim=0).squeeze(1)

                pred_seq = []
                prev = -1
                for idx_val in pred_sequence:
                    if idx_val != 3 and idx_val != prev:
                        pred_seq.append(idx_val.item())
                    prev = idx_val

                pred_text = decode_sequence(pred_seq, idx_to_char)

            target_text = texts[0]

            cer = calculate_cer(pred_text, target_text)
            wer = calculate_wer(pred_text, target_text)

            results['num_samples'] += 1
            results['total_cer'] += cer
            results['total_wer'] += wer

            if pred_text == target_text:
                results['exact_matches'] += 1

            pred_detail = {
                'target': target_text,
                'prediction': pred_text,
                'cer': cer,
                'wer': wer,
                'exact_match': pred_text == target_text
            }
            results['predictions'].append(pred_detail)

            if pred_text != target_text and len(results['errors']) < 10:
                results['errors'].append({
                    'target': target_text,
                    'predicted': pred_text,
                    'cer': cer
                })

    results['avg_cer'] = results['total_cer'] / max(results['num_samples'], 1)
    results['avg_wer'] = results['total_wer'] / max(results['num_samples'], 1)
    results['exact_match_rate'] = results['exact_matches'] / max(results['num_samples'], 1)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {dataset_type}")
    print(f"Model Type: {saved_model_type}")
    print(f"Samples Evaluated: {results['num_samples']}")
    print(f"\nMetrics:")
    print(f"  Character Error Rate (CER): {results['avg_cer']:.4f}")
    print(f"  Word Error Rate (WER): {results['avg_wer']:.4f}")
    print(f"  Exact Match Rate: {results['exact_match_rate']:.2%}")

    if results['errors']:
        print(f"\nSample Errors:")
        for err in results['errors'][:5]:
            print(f"  Target: '{err['target']}' → Predicted: '{err['predicted']}' (CER: {err['cer']:.2f})")

    output_dir = Path(model_path).parent.parent / 'reports'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'{dataset_type}_generation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate OCR model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['easy', 'hard', 'bonus'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, default='../../data',
                       help='Data directory')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--model-type', type=str, default='seq2seq',
                       choices=['seq2seq', 'ctc'],
                       help='Model type')

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        model_type=args.model_type
    )

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()