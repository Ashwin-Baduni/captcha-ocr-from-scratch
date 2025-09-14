"""
Comprehensive visualization utilities for all tasks
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from collections import Counter
import random

class CaptchaVisualizer:
    """Main visualizer for all CAPTCHA tasks"""

    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def visualize_task0_generation(self,
                                  easy_dir: Path,
                                  hard_dir: Path,
                                  bonus_dir: Path,
                                  num_samples: int = 12) -> Path:
        """Create comprehensive visualization for Task 0 dataset generation"""

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, num_samples//3, figure=fig, hspace=0.3, wspace=0.1)

        fig.suptitle('Task 0: CAPTCHA Dataset Generation Samples', fontsize=20, fontweight='bold')

        datasets = [
            ('Easy', easy_dir, 0),
            ('Hard', hard_dir, 1),
            ('Bonus', bonus_dir, 2)
        ]

        for dataset_name, dataset_dir, row_idx in datasets:

            images = list(dataset_dir.glob("*.png"))[:num_samples]

            for col_idx, img_path in enumerate(images[:num_samples//3]):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')

                word = img_path.stem.split('_')[-1]
                if row_idx == 0 and col_idx == 0:
                    ax.set_ylabel(dataset_name, fontsize=14, fontweight='bold')
                ax.set_title(f'"{word}"', fontsize=10)

        stats_row = 3

        ax_dist = fig.add_subplot(gs[stats_row, :num_samples//3])
        self._plot_word_distribution(easy_dir, hard_dir, bonus_dir, ax_dist)

        output_path = self.output_dir / "task0_generation_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Task 0 visualization saved to: {output_path}")
        return output_path

    def _plot_word_distribution(self, easy_dir, hard_dir, bonus_dir, ax):
        """Plot word frequency distribution"""
        all_words = []

        for dataset_dir in [easy_dir, hard_dir, bonus_dir]:
            for img_path in dataset_dir.glob("*.png"):
                word = img_path.stem.split('_')[-1]
                all_words.append(word)

        word_counts = Counter(all_words)
        top_words = dict(word_counts.most_common(15))

        ax.bar(range(len(top_words)), list(top_words.values()))
        ax.set_xticks(range(len(top_words)))
        ax.set_xticklabels(list(top_words.keys()), rotation=45, ha='right')
        ax.set_title('Top 15 Most Frequent Words', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    def visualize_task1_training(self,
                                 training_logs: Dict,
                                 model_performance: Dict) -> Path:
        """Create training visualization for Task 1"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Task 1: Classification Model Training Results', fontsize=20, fontweight='bold')

        datasets = ['easy', 'hard', 'bonus']

        for idx, dataset in enumerate(datasets):
            if dataset not in training_logs:
                continue

            log = training_logs[dataset]

            ax_loss = axes[0, idx]
            epochs = range(1, len(log['train_loss']) + 1)
            ax_loss.plot(epochs, log['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax_loss.plot(epochs, log['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax_loss.set_title(f'{dataset.capitalize()} - Loss', fontweight='bold')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            ax_acc = axes[1, idx]
            ax_acc.plot(epochs, log['train_acc'], 'g-', label='Train Acc', linewidth=2)
            ax_acc.plot(epochs, log['val_acc'], 'orange', label='Val Acc', linewidth=2)
            ax_acc.set_title(f'{dataset.capitalize()} - Accuracy', fontweight='bold')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy (%)')
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

            final_acc = log['val_acc'][-1] if log['val_acc'] else 0
            ax_acc.text(0.5, 0.95, f'Final: {final_acc:.1f}%',
                       transform=ax_acc.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = self.output_dir / "task1_training_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Task 1 visualization saved to: {output_path}")
        return output_path

    def visualize_task2_ocr(self,
                            ocr_results: Dict) -> Path:
        """Create OCR results visualization for Task 2"""

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Task 2: CAPTCHA OCR Text Extraction Results', fontsize=20, fontweight='bold')

        ax_cer = fig.add_subplot(gs[0, :])
        datasets = list(ocr_results.keys())
        cer_rates = [ocr_results[d].get('cer', 0) for d in datasets]
        colors = ['green', 'orange', 'red']

        bars = ax_cer.bar(datasets, cer_rates, color=colors, alpha=0.7)
        ax_cer.set_title('Character Error Rates (Lower is Better)', fontweight='bold')
        ax_cer.set_ylabel('CER (%)')

        for bar, rate in zip(bars, cer_rates):
            height = bar.get_height()
            ax_cer.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom')

        for idx, dataset in enumerate(datasets):
            ax_wer = fig.add_subplot(gs[1, idx])
            wer = ocr_results[dataset].get('wer', 0)
            ax_wer.bar(['WER'], [wer], color=colors[idx], alpha=0.7)
            ax_wer.set_title(f'{dataset.capitalize()} Word Error Rate', fontweight='bold')
            ax_wer.set_ylabel('WER (%)')
            ax_wer.set_ylim(0, 100)
            ax_wer.text(0, wer, f'{wer:.1f}%', ha='center', va='bottom')

        ax_metrics = fig.add_subplot(gs[2, :])
        metrics_text = "\n".join([
            f"{dataset.capitalize()}: CER={ocr_results[dataset].get('cer', 0):.1f}%, "
            f"WER={ocr_results[dataset].get('wer', 0):.1f}%, "
            f"Exact Match={ocr_results[dataset].get('exact_match', 0):.1f}%"
            for dataset in datasets
        ])
        ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center',
                       fontsize=12, family='monospace')
        ax_metrics.set_title('OCR Performance Summary', fontweight='bold')
        ax_metrics.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / "task2_ocr_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Task 2 OCR visualization saved to: {output_path}")
        return output_path

    def visualize_task3_bonus(self,
                             bonus_results: Dict,
                             bonus_dir: Path,
                             num_samples: int = 8) -> Path:
        """Create bonus set results visualization for Task 3"""

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Task 3: Bonus Set with Conditional Rendering', fontsize=20, fontweight='bold')

        ax_cond = fig.add_subplot(gs[0, :])
        conditions = ['green', 'red']
        condition_labels = ['Normal (Green)', 'Reversed (Red)']
        accuracies = [bonus_results.get(f'{cond}_accuracy', 0) for cond in conditions]
        colors_map = {'green': 'green', 'red': 'red'}
        bar_colors = [colors_map[c] for c in conditions]

        bars = ax_cond.bar(condition_labels, accuracies, color=bar_colors, alpha=0.7)
        ax_cond.set_title('OCR Accuracy by Background Condition', fontweight='bold')
        ax_cond.set_ylabel('Accuracy (%)')
        ax_cond.set_ylim(0, 100)

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax_cond.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.1f}%', ha='center', va='bottom')

        for idx, (cond, label) in enumerate(zip(conditions, condition_labels)):
            ax_sample = fig.add_subplot(gs[1, idx])

            sample_images = [p for p in bonus_dir.glob("*.png") if cond in str(p)][:1]
            if sample_images:
                img = Image.open(sample_images[0])
                ax_sample.imshow(img)
                ax_sample.set_title(f'{label}', fontsize=10, fontweight='bold')
                ax_sample.axis('off')
            else:
                ax_sample.text(0.5, 0.5, f'No {label}\nsamples', ha='center', va='center')
                ax_sample.axis('off')

        ax_overall = fig.add_subplot(gs[2, :])
        overall_text = f"""
        Overall Performance:
        • Average CER: {bonus_results.get('overall_cer', 0):.1f}%
        • Average WER: {bonus_results.get('overall_wer', 0):.1f}%
        • Exact Match Rate: {bonus_results.get('exact_match', 0):.1f}%

        Conditional Rendering (ONLY 2 conditions):
        • Green Background: Normal text - baseline OCR performance
        • Red Background: Text displayed reversed but labeled normally

        Key Challenge: Model must learn to reverse the visual pattern for red backgrounds
        """
        ax_overall.text(0.5, 0.5, overall_text, ha='center', va='center',
                       fontsize=10, family='monospace')
        ax_overall.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / "task3_bonus_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Task 3 bonus visualization saved to: {output_path}")
        return output_path

    def create_final_summary(self, all_results: Dict) -> Path:
        """Create a comprehensive summary visualization"""

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('CAPTCHA OCR Project - Complete Results Summary',
                    fontsize=24, fontweight='bold')

        ax_data = fig.add_subplot(gs[0, 0:2])
        datasets = ['Easy', 'Hard', 'Bonus']
        sizes = [all_results.get(f'{d.lower()}_count', 0) for d in datasets]
        colors = plt.cm.Set3(range(len(datasets)))

        bars = ax_data.bar(datasets, sizes, color=colors)
        ax_data.set_title('Dataset Sizes', fontweight='bold', fontsize=14)
        ax_data.set_ylabel('Number of Images')

        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax_data.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(size)}', ha='center', va='bottom')

        ax_acc = fig.add_subplot(gs[0, 2:4])
        models = ['Easy', 'Hard', 'Bonus']
        accuracies = [all_results.get(f'{m.lower()}_accuracy', 0) for m in models]
        colors = ['green', 'orange', 'red']

        bars = ax_acc.bar(models, accuracies, color=colors, alpha=0.7)
        ax_acc.set_title('Model Accuracies', fontweight='bold', fontsize=14)
        ax_acc.set_ylabel('Validation Accuracy (%)')
        ax_acc.set_ylim(0, 100)

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom')

        ax_ocr = fig.add_subplot(gs[1, :2])
        cer_rates = [all_results.get(f'{m.lower()}_cer', 0) for m in models]

        bars = ax_ocr.bar(models, cer_rates, color=['darkgreen', 'darkorange', 'darkred'], alpha=0.7)
        ax_ocr.set_title('OCR Character Error Rates', fontweight='bold', fontsize=14)
        ax_ocr.set_ylabel('CER (%)')
        ax_ocr.set_ylim(0, max(cer_rates) * 1.2 if cer_rates else 10)

        for bar, rate in zip(bars, cer_rates):
            height = bar.get_height()
            ax_ocr.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom')

        ax_bonus = fig.add_subplot(gs[1, 2:])
        conditions = ['Normal', 'Reversed']
        cond_perfs = [all_results.get(f'{c.lower()}_performance', 0) for c in ['green', 'red']]

        if any(cond_perfs):
            colors_cond = ['green', 'red']
            bars = ax_bonus.bar(conditions, cond_perfs, color=colors_cond, alpha=0.7)
            ax_bonus.set_title('Bonus Set Performance by Condition', fontweight='bold', fontsize=14)
            ax_bonus.set_ylabel('Accuracy (%)')
            ax_bonus.set_ylim(0, 100)

            for bar, perf in zip(bars, cond_perfs):
                height = bar.get_height()
                ax_bonus.text(bar.get_x() + bar.get_width()/2., height,
                            f'{perf:.1f}%', ha='center', va='bottom')

        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')

        summary_text = f"""
        PIPELINE EXECUTION SUMMARY
        ════════════════════════════════════════════════════════════════════

        ✓ Task 0: Generated {sum(sizes)} training images across 3 datasets
        ✓ Task 1: Trained 3 classification models with {np.mean(accuracies):.1f}% average accuracy
        ✓ Task 2: OCR text extraction with {np.mean(cer_rates):.1f}% average CER
        ✓ Task 3: Bonus set handling with conditional rendering

        Key Insights:
        • Easy CAPTCHAs: High accuracy ({accuracies[0]:.1f}%) with low CER ({cer_rates[0]:.1f}%)
        • Hard CAPTCHAs: Challenging ({accuracies[1]:.1f}% acc, {cer_rates[1]:.1f}% CER)
        • Bonus CAPTCHAs: Conditional rendering affects OCR performance
        • Sequence-to-sequence models handle variable-length text extraction
        """

        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                       fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.tight_layout()

        output_path = self.output_dir / "final_summary.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Final summary visualization saved to: {output_path}")
        return output_path