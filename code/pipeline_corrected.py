"""
Corrected CAPTCHA OCR Pipeline with proper Task 2 (Text Generation) and Task 3 (Bonus Set)
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

try:
    from utils.comprehensive_visualizer import CaptchaVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization module not available")

class CorrectedCaptchaPipeline:
    """Pipeline with corrected Task 2 (Text Generation) and Task 3 (Bonus Set)"""

    def __init__(self,
                 num_images: int = 1000,
                 tasks: List[int] = None,
                 timeout: int = 600,
                 visualize: bool = True,
                 base_dir: Path = None,
                 task1_epochs: List[int] = None,
                 task2_epochs: List[int] = None,
                 task3_epochs: int = None):
        """
        Initialize the corrected pipeline

        Args:
            num_images: Number of images per dataset
            tasks: List of tasks to run [0,1,2,3] or None for all
            timeout: Timeout per task in seconds
            visualize: Whether to generate visualizations
            base_dir: Base directory for the project
            task1_epochs: List of epochs for Task 1 [easy, hard, bonus] or None for defaults
            task2_epochs: List of epochs for Task 2 [easy, hard] or None for defaults
            task3_epochs: Epochs for Task 3 or None for default
        """
        self.num_images = num_images
        self.tasks = tasks or [0, 1, 2, 3]
        self.timeout = timeout
        self.visualize = visualize and VISUALIZATION_AVAILABLE

        # Set epoch configurations with defaults
        self.task1_epochs = task1_epochs or [10, 50, 30]  # easy, hard, bonus
        self.task2_epochs = task2_epochs or [20, 30]      # easy, hard
        self.task3_epochs = task3_epochs or 25             # bonus

        self.base_dir = base_dir or Path(__file__).parent.parent
        self.code_dir = self.base_dir / 'code'
        self.data_dir = self.base_dir / 'data'
        self.results_dir = self.base_dir / 'results'

        # Create all necessary directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'easy').mkdir(exist_ok=True)
        (self.data_dir / 'hard').mkdir(exist_ok=True)
        (self.data_dir / 'bonus').mkdir(exist_ok=True)
        (self.data_dir / 'metadata').mkdir(exist_ok=True)
        (self.data_dir / 'samples').mkdir(exist_ok=True)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        (self.results_dir / 'reports').mkdir(exist_ok=True)
        (self.results_dir / 'visualizations').mkdir(exist_ok=True)

        self.visualizer = CaptchaVisualizer(self.results_dir / 'visualizations') if self.visualize else None

        self.results = {}
        self.generation_results = {}
        self.training_results = {}
        self.extraction_results = {}
        self.bonus_results = {}

    def run_command(self, cmd: List[str], description: str,
                   capture_output: bool = True) -> Tuple[bool, str]:
        """Run a command with timeout and error handling"""
        print(f"\n{'='*60}")
        print(f"🔄 Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)

        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    cwd=self.code_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if result.returncode == 0:
                    print(f"✅ {description} completed successfully!")
                    return True, result.stdout
                else:
                    print(f"❌ {description} failed with error:")
                    print(result.stderr[:500])
                    return False, result.stderr
            else:

                result = subprocess.run(
                    cmd,
                    cwd=self.code_dir,
                    timeout=self.timeout
                )
                return result.returncode == 0, ""

        except subprocess.TimeoutExpired:
            print(f"⏱️ {description} timed out after {self.timeout} seconds")
            return False, "Timeout"
        except Exception as e:
            print(f"❌ Error running {description}: {e}")
            return False, str(e)

    def task0_generate_datasets(self) -> bool:
        """Task 0: Generate CAPTCHA datasets"""
        print("\n" + "="*80)
        print(" 🎨 TASK 0: DATASET GENERATION ".center(80, "="))
        print("="*80)

        print(f"\n🔄 Generating datasets with {self.num_images} images each...")

        cmd = [
            sys.executable, "-m", "task0.generate",
            "--num-images", str(self.num_images)
        ]

        success, output = self.run_command(cmd, "Dataset generation", capture_output=False)

        if success:

            self.generation_results = {
                'easy': self.num_images,
                'hard': self.num_images,
                'bonus': self.num_images
            }

            if self.visualizer:
                try:
                    print("\n🎨 Creating Task 0 Visualization...")
                    viz_path = self.visualizer.visualize_task0_generation(
                        self.data_dir / 'easy',
                        self.data_dir / 'hard',
                        self.data_dir / 'bonus'
                    )
                    print(f"✅ Visualization saved to: {viz_path}")
                except Exception as e:
                    print(f"⚠️ Visualization error: {e}")

            print("\n✅ Task 0: Dataset Generation completed successfully!")
            return True

        return False

    def task1_train_classifiers(self) -> bool:
        """Task 1: Train classification models (100 words)"""
        print("\n" + "="*80)
        print(" 🧠 TASK 1: CLASSIFICATION TRAINING ".center(80, "="))
        print("="*80)

        datasets = [
            ('easy', 'lightweight', self.task1_epochs[0]),
            ('hard', 'improved', self.task1_epochs[1]),
            ('bonus', 'improved', self.task1_epochs[2])
        ]

        all_success = True

        for dataset, model_type, epochs in datasets:
            print(f"\n🔄 Training {dataset.upper()} classifier...")
            print(f"  Model: {model_type}")
            print(f"  Epochs: {epochs}")

            cmd = [
                sys.executable, "-m", "task1.train",
                "--dataset", dataset,
                "--model", model_type,
                "--epochs", str(epochs),
                "--batch-size", "32",
                "--vocab-size", "100",
                "--data-dir", str(self.data_dir),
                "--output-dir", str(self.results_dir)
            ]

            success, output = self.run_command(
                cmd,
                f"Training {dataset} classifier",
                capture_output=False
            )

            if success:

                self.training_results[dataset] = {
                    'model': model_type,
                    'epochs': epochs,
                    'success': True
                }
            else:
                all_success = False
                self.training_results[dataset] = {
                    'model': model_type,
                    'epochs': epochs,
                    'success': False
                }

        if self.visualizer and self.training_results:
            try:
                print("\n🎨 Creating Task 1 Training Visualizations...")

                training_logs = {}
                model_performance = {}

                for dataset, info in self.training_results.items():
                    report_file = self.results_dir / 'reports' / f'{dataset}_{info["model"]}_report.json'
                    if report_file.exists():
                        with open(report_file) as f:
                            report = json.load(f)
                            training_logs[dataset] = {
                                'train_loss': report.get('train_losses', []),
                                'val_loss': report.get('val_losses', []),
                                'train_acc': report.get('train_accs', []),
                                'val_acc': report.get('val_accs', [])
                            }
                            model_performance[dataset] = {
                                'best_acc': report.get('best_val_acc', 0),
                                'final_loss': report.get('final_train_loss', 0)
                            }

                if training_logs:
                    viz_path = self.visualizer.visualize_task1_training(training_logs, model_performance)
                    print(f"✅ Task 1 training curves saved to: {viz_path}")
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")

        print("\n✅ Task 1: Classification Training completed!")
        return all_success

    def task2_text_extraction(self) -> bool:
        """Task 2: Text Generation/Extraction (OCR)"""
        print("\n" + "="*80)
        print(" 📝 TASK 2: TEXT EXTRACTION (OCR) ".center(80, "="))
        print("="*80)

        print("\n📚 Training sequence-to-sequence models for text extraction...")
        print("This task extracts the actual text from images (variable length)")

        datasets = [
            ('easy', 'seq2seq', self.task2_epochs[0]),
            ('hard', 'seq2seq', self.task2_epochs[1]),

        ]

        all_success = True

        for dataset, model_type, epochs in datasets:
            print(f"\n🔄 Training {dataset.upper()} OCR model...")
            print(f"  Model: {model_type}")
            print(f"  Output: Variable-length text sequences")

            cmd = [
                sys.executable, "-m", "task2.train_generation",
                "--dataset", dataset,
                "--model", model_type,
                "--epochs", str(epochs),
                "--batch-size", "32",
                "--lr", "0.001",
                "--data-dir", str(self.data_dir),
                "--output-dir", str(self.results_dir)
            ]

            success, output = self.run_command(
                cmd,
                f"Training {dataset} OCR model",
                capture_output=False
            )

            if not success:
                print(f"⚠️ Training failed for {dataset} dataset")
                all_success = False
                continue

            print(f"\n📊 Evaluating {dataset} OCR model...")

            model_path = self.results_dir / 'models' / f'{dataset}_{model_type}_best.pth'

            if not model_path.exists():
                print(f"⚠️ Model not found: {model_path}")
                continue

            eval_cmd = [
                sys.executable, "-m", "task2.evaluate_generation",
                "--model", str(model_path),
                "--dataset", dataset,
                "--data-dir", str(self.data_dir),
                "--num-samples", "50",
                "--model-type", model_type
            ]

            success, output = self.run_command(
                eval_cmd,
                f"Evaluating {dataset} OCR",
                capture_output=True
            )

            if success:

                try:
                    results_file = self.results_dir / 'reports' / f'{dataset}_generation_results.json'
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                            self.extraction_results[dataset] = {
                                'cer': results.get('avg_cer', 0),
                                'wer': results.get('avg_wer', 0),
                                'exact_match': results.get('exact_match_rate', 0)
                            }
                            print(f"  CER: {results.get('avg_cer', 0):.4f}")
                            print(f"  WER: {results.get('avg_wer', 0):.4f}")
                            print(f"  Exact Match: {results.get('exact_match_rate', 0):.2%}")
                except Exception as e:
                    print(f"⚠️ Could not parse results: {e}")

        if self.visualizer and self.extraction_results:
            try:
                print("\n🎨 Creating Task 2 OCR Visualizations...")

                self._create_ocr_visualization()

                viz_path = self.visualizer.visualize_task2_ocr(self.extraction_results)
                print(f"✅ Task 2 OCR metrics saved to: {viz_path}")

                self._create_sample_predictions_visualization()
                print(f"✅ Task 2 sample predictions visualization created")
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")

        print("\n✅ Task 2: Text Extraction completed!")
        return all_success

    def task3_bonus_generation(self) -> bool:
        """Task 3: Bonus Set with Conditional Rendering"""
        print("\n" + "="*80)
        print(" 🎯 TASK 3: BONUS SET GENERATION ".center(80, "="))
        print("="*80)

        print("\n🎯 Training on bonus set with conditional rendering...")
        print("Challenge: Extract correct text regardless of display variation")
        print("  • Green: Normal text")
        print("  • Red: Reversed display")

        print("\n📊 Analyzing bonus set challenges...")

        analyze_cmd = [
            sys.executable, "-m", "task3.bonus_generation",
            "--analyze-only"
        ]

        self.run_command(analyze_cmd, "Analyzing bonus challenges", capture_output=False)

        print("\n🔄 Training bonus OCR model...")

        train_cmd = [
            sys.executable, "-m", "task3.bonus_generation",
            "--model", "seq2seq",
            "--epochs", str(self.task3_epochs),
            "--batch-size", "32",
            "--data-dir", str(self.data_dir)
        ]

        success, output = self.run_command(
            train_cmd,
            "Training bonus OCR model",
            capture_output=False
        )

        if success:

            bonus_report = self.results_dir / 'reports' / 'bonus_seq2seq_analysis.json'
            if bonus_report.exists():
                with open(bonus_report, 'r') as f:
                    self.bonus_results = json.load(f)

                print("\n📈 Bonus Set Performance:")
                if 'overall_metrics' in self.bonus_results:
                    metrics = self.bonus_results['overall_metrics']
                    print(f"  Overall CER: {metrics.get('cer', 0):.4f}")
                    print(f"  Overall WER: {metrics.get('wer', 0):.4f}")
                    print(f"  Exact Match: {metrics.get('exact_match_rate', 0):.2%}")

                if 'condition_specific' in self.bonus_results:
                    print("\n  Performance by Condition:")
                    for condition, perf in self.bonus_results['condition_specific'].items():
                        print(f"    {condition}: CER={perf.get('cer', 0):.4f}")

        if self.visualizer and self.bonus_results:
            try:
                print("\n🎨 Creating Task 3 Bonus Visualizations...")
                viz_path = self.visualizer.visualize_task3_bonus(
                    self.bonus_results,
                    self.data_dir / 'bonus'
                )
                print(f"✅ Task 3 bonus visualization saved to: {viz_path}")

                self._create_bonus_condition_comparison()
                print(f"✅ Task 3 condition comparison visualization created")
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")

        print("\n✅ Task 3: Bonus Generation completed!")
        return success

    def _create_ocr_visualization(self):
        """Create custom visualization for OCR results"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Task 2: Text Extraction (OCR) Results', fontsize=16, fontweight='bold')

        ax1 = fig.add_subplot(gs[0, 0])
        datasets = list(self.extraction_results.keys())
        cers = [self.extraction_results[d].get('cer', 0) for d in datasets]
        bars = ax1.bar(datasets, cers, color=['green', 'orange', 'red'])
        ax1.set_title('Character Error Rate (CER)', fontweight='bold')
        ax1.set_ylabel('CER (lower is better)')
        ax1.set_ylim(0, 1)
        for bar, cer in zip(bars, cers):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{cer:.3f}', ha='center')

        ax2 = fig.add_subplot(gs[0, 1])
        wers = [self.extraction_results[d].get('wer', 0) for d in datasets]
        bars = ax2.bar(datasets, wers, color=['green', 'orange', 'red'])
        ax2.set_title('Word Error Rate (WER)', fontweight='bold')
        ax2.set_ylabel('WER (lower is better)')
        ax2.set_ylim(0, 1)
        for bar, wer in zip(bars, wers):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{wer:.3f}', ha='center')

        ax3 = fig.add_subplot(gs[1, :])
        exact_matches = [self.extraction_results[d].get('exact_match', 0) * 100 for d in datasets]
        bars = ax3.bar(datasets, exact_matches, color=['green', 'orange', 'red'])
        ax3.set_title('Exact Match Rate', fontweight='bold')
        ax3.set_ylabel('Exact Match %')
        ax3.set_ylim(0, 100)
        for bar, em in zip(bars, exact_matches):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{em:.1f}%', ha='center')

        fig.text(0.5, 0.02,
                'OCR models extract variable-length text from CAPTCHA images using seq2seq architecture',
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        viz_path = self.results_dir / 'visualizations' / 'task2_ocr_results.png'
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ OCR visualization saved to: {viz_path}")

    def _create_sample_predictions_visualization(self):
        """Create visualization showing sample OCR predictions vs ground truth"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from PIL import Image
        import random

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

        fig.suptitle('Task 2: OCR Sample Predictions', fontsize=18, fontweight='bold')

        sample_idx = 0
        for dataset_idx, dataset in enumerate(['easy', 'hard', 'bonus']):
            if dataset not in self.extraction_results:
                continue

            dataset_dir = self.data_dir / dataset
            results_file = self.results_dir / 'reports' / f'{dataset}_generation_results.json'

            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)

                predictions = results.get('predictions', [])[:4]

                for pred_idx, pred in enumerate(predictions):
                    if sample_idx >= 12:
                        break

                    row = sample_idx // 4
                    col = sample_idx % 4

                    ax = fig.add_subplot(gs[row, col])

                    img_files = list(dataset_dir.glob(f"*_{pred['target']}.png"))
                    if img_files:
                        img = Image.open(img_files[0])
                        ax.imshow(img)

                    ax.set_title(f"{dataset.capitalize()} Dataset", fontweight='bold', fontsize=10)
                    ax.text(0.5, -0.15, f"True: {pred['target']}",
                           transform=ax.transAxes, ha='center', fontsize=9, color='green')
                    ax.text(0.5, -0.25, f"Pred: {pred['prediction']}",
                           transform=ax.transAxes, ha='center', fontsize=9, color='red')
                    ax.text(0.5, -0.35, f"CER: {pred['cer']:.2f}",
                           transform=ax.transAxes, ha='center', fontsize=8)
                    ax.axis('off')

                    sample_idx += 1

        plt.tight_layout()
        viz_path = self.results_dir / 'visualizations' / 'task2_ocr_predictions.png'
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ OCR predictions visualization saved to: {viz_path}")

    def _create_bonus_condition_comparison(self):
        """Create visualization comparing green vs red condition performance"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from PIL import Image

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Task 3: Bonus Set Condition Comparison (Green vs Red)', fontsize=18, fontweight='bold')

        bonus_dir = self.data_dir / 'bonus'
        metadata_file = self.data_dir / 'metadata' / 'bonus_metadata.json'

        condition_samples = {'green': [], 'red': []}

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                for img_meta in metadata.get('images', []):
                    condition = img_meta.get('condition')
                    if condition in condition_samples and len(condition_samples[condition]) < 4:
                        condition_samples[condition].append(img_meta)

        for cond_idx, (condition, samples) in enumerate(condition_samples.items()):
            for sample_idx, sample in enumerate(samples[:4]):
                if sample_idx >= 4:
                    break

                row = cond_idx
                col = sample_idx % 3

                ax = fig.add_subplot(gs[row, col])

                img_path = bonus_dir / sample['filename']
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)

                    bg_color = 'lightgreen' if condition == 'green' else 'lightcoral'
                    ax.set_title(f"{condition.upper()}: {sample.get('display_text', '')}",
                               fontweight='bold', fontsize=10, bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
                    ax.text(0.5, -0.1, f"Label: {sample.get('original_text', '')}",
                           transform=ax.transAxes, ha='center', fontsize=9)

                ax.axis('off')

        ax_stats = fig.add_subplot(gs[2, :])

        if self.bonus_results and 'condition_specific' in self.bonus_results:
            conditions = ['green', 'red']
            colors = ['green', 'red']
            cers = []
            wers = []

            for cond in conditions:
                perf = self.bonus_results['condition_specific'].get(cond, {})
                cers.append(perf.get('cer', 0) * 100)
                wers.append(perf.get('wer', 0) * 100)

            x = range(len(conditions))
            width = 0.35

            bars1 = ax_stats.bar([i - width/2 for i in x], cers, width, label='CER', color=colors, alpha=0.7)
            bars2 = ax_stats.bar([i + width/2 for i in x], wers, width, label='WER', color=colors, alpha=0.5)

            ax_stats.set_xlabel('Background Condition')
            ax_stats.set_ylabel('Error Rate (%)')
            ax_stats.set_title('Performance by Condition', fontweight='bold')
            ax_stats.set_xticks(x)
            ax_stats.set_xticklabels([c.upper() for c in conditions])
            ax_stats.legend()

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax_stats.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        viz_path = self.results_dir / 'visualizations' / 'task3_condition_comparison.png'
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ Bonus condition comparison saved to: {viz_path}")

    def _create_comprehensive_summary_visualization(self):
        """Create comprehensive summary visualization of entire pipeline"""
        all_results = {
            'easy_count': self.generation_results.get('easy', 0),
            'hard_count': self.generation_results.get('hard', 0),
            'bonus_count': self.generation_results.get('bonus', 0)
        }

        for dataset in ['easy', 'hard', 'bonus']:
            if dataset in self.training_results:
                report_file = self.results_dir / 'reports' / f'{dataset}_{self.training_results[dataset]["model"]}_report.json'
                if report_file.exists():
                    with open(report_file) as f:
                        report = json.load(f)
                        all_results[f'{dataset}_accuracy'] = report.get('best_val_acc', 0)

            if dataset in self.extraction_results:
                all_results[f'{dataset}_cer'] = self.extraction_results[dataset].get('cer', 0) * 100
                all_results[f'{dataset}_wer'] = self.extraction_results[dataset].get('wer', 0) * 100

        if self.bonus_results and 'condition_specific' in self.bonus_results:
            for cond in ['green', 'red']:
                perf = self.bonus_results['condition_specific'].get(cond, {})
                all_results[f'{cond}_performance'] = (1 - perf.get('cer', 1)) * 100

        if self.visualizer:
            viz_path = self.visualizer.create_final_summary(all_results)
            print(f"✅ Comprehensive summary saved to: {viz_path}")

    def run_pipeline(self):
        """Run the complete corrected pipeline"""
        print("\n" + "="*80)
        print(" 🚀 CAPTCHA OCR PIPELINE ".center(80, "█"))
        print("="*80)

        print(f"\n⚙️ Configuration:")
        print(f"  • Images per dataset: {self.num_images}")
        print(f"  • Tasks to run: {self.tasks}")
        print(f"  • Timeout per task: {self.timeout}s")
        print(f"  • Visualizations: {'✅ Enabled' if self.visualize else '❌ Disabled'}")

        self.start_time = time.time()

        task_functions = {
            0: ('Dataset Generation', self.task0_generate_datasets),
            1: ('Classification Training', self.task1_train_classifiers),
            2: ('Text Extraction (OCR)', self.task2_text_extraction),
            3: ('Bonus Set Generation', self.task3_bonus_generation)
        }

        for task_num in self.tasks:
            if task_num in task_functions:
                task_name, task_func = task_functions[task_num]

                print(f"\n{'='*80}")
                print(f" ▶️ Starting Task {task_num}: {task_name} ".center(80, "="))
                print('='*80)

                success = task_func()

                if not success:
                    print(f"\n⚠️ Task {task_num} encountered issues")

                if task_num < max(self.tasks):
                    print("\n⏳ Pausing before next task...")
                    time.sleep(2)

        self.generate_final_summary()

        if self.visualizer:
            try:
                print("\n🎨 Creating Final Summary Visualization...")
                self._create_comprehensive_summary_visualization()
                print("✅ Final summary visualization created")
            except Exception as e:
                print(f"⚠️ Final visualization error: {e}")

        total_time = time.time() - self.start_time
        print(f"\n⏱️ Total pipeline execution time: {total_time:.2f} seconds")

        print("\n" + "="*80)
        print(" ✅ PIPELINE COMPLETE! ".center(80, "█"))
        print("="*80)

    def generate_final_summary(self):
        """Generate comprehensive final summary of all tasks with full details"""
        print("\n" + "="*80)
        print(" 📊 DETAILED FINAL SUMMARY ".center(80, "="))
        print("="*80)

        if 0 in self.tasks and self.generation_results:
            print("\n📦 Task 0 - Dataset Generation:")
            print("-" * 40)
            total_images = 0
            for dataset, count in self.generation_results.items():
                print(f"  • {dataset.upper()}: {count} images generated")
                total_images += count
            print(f"\n  📊 Total: {total_images} images across {len(self.generation_results)} datasets")

            stats_file = self.data_dir / 'generation_stats.json'
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                    print(f"  ⏱️ Generation time: {stats.get('total_time', 0):.2f} seconds")
                    if 'difficulty_ranges' in stats:
                        print("\n  📈 Difficulty Score Ranges:")
                        for dataset, ranges in stats['difficulty_ranges'].items():
                            print(f"    {dataset}: {ranges['min']:.3f} - {ranges['max']:.3f} (avg: {ranges['avg']:.3f})")
                    if 'bonus_conditions' in stats:
                        print("\n  🎨 Bonus Conditions Distribution:")
                        for condition, count in stats['bonus_conditions'].items():
                            print(f"    {condition}: {count} images")

        if 1 in self.tasks and self.training_results:
            print("\n🎯 Task 1 - Classification Training (100-word vocabulary):")
            print("-" * 40)
            for dataset, info in self.training_results.items():
                status = "✅" if info['success'] else "❌"
                print(f"\n  {dataset.upper()} Dataset:")
                print(f"    Model: {info['model']} {status}")

                report_file = self.results_dir / 'reports' / f'{dataset}_{info["model"]}_report.json'
                if report_file.exists():
                    with open(report_file) as f:
                        report = json.load(f)
                        print(f"    Best Val Accuracy: {report.get('best_val_acc', 0):.1f}%")
                        print(f"    Final Train Loss: {report.get('final_train_loss', 0):.4f}")
                        print(f"    Training Time: {report.get('training_time', 0):.1f}s")
                        print(f"    Total Epochs: {report.get('epochs_trained', 0)}")

        if 2 in self.tasks and self.extraction_results:
            print("\n📝 Task 2 - Text Extraction (OCR):")
            print("-" * 40)
            print("  🤖 Model: Sequence-to-sequence with attention mechanism")
            print("\n  Performance Metrics:")

            for dataset, metrics in self.extraction_results.items():
                print(f"\n  {dataset.upper()} Dataset:")
                print(f"    Character Error Rate (CER): {metrics.get('cer', 1.0)*100:.1f}%")
                print(f"    Word Error Rate (WER): {metrics.get('wer', 1.0)*100:.1f}%")
                print(f"    Exact Match Rate: {metrics.get('exact_match', 0)*100:.1f}%")

                report_file = self.results_dir / 'reports' / f'{dataset}_seq2seq_report.json'
                if report_file.exists():
                    with open(report_file) as f:
                        report = json.load(f)
                        if 'model_params' in report:
                            print(f"    Model Parameters: {report['model_params']:,}")
                        if 'training_time' in report:
                            print(f"    Training Time: {report['training_time']:.1f}s")

        if 3 in self.tasks and self.bonus_results:
            print("\n🎯 Task 3 - Bonus Set (Conditional Rendering):")
            print("-" * 40)
            print("\n  🎨 Conditional Transformations:")
            print("    • GREEN background: Normal text (baseline)")
            print("    • RED background: Reversed display (model must un-reverse)")

            if 'overall_metrics' in self.bonus_results:
                metrics = self.bonus_results['overall_metrics']
                print("\n  📊 Overall Performance:")
                print(f"    Character Error Rate: {metrics.get('cer', 1.0)*100:.1f}%")
                print(f"    Word Error Rate: {metrics.get('wer', 1.0)*100:.1f}%")
                print(f"    Exact Match Rate: {metrics.get('exact_match', 0)*100:.1f}%")

            if 'condition_performance' in self.bonus_results:
                print("\n  📈 Performance by Background Condition:")
                for condition, perf in self.bonus_results['condition_performance'].items():
                    print(f"    {condition.upper()}: {perf*100:.1f}% accuracy")

            if 'insights' in self.bonus_results:
                print("\n  💡 Key Insights:")
                for i, insight in enumerate(self.bonus_results['insights'], 1):
                    print(f"    {i}. {insight}")

        print("\n" + "="*80)
        print(" 📊 PIPELINE EXECUTION STATISTICS ".center(80, "="))
        print("="*80)

        successful_tasks = 0
        if 0 in self.tasks and self.generation_results:
            successful_tasks += 1
        if 1 in self.tasks and self.training_results:
            successful_tasks += len([r for r in self.training_results.values() if r.get('success', False)])
        if 2 in self.tasks and self.extraction_results:
            successful_tasks += len(self.extraction_results)
        if 3 in self.tasks and self.bonus_results:
            successful_tasks += 1

        print(f"\n✅ Tasks Successfully Completed: {successful_tasks}")
        print(f"⏱️ Total Execution Time: {time.time() - self.start_time:.1f} seconds")

        print("\n📁 Generated Files:")

        if self.visualize and VISUALIZATION_AVAILABLE:
            viz_dir = self.results_dir / 'visualizations'
            viz_files = list(viz_dir.glob('*.png'))
            if viz_files:
                print(f"\n  📊 Visualizations ({len(viz_files)} files):")
                for vf in sorted(viz_files):
                    print(f"    • {vf.name}")

        report_dir = self.results_dir / 'reports'
        json_reports = list(report_dir.glob('*.json'))
        other_reports = list(report_dir.glob('*.txt')) + list(report_dir.glob('*.png'))

        if json_reports or other_reports:
            print(f"\n  📄 Reports ({len(json_reports) + len(other_reports)} files):")
            for rf in sorted(json_reports):
                print(f"    • {rf.name}")
            for rf in sorted(other_reports):
                print(f"    • {rf.name}")

        model_dir = self.results_dir / 'models'
        model_files = list(model_dir.glob('*.pth'))
        if model_files:
            print(f"\n  🤖 Trained Models ({len(model_files)} files):")
            for mf in sorted(model_files):
                size_mb = mf.stat().st_size / (1024 * 1024)
                print(f"    • {mf.name} ({size_mb:.1f} MB)")

        print("\n" + "="*80)

    def _create_bonus_analysis_report(self):
        """Create comprehensive bonus set analysis report"""
        report = {
            'task': 'Bonus Set with Conditional Rendering',
            'timestamp': datetime.now().isoformat(),
            'challenge_description': 'Extract correct text regardless of visual transformation based on background color',
            'conditional_transformations': {
                'green': {
                    'transformation': 'Normal text',
                    'difficulty': 'Baseline',
                    'expected_challenge': 'Standard OCR task',
                    'performance': self.bonus_results.get('condition_performance', {}).get('green', 0)
                },
                'red': {
                    'transformation': 'Reversed display',
                    'difficulty': 'High',
                    'expected_challenge': 'Model must learn to reverse visual pattern',
                    'performance': self.bonus_results.get('condition_performance', {}).get('red', 0)
                }
            },
            'overall_metrics': self.bonus_results.get('overall_metrics', {}),
            'key_insights': self.bonus_results.get('insights', []),
            'recommendations': [
                'Augment training data with synthetic transformations',
                'Use separate attention heads for each condition type',
                'Implement condition-aware decoding strategies',
                'Consider ensemble models for different transformations'
            ]
        }

        report_path = self.results_dir / 'reports' / 'task3_bonus_comprehensive_analysis.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  • Comprehensive bonus analysis: {report_path.name}")

    def _create_pipeline_summary_report(self, elapsed_time):
        """Create final pipeline execution summary report"""
        summary = {
            'pipeline': 'Corrected CAPTCHA OCR Pipeline',
            'timestamp': datetime.now().isoformat(),
            'execution_time': f"{elapsed_time:.2f} seconds",
            'tasks_executed': self.tasks,
            'results': {}
        }

        if 0 in self.tasks and self.generation_results:
            summary['results']['task0_generation'] = {
                'datasets_created': self.generation_results,
                'total_images': sum(self.generation_results.values())
            }

        if 1 in self.tasks and self.training_results:
            summary['results']['task1_classification'] = {
                'models_trained': len(self.training_results),
                'datasets': self.training_results
            }

        if 2 in self.tasks and self.extraction_results:
            summary['results']['task2_ocr'] = {
                'datasets_evaluated': len(self.extraction_results),
                'average_cer': sum(m['cer'] for m in self.extraction_results.values()) / len(self.extraction_results),
                'details': self.extraction_results
            }

        if 3 in self.tasks and self.bonus_results:
            summary['results']['task3_bonus'] = self.bonus_results

        summary['outputs'] = {
            'visualizations': len(list((self.results_dir / 'visualizations').glob('*.png'))),
            'reports': len(list((self.results_dir / 'reports').glob('*.json'))),
            'models': len(list((self.results_dir / 'models').glob('*.pth')))
        }

        summary_path = self.results_dir / 'reports' / 'pipeline_execution_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📊 Pipeline summary report saved: {summary_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description='Corrected CAPTCHA OCR Pipeline with proper OCR implementation'
    )
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of images per dataset')
    parser.add_argument('--tasks', type=int, nargs='+', default=None,
                       help='Tasks to run (0,1,2,3)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per task in seconds')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualizations')
    parser.add_argument('--clean', action='store_true',
                       help='Clean previous runs before starting')

    # Epoch configuration arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Set all epochs to this value (overrides individual settings)')
    parser.add_argument('--task1-epochs', type=int, nargs=3, default=None,
                       metavar=('EASY', 'HARD', 'BONUS'),
                       help='Epochs for Task 1 classification (default: 10 50 30)')
    parser.add_argument('--task2-epochs', type=int, nargs=2, default=None,
                       metavar=('EASY', 'HARD'),
                       help='Epochs for Task 2 OCR (default: 20 30)')
    parser.add_argument('--task3-epochs', type=int, default=None,
                       help='Epochs for Task 3 bonus (default: 25)')

    args = parser.parse_args()

    if args.clean:
        print("🧹 Cleaning previous runs...")
        base_dir = Path(__file__).parent.parent

        # Clean data directory completely
        data_dir = base_dir / 'data'
        if data_dir.exists():
            print("  Removing entire data directory...")
            shutil.rmtree(data_dir)
            print("  ✅ Cleaned data directory")

        # Clean results directory completely
        results_dir = base_dir / 'results'
        if results_dir.exists():
            print("  Removing entire results directory...")
            shutil.rmtree(results_dir)
            print("  ✅ Cleaned results directory")

        # Recreate directory structure
        print("\n  📁 Recreating directory structure...")
        data_dir.mkdir(parents=True)
        (data_dir / 'easy').mkdir()
        (data_dir / 'hard').mkdir()
        (data_dir / 'bonus').mkdir()
        (data_dir / 'metadata').mkdir()
        (data_dir / 'samples').mkdir()
        print("  ✅ Created data subdirectories: easy, hard, bonus, metadata, samples")

        results_dir.mkdir(parents=True)
        (results_dir / 'models').mkdir()
        (results_dir / 'reports').mkdir()
        (results_dir / 'visualizations').mkdir()
        (results_dir / 'logs').mkdir()
        print("  ✅ Created results subdirectories: models, reports, visualizations, logs")

        print("\n✨ Clean complete! All previous data and results removed.")
        print("="*60)

    # Process epoch arguments
    task1_epochs = args.task1_epochs
    task2_epochs = args.task2_epochs
    task3_epochs = args.task3_epochs

    # If --epochs is specified, use it for all tasks
    if args.epochs is not None:
        task1_epochs = [args.epochs, args.epochs, args.epochs]
        task2_epochs = [args.epochs, args.epochs]
        task3_epochs = args.epochs
        print(f"📊 Using {args.epochs} epochs for all tasks")

    # Print epoch configuration if custom values are provided
    if task1_epochs or task2_epochs or task3_epochs:
        print("\n⚙️ Custom epoch configuration:")
        if task1_epochs:
            print(f"  Task 1: easy={task1_epochs[0]}, hard={task1_epochs[1]}, bonus={task1_epochs[2]}")
        if task2_epochs:
            print(f"  Task 2: easy={task2_epochs[0]}, hard={task2_epochs[1]}")
        if task3_epochs:
            print(f"  Task 3: {task3_epochs}")

    pipeline = CorrectedCaptchaPipeline(
        num_images=args.num_images,
        tasks=args.tasks,
        timeout=args.timeout,
        visualize=not args.no_visualize,
        task1_epochs=task1_epochs,
        task2_epochs=task2_epochs,
        task3_epochs=task3_epochs
    )

    pipeline.run_pipeline()

if __name__ == "__main__":
    main()