#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# evaluation2_ver2.py의 함수들을 import
sys.path.append(os.path.dirname(__file__))
from evaluation import load_json_config, compute_category_similarity

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_bleu_score(predictions, references):
    """Calculate BLEU score (Llama Factory method)"""
    smoothie = SmoothingFunction().method3
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred)
        ref_tokens = list(ref)
        try:
            bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            bleu_scores.append(0.0)
    
    return np.mean(bleu_scores)

def calculate_our_metric(predictions, references, config):
    """Calculate our metric score"""
    total_mean = 0.0
    for pred, ref in zip(predictions, references):
        pred_clean = pred.replace('舌诊结果: ', '').strip()
        ref_clean = ref.replace('舌诊结果: ', '').strip()
        _, _, _, _, s_mean = compute_category_similarity(pred_clean, ref_clean, config)
        total_mean += s_mean
    
    return total_mean / len(predictions) if predictions else 0.0

def main():
    input_dir = "../../shezhen_results"
    output_dir = "../../metric_results"
    config_path = "token_config.json"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = load_json_config(config_path)
    
    # JSONL files to evaluate (excluding BLEU result files)
    exclude_files = ['bleu_scores.jsonl', 'bleu_scores_char.jsonl', 'bleu_summary.txt', 'bleu_summary_char.txt']
    
    jsonl_files = []
    for f in os.listdir(input_dir):
        if f.endswith('.jsonl') and f not in exclude_files:
            jsonl_files.append(f)
    
    print("Generating combined results...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of files to evaluate: {len(jsonl_files)}")
    
    # Dictionary to store results
    results = {}
    
    for fname in sorted(jsonl_files):
        print(f"\nProcessing: {fname}")
        file_path = os.path.join(input_dir, fname)
        
        try:
            # Load data
            data = load_jsonl(file_path)
            if not data:
                print(f"  Warning: No data in {fname}")
                continue
            
            # Separate predict and label
            predictions = [item.get('predict', '') for item in data]
            references = [item.get('label', '') for item in data]
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(predictions, references)
            
            # Calculate our metric score
            our_metric_score = calculate_our_metric(predictions, references, config)
            
            # Store results
            results[fname] = {
                'our_metric': our_metric_score,
                'bleu': bleu_score,
                'sample_count': len(data)
            }
            
            print(f"  Our Metric: {our_metric_score:.4f}")
            print(f"  BLEU: {bleu_score:.4f}")
            print(f"  Sample count: {len(data)}")
            
        except Exception as e:
            print(f"  Error: Error processing {fname} - {e}")
            continue
    
    # Create combined results file
    output_file = os.path.join(output_dir, "combined_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Combined Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for fname, result in sorted(results.items()):
            f.write(f"File: {fname}\n")
            f.write(f"  Our Metric: {result['our_metric']:.4f}\n")
            f.write(f"  BLEU: {result['bleu']:.4f}\n")
            f.write(f"  Sample count: {result['sample_count']}\n")
            f.write("-" * 30 + "\n")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "results_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Evaluation Results Summary\n")
        f.write("=" * 30 + "\n\n")
        
        # Sort by Our Metric
        f.write("By Our Metric (descending):\n")
        sorted_by_metric = sorted(results.items(), key=lambda x: x[1]['our_metric'], reverse=True)
        for fname, result in sorted_by_metric:
            f.write(f"{fname}: {result['our_metric']:.4f}\n")
        
        f.write("\nBy BLEU (descending):\n")
        sorted_by_bleu = sorted(results.items(), key=lambda x: x[1]['bleu'], reverse=True)
        for fname, result in sorted_by_bleu:
            f.write(f"{fname}: {result['bleu']:.4f}\n")
    
    print(f"\nCombined results saved to {output_file}")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 