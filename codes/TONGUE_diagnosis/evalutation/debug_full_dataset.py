import json
import os
from evaluation import parse_tongue_features, compute_category_similarity, load_json_config

def debug_full_dataset():
    config = load_json_config('token_config.json')
    
    with open('../../dataset/result_new_best.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total data count: {len(data)}")
    
    # Classify by score
    high_scores = []  # >= 0.8
    medium_scores = []  # 0.5-0.8
    low_scores = []  # < 0.5
    
    for i, item in enumerate(data):
        pred = item.get('predict', '').strip()
        label = item.get('label', '').strip()
        
        s_tongue, s_coat, s_location, s_other, s_mean = compute_category_similarity(pred, label, config)
        
        if s_mean >= 0.8:
            high_scores.append((i, pred, label, s_mean))
        elif s_mean >= 0.5:
            medium_scores.append((i, pred, label, s_mean))
        else:
            low_scores.append((i, pred, label, s_mean))
    
    print(f"\nHigh scores (≥0.8): {len(high_scores)} items")
    print(f"Medium scores (0.5-0.8): {len(medium_scores)} items")
    print(f"Low scores (<0.5): {len(low_scores)} items")
    
    print(f"\n=== Low score samples (top 10) ===")
    low_scores.sort(key=lambda x: x[3])  # Sort by score
    for i, (idx, pred, label, score) in enumerate(low_scores[:10]):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Prediction: {pred}")
        print(f"   Label: {label}")
        
        pred_tokens = parse_tongue_features(pred)
        label_tokens = parse_tongue_features(label)
        print(f"   Prediction tokens: {list(pred_tokens)}")
        print(f"   Label tokens: {list(label_tokens)}")
        print()
    
    print(f"\n=== High score samples (top 5) ===")
    high_scores.sort(key=lambda x: x[3], reverse=True)  # Sort by score
    for i, (idx, pred, label, score) in enumerate(high_scores[:5]):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Prediction: {pred}")
        print(f"   Label: {label}")
        print()

if __name__ == "__main__":
    debug_full_dataset() 