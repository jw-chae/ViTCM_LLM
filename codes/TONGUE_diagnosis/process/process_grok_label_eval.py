import json
import os

# Path setup
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, '../../dataset')
result_dir = os.path.join(base_dir, '../../shezhen_results')

# File paths
pred_file = os.path.join(dataset_dir, 'val.grok_output.jsonl')
label_file = os.path.join(result_dir, 'label.txt')
out_file = os.path.join(result_dir, 'compare_grok_label.jsonl')

# Load predictions
grok_preds = []
with open(pred_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            grok_preds.append(item['grok_output'].strip().replace('。', '').replace('.', ''))

# Load labels
labels = []
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        labels.append(line.strip().replace('。', '').replace('.', ''))

assert len(grok_preds) == len(labels), f"Prediction({len(grok_preds)}) and label({len(labels)}) count mismatch"

# Create jsonl
with open(out_file, 'w', encoding='utf-8') as fout:
    for pred, label in zip(grok_preds, labels):
        fout.write(json.dumps({'predict': pred, 'label': label}, ensure_ascii=False) + '\n')

print(f"[Complete] {out_file} created. Total {len(grok_preds)} items") 