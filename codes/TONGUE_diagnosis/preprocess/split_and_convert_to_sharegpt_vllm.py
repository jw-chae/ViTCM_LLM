#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Qwen2.5-VL json data to ShareGPT/vllm format (messages+images) 
by randomly splitting into 8:2 ratio

Usage example:
python split_and_convert_to_sharegpt_vllm.py --input_json ../dataset/25.1.10-25.6.3.json --image_dir ../dataset/25.1.10-25.6.3 --output_train_jsonl ../dataset/train.sharegpt.jsonl --output_val_jsonl ../dataset/val.sharegpt.jsonl
"""
import json
import argparse
import os
import random

def convert_and_write(data, image_dir, output_jsonl):
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in data:
            user_content = "<image>" + item['prompt'].replace('<image>', '').strip()
            messages = [
                {"content": user_content, "role": "user"},
                {"content": item['output'], "role": "assistant"}
            ]
            image_path = os.path.join(image_dir, item['image'])
            rel_image_path = os.path.relpath(image_path, os.path.dirname(output_jsonl))
            images = [rel_image_path]
            out = {"messages": messages, "images": images}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"{output_jsonl} file created. (Sample count: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL json 8:2 split and ShareGPT/vllm format converter")
    parser.add_argument('--input_json', type=str, required=True, help='Original json file path')
    parser.add_argument('--image_dir', type=str, required=True, help='Image folder path')
    parser.add_argument('--output_train_jsonl', type=str, required=True, help='Train set jsonl file path')
    parser.add_argument('--output_val_jsonl', type=str, required=True, help='Val set jsonl file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for reproducibility)')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]

    convert_and_write(train_data, args.image_dir, args.output_train_jsonl)
    convert_and_write(val_data, args.image_dir, args.output_val_jsonl)
    print(f"Split completed: {len(train_data)} train, {len(val_data)} val out of {n_total} total.")

if __name__ == "__main__":
    main() 