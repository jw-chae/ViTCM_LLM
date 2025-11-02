#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Qwen2.5-VL json data to ShareGPT/vllm format (messages+images, multi-turn support) 
by randomly splitting into 8:2 ratio

Instead of jsonl, saves as a single json file with the entire list.

Usage example:
python split_and_convert_to_qwen25vl_jsonl.py --input_json ../dataset/25.1.10-25.6.3.json --image_dir ../dataset/25.1.10-25.6.3 --output_train_json ../dataset/train.sharegpt.json --output_val_json ../dataset/val.sharegpt.json

- input_json: Original json file path
- image_dir: Folder path containing images
- output_train_json: Train set json file path
- output_val_json: Val set json file path

Qwen2.5-VL format:
One conversation (chat) list per line
"""
import json
import argparse
import os
import random
import re

def convert_and_write(data, image_dir, output_json):
    out_list = []
    for item in data:
        messages = []
        images = []
        user_content = item['prompt']
        user_image_count = user_content.count('<image>')
        if user_image_count == 0:
            user_content = '<image>' + user_content
            user_image_count = 1
        image_path = os.path.join(image_dir, item['image'])
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_json))
        images.extend([rel_image_path] * user_image_count)
        messages.append({"content": user_content, "role": "user"})
        messages.append({"content": item['output'], "role": "assistant"})
        out_list.append({"messages": messages, "images": images})
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"{output_json} file created. (Sample count: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL json 8:2 split and ShareGPT/vllm format converter (json list version)")
    parser.add_argument('--input_json', type=str, required=True, help='Original json file path')
    parser.add_argument('--image_dir', type=str, required=True, help='Image folder path')
    parser.add_argument('--output_train_json', type=str, required=True, help='Train set json file path')
    parser.add_argument('--output_val_json', type=str, required=True, help='Val set json file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for reproducibility)')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)
    n_total = len(data)
    n_train = int(n_total * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    convert_and_write(train_data, args.image_dir, args.output_train_json)
    convert_and_write(val_data, args.image_dir, args.output_val_json)
    print(f"Split completed: {len(train_data)} train, {len(val_data)} val out of {n_total} total.")

if __name__ == "__main__":
    main() 