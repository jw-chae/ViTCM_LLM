import json
import os
import argparse
import random
import re

def normalize_filename(filename):
    """Normalize date format in filename"""
    # Convert 2024-12-01 format to 24.12.1 format (remove leading zeros)
    pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(pattern, filename)
    if match:
        year = match.group(1)[2:]  # 2024 -> 24
        month = str(int(match.group(2)))  # 12 -> 12, 01 -> 1
        day = str(int(match.group(3)))    # 01 -> 1, 08 -> 8
        normalized = filename.replace(match.group(0), f"{year}.{month}.{day}")
        return normalized
    return filename

def filter_existing_images(data, image_dir):
    """Filter data to only include items with existing image files (improved matching)"""
    existing_files = set(os.listdir(image_dir))
    filtered_data = []
    missing_count = 0
    
    for item in data:
        original_image = item['image']
        normalized_image = normalize_filename(original_image)
        
        # Also try version without spaces
        no_space_image = original_image.replace(' ', '')
        no_space_normalized = normalized_image.replace(' ', '')
        
        if (original_image in existing_files or 
            normalized_image in existing_files or
            no_space_image in existing_files or
            no_space_normalized in existing_files):
            
            # Update with matched filename
            if original_image in existing_files:
                pass  # Keep as is
            elif normalized_image in existing_files:
                item['image'] = normalized_image
            elif no_space_image in existing_files:
                item['image'] = no_space_image
            elif no_space_normalized in existing_files:
                item['image'] = no_space_normalized
                
            filtered_data.append(item)
        else:
            missing_count += 1
    
    print(f"Out of {len(data)} total, {len(filtered_data)} have existing images, {missing_count} missing images")
    return filtered_data

def convert_and_write_json(data, image_dir, output_json):
    out_list = []
    for item in data:
        user_content = "<image>" + item['prompt'].replace('<image>', '').strip()
        messages = [
            {"content": user_content, "role": "user"},
            {"content": item['output'], "role": "assistant"}
        ]
        image_path = os.path.join(image_dir, os.path.basename(item['image']))
        rel_image_path = os.path.relpath(image_path, os.path.dirname(output_json))
        images = [rel_image_path]
        out = {"messages": messages, "images": images}
        out_list.append(out)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"{output_json} file created. (Sample count: {len(data)})")

def main():
    parser = argparse.ArgumentParser(description="ShareGPT/vllm format json 9:1 split converter (improved image matching)")
    parser.add_argument('--input_json', type=str, required=True, help='Original json file path')
    parser.add_argument('--image_dir', type=str, required=True, help='Image folder path')
    parser.add_argument('--output_train_json', type=str, required=True, help='Train set json file path')
    parser.add_argument('--output_val_json', type=str, required=True, help='Val set json file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for reproducibility)')
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter data to only include items with existing images (improved matching)
    filtered_data = filter_existing_images(data, args.image_dir)
    
    if len(filtered_data) == 0:
        print("No matching images found!")
        return

    random.seed(args.seed)
    random.shuffle(filtered_data)
    n_total = len(filtered_data)
    n_train = int(n_total * 0.9)
    train_data = filtered_data[:n_train]
    val_data = filtered_data[n_train:]

    convert_and_write_json(train_data, args.image_dir, args.output_train_json)
    convert_and_write_json(val_data, args.image_dir, args.output_val_json)
    print(f"Split completed: {len(train_data)} train, {len(val_data)} val out of {n_total} total.")

if __name__ == "__main__":
    main() 