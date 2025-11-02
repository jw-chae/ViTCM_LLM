#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tongue Image Dataset Augmentation Script (Modified Version)
Properly handles image paths in JSON files.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

class TongueImageAugmenter:
    def __init__(self, augmentation_factor=3):
        """
        Initialize tongue image augmenter
        
        Args:
            augmentation_factor (int): Number of augmented images to generate per image
        """
        self.augmentation_factor = augmentation_factor
        
    def rotate_image(self, image, angle_range=(-15, 15)):
        """Rotate image"""
        angle = random.uniform(angle_range[0], angle_range[1])
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Generate rotated image
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def scale_image(self, image, scale_range=(0.9, 1.1)):
        """Resize image"""
        scale = random.uniform(scale_range[0], scale_range[1])
        height, width = image.shape[:2]
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop to original size
        if scale < 1.0:
            # Padding
            top = (height - new_height) // 2
            bottom = height - new_height - top
            left = (width - new_width) // 2
            right = width - new_width - left
            scaled = cv2.copyMakeBorder(scaled, top, bottom, left, right, 
                                      cv2.BORDER_REFLECT)
        else:
            # Crop
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            scaled = scaled[start_y:start_y+height, start_x:start_x+width]
        
        return scaled
    
    def adjust_brightness(self, image, brightness_range=(0.8, 1.2)):
        """Adjust brightness"""
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        
        # Convert to PIL Image for brightness adjustment
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        adjusted = enhancer.enhance(brightness_factor)
        
        # Convert back to OpenCV format
        adjusted_cv = cv2.cvtColor(np.array(adjusted), cv2.COLOR_RGB2BGR)
        return adjusted_cv
    
    def adjust_color_temperature(self, image, temp_range=(0.9, 1.1)):
        """Adjust color temperature (warm/cool tones)"""
        temp_factor = random.uniform(temp_range[0], temp_range[1])
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Adjust color temperature (Color Balance)
        r, g, b = pil_image.split()
        
        if temp_factor > 1.0:  # Warm tone (increase red, yellow)
            r = r.point(lambda x: min(255, int(x * temp_factor)))
            b = b.point(lambda x: max(0, int(x / temp_factor)))
        else:  # Cool tone (increase blue)
            b = b.point(lambda x: min(255, int(x / temp_factor)))
            r = r.point(lambda x: max(0, int(x * temp_factor)))
        
        adjusted = Image.merge('RGB', (r, g, b))
        
        # Convert to OpenCV format
        adjusted_cv = cv2.cvtColor(np.array(adjusted), cv2.COLOR_RGB2BGR)
        return adjusted_cv
    
    def augment_single_image(self, image_path, output_dir, base_filename):
        """Augment a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            return []
        
        augmented_images = []
        
        for i in range(self.augmentation_factor):
            # Copy original image
            aug_image = image.copy()
            
            # Apply augmentation technique
            aug_type = random.choice(['rotate', 'scale', 'brightness', 'color_temp'])
            
            if aug_type == 'rotate':
                aug_image = self.rotate_image(aug_image)
                suffix = f"_rot_{i+1}"
            elif aug_type == 'scale':
                aug_image = self.scale_image(aug_image)
                suffix = f"_scale_{i+1}"
            elif aug_type == 'brightness':
                aug_image = self.adjust_brightness(aug_image)
                suffix = f"_bright_{i+1}"
            elif aug_type == 'color_temp':
                aug_image = self.adjust_color_temperature(aug_image)
                suffix = f"_temp_{i+1}"
            
            # Generate filename
            name, ext = os.path.splitext(base_filename)
            aug_filename = f"{name}{suffix}{ext}"
            aug_path = os.path.join(output_dir, aug_filename)
            
            # Save augmented image
            cv2.imwrite(aug_path, aug_image)
            augmented_images.append(aug_path)
        
        return augmented_images
    
    def augment_dataset(self, json_path, image_dir, output_dir):
        """Augment entire dataset"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        augmented_image_dir = os.path.join(output_dir, "augmented_images")
        os.makedirs(augmented_image_dir, exist_ok=True)
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Original dataset size: {len(data)}")
        
        augmented_data = []
        image_mapping = {}  # Original image path -> augmented image paths
        
        # Perform augmentation for each sample
        for idx, sample in enumerate(tqdm(data, desc="Augmenting data")):
            augmented_data.append(sample)  # Add original sample
            
            # Process image paths
            if 'images' in sample and sample['images']:
                original_images = sample['images']
                augmented_images = []
                
                for img_path in original_images:
                    # Extract filename from JSON image path
                    img_path = os.path.basename(img_path)
                    
                    # Create absolute path
                    full_img_path = os.path.join(image_dir, img_path)
                    
                    # Extract filename
                    base_filename = os.path.basename(img_path)
                    
                    # Check if already augmented
                    if img_path in image_mapping:
                        augmented_images.extend(image_mapping[img_path])
                    else:
                        # Perform new augmentation
                        aug_images = self.augment_single_image(
                            full_img_path, augmented_image_dir, base_filename
                        )
                        
                        # Convert to relative paths and save
                        rel_aug_images = []
                        for aug_img in aug_images:
                            rel_path = os.path.join("augmented_images", os.path.basename(aug_img))
                            rel_aug_images.append(rel_path)
                        
                        image_mapping[img_path] = rel_aug_images
                        augmented_images.extend(rel_aug_images)
                
                # Create augmented samples
                for i in range(self.augmentation_factor):
                    aug_sample = {
                        'messages': sample['messages'].copy(),
                        'images': [augmented_images[i]] if augmented_images else []
                    }
                    augmented_data.append(aug_sample)
        
        # Save augmented dataset
        output_json_path = os.path.join(output_dir, "augmented_dataset.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        print(f"Augmented dataset size: {len(augmented_data)}")
        print(f"Augmented dataset saved: {output_json_path}")
        print(f"Augmented images saved: {augmented_image_dir}")
        
        return output_json_path, augmented_image_dir

def main():
    parser = argparse.ArgumentParser(description='Tongue image dataset augmentation (Modified version)')
    parser.add_argument('--json_path', type=str, 
                       default='/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/unique_25.1.8_dataset_train_sharegpt_fixed_cleaned.json',
                       help='Input JSON file path')
    parser.add_argument('--image_dir', type=str,
                       default='/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/25.1.8之前所有with上中医三院',
                       help='Image directory path')
    parser.add_argument('--output_dir', type=str,
                       default='/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/augmented_dataset_final',
                       help='Output directory path')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                       help='Number of augmented images to generate per image')
    
    args = parser.parse_args()
    
    # Initialize augmenter
    augmenter = TongueImageAugmenter(augmentation_factor=args.augmentation_factor)
    
    # Perform dataset augmentation
    augmenter.augment_dataset(args.json_path, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main() 