# Copyright (c) 2026 MyCompany LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
YOLOv11 Density-tiered Validation Script
---------------------------------------
This script evaluates a trained YOLOv11m crowd detection model's performance
separately across low, medium, and high density images in the validation set.

It automatically:
1. Locates the trained YOLOv11m checkpoint.
2. Identifies the validation images and labels.
3. Groups validation images into three density buckets based on ground-truth box count:
   - LOW density: < 30 people
   - MEDIUM density: 30-100 people
   - HIGH density: > 100 people
4. Runs YOLO validation separately on each bucket using temporary dataset configs.
5. Prints a clear comparison table of metrics (Precision, Recall, mAP50, mAP50-95).
6. Displays a comparative conclusion against the overall blended average.
"""

import os
import sys
import argparse
import shutil
import yaml
import torch
from ultralytics import YOLO

# Overall blended average metrics from epoch 60 for comparison
BLENDED_PRECISION = 0.693
BLENDED_RECALL = 0.392
BLENDED_MAP50 = 0.410
BLENDED_MAP50_95 = 0.161

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11m performance separately by crowd density tiers.")
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to YOLO model checkpoint (.pt). If not specified, auto-detects from runs or models.')
    parser.add_argument('--data', type=str, default=None, 
                        help='Path to dataset data.yaml file. If not specified, auto-detects in crowd density folder.')
    parser.add_argument('--imgsz', type=int, default=960, 
                        help='Image resolution for validation (default: 960, matching training resolution).')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to run inference on (e.g. 0, cpu). Auto-detects CUDA GPU by default.')
    return parser.parse_args()

def locate_model_and_copy(model_arg):
    """
    Locates the model checkpoint from candidates, prioritizing model_arg,
    and copies it to 'models/yolo11m_best.pt' if it exists elsewhere.
    """
    model_path = model_arg
    
    if not model_path:
        candidates = [
            os.path.join("models", "yolo11m_best.pt"),
            os.path.join("runs", "detect", "runs", "detect", "train_yolo11m_960px", "weights", "best.pt"),
            os.path.join("runs", "detect", "train_yolo11m_960px", "weights", "best.pt"),
            os.path.join("models", "best.pt")
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if not model_path or not os.path.exists(model_path):
        print("ERROR: Could not locate a valid model checkpoint.")
        print("Please ensure you have trained the model or pass the exact path using --model <path>.")
        sys.exit(1)

    target_path = os.path.join("models", "yolo11m_best.pt")
    
    # If the located model is not at the target path, copy it there to align with standard project structure
    if os.path.abspath(model_path) != os.path.abspath(target_path):
        os.makedirs("models", exist_ok=True)
        print(f"Found model checkpoint at: {model_path}")
        print(f"Copying model checkpoint to: {target_path} ...")
        try:
            shutil.copy(model_path, target_path)
            print("Successfully copied checkpoint.")
            model_path = target_path
        except Exception as e:
            print(f"Warning: Failed to copy model checkpoint: {e}. Proceeding with original path.")
    else:
        print(f"Using model checkpoint: {model_path}")

    return os.path.abspath(model_path)

def locate_dataset_yaml(data_arg):
    """
    Locates the dataset configuration data.yaml.
    """
    data_path = data_arg
    
    if not data_path:
        candidates = [
            r"E:\Crowd Detection &  Prevention\crowd density.v1-v1.yolov8\data.yaml",
            os.path.join("crowd density.v1-v1.yolov8", "data.yaml"),
            os.path.join("..", "crowd density.v1-v1.yolov8", "data.yaml")
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                data_path = candidate
                break

    if not data_path or not os.path.exists(data_path):
        print("ERROR: Could not locate dataset configuration (data.yaml).")
        print("Please specify the dataset YAML path using --data <path>.")
        sys.exit(1)

    print(f"Using dataset YAML: {os.path.abspath(data_path)}")
    return os.path.abspath(data_path)

def clean_temp_files(dataset_dir, bucket_name):
    """
    Deletes all temporary configuration, list, and cache files created for the bucket.
    """
    for entry in os.listdir(dataset_dir):
        if entry.startswith(f"temp_val_{bucket_name}") or entry.startswith(f"temp_data_{bucket_name}"):
            file_to_del = os.path.join(dataset_dir, entry)
            try:
                if os.path.isfile(file_to_del):
                    os.remove(file_to_del)
                elif os.path.isdir(file_to_del):
                    shutil.rmtree(file_to_del)
            except Exception as e:
                print(f"Warning: Could not remove temporary artifact '{file_to_del}': {e}")

def run_validation_on_bucket(bucket_name, images, data_yaml_path, model_path, device, imgsz):
    """
    Generates a temporary subset dataset configuration, runs YOLO validation,
    extracts the target metrics, and cleans up temporary files.
    """
    if not images:
        print(f"Skipping {bucket_name.upper()} density bucket because it contains 0 images.")
        return None

    dataset_dir = os.path.dirname(data_yaml_path)
    
    # Write the list of validation images in this bucket
    txt_filename = f"temp_val_{bucket_name}.txt"
    txt_path = os.path.join(dataset_dir, txt_filename)
    
    with open(txt_path, 'w', encoding='utf-8') as tf:
        for img in images:
            # Use forward slashes and absolute paths for cross-platform compatibility in Ultralytics
            tf.write(os.path.abspath(img).replace('\\', '/') + '\n')

    # Load the base dataset config to copy category mappings
    with open(data_yaml_path, 'r') as df:
        base_data = yaml.safe_load(df)

    # Build the custom dataset yaml for this bucket validation run
    temp_yaml_data = {
        'path': dataset_dir.replace('\\', '/'),
        'train': base_data.get('train', 'train/images').replace('\\', '/'),
        'val': txt_filename,
        'nc': base_data['nc'],
        'names': base_data['names']
    }

    temp_yaml_filename = f"temp_data_{bucket_name}.yaml"
    temp_yaml_path = os.path.join(dataset_dir, temp_yaml_filename)
    with open(temp_yaml_path, 'w', encoding='utf-8') as yf:
        yaml.dump(temp_yaml_data, yf)

    print(f"\n=====================================================================")
    print(f" Running YOLOv11m validation on {bucket_name.upper()} tier ({len(images)} images)")
    print(f"=====================================================================")
    
    try:
        model = YOLO(model_path)
        # Run validation with verbose=False to keep logs clean
        # save=False and plots=False prevents writing lots of JPGs to the validation runs directory
        results = model.val(
            data=temp_yaml_path,
            split='val',
            imgsz=imgsz,
            device=device,
            verbose=False,
            save=False,
            plots=False
        )
    except Exception as e:
        print(f"ERROR: Validation failed for {bucket_name} bucket: {e}")
        results = None
    finally:
        # Clean up temporary dataset configs, text lists, and generated cache files
        clean_temp_files(dataset_dir, bucket_name)

    return results

def main():
    args = parse_args()
    
    # 1. Determine device
    if args.device is not None:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Locate and copy model checkpoint
    model_path = locate_model_and_copy(args.model)
    
    # 3. Locate dataset YAML
    data_yaml_path = locate_dataset_yaml(args.data)
    
    # 4. Load dataset YAML and identify validation images/labels
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    val_images_rel = yaml_data.get('val')
    if not val_images_rel:
        print("ERROR: 'val' key is missing from dataset data.yaml.")
        sys.exit(1)
        
    # Resolve relative or absolute path of validation images
    dataset_root = yaml_data.get('path', '')
    if dataset_root and not os.path.isabs(val_images_rel):
        val_images_dir = os.path.join(dataset_root, val_images_rel)
    else:
        val_images_dir = val_images_rel
        
    val_images_dir = os.path.abspath(val_images_dir)
    
    # If the resolved path does not exist, look relative to the data.yaml file's parent directory
    if not os.path.exists(val_images_dir):
        yaml_dir = os.path.dirname(data_yaml_path)
        alternative_path = os.path.join(yaml_dir, val_images_rel)
        if os.path.exists(alternative_path):
            val_images_dir = os.path.abspath(alternative_path)
            
    if not os.path.exists(val_images_dir):
        print(f"ERROR: Could not find validation images directory at: {val_images_dir}")
        sys.exit(1)
        
    # Validation labels directory (sibling of images directory named 'labels')
    val_labels_dir = os.path.join(os.path.dirname(val_images_dir), 'labels')
    val_labels_dir = os.path.abspath(val_labels_dir)
    
    if not os.path.exists(val_labels_dir):
        print(f"ERROR: Could not find validation labels directory at: {val_labels_dir}")
        sys.exit(1)

    print(f"Resolved Validation Images Path: {val_images_dir}")
    print(f"Resolved Validation Labels Path: {val_labels_dir}")

    # 5. Count ground truth annotations per image and group into buckets
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        os.path.join(val_images_dir, f) 
        for f in os.listdir(val_images_dir) 
        if f.lower().endswith(supported_extensions)
    ]
    
    if not image_files:
        print("ERROR: No images found in the validation directory.")
        sys.exit(1)

    low_density_images = []
    medium_density_images = []
    high_density_images = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(val_labels_dir, label_name)
        
        # Count lines in the label file (each line corresponds to one person box)
        box_count = 0
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as lf:
                for line in lf:
                    if line.strip():
                        box_count += 1
                        
        if box_count < 30:
            low_density_images.append(img_path)
        elif box_count <= 100:
            medium_density_images.append(img_path)
        else:
            high_density_images.append(img_path)

    # Print distribution
    total_images = len(image_files)
    print("\n" + "=" * 60)
    print("           VALIDATION DATASET DENSITY DISTRIBUTION")
    print("=" * 60)
    print(f"Total Validation Images: {total_images}")
    print(f"  - LOW Density (< 30 people):      {len(low_density_images):>4} images ({len(low_density_images)/total_images*100:5.1f}%)")
    print(f"  - MEDIUM Density (30-100 people):  {len(medium_density_images):>4} images ({len(medium_density_images)/total_images*100:5.1f}%)")
    print(f"  - HIGH Density (> 100 people):    {len(high_density_images):>4} images ({len(high_density_images)/total_images*100:5.1f}%)")
    print("=" * 60 + "\n")

    # 6. Run validation per bucket and aggregate results
    buckets = {
        'LOW (<30)': low_density_images,
        'MEDIUM (30-100)': medium_density_images,
        'HIGH (>100)': high_density_images
    }
    
    bucket_results = {}
    
    for bucket_name, bucket_imgs in buckets.items():
        # Sanitize bucket name for filenames
        clean_name = bucket_name.split()[0].lower()
        results = run_validation_on_bucket(
            bucket_name=clean_name,
            images=bucket_imgs,
            data_yaml_path=data_yaml_path,
            model_path=model_path,
            device=device,
            imgsz=args.imgsz
        )
        
        if results is not None:
            # Extract standard metrics
            # Support both box properties and results_dict dictionary mapping
            p = getattr(results.box, 'mp', 0.0)
            r = getattr(results.box, 'mr', 0.0)
            map50 = getattr(results.box, 'map50', 0.0)
            map_all = getattr(results.box, 'map', 0.0)
            
            # Fallback if properties are zero but dict has keys
            if p == 0.0 and hasattr(results, 'results_dict') and results.results_dict:
                p = results.results_dict.get('metrics/precision(B)', 0.0)
                r = results.results_dict.get('metrics/recall(B)', 0.0)
                map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
                map_all = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
                
            bucket_results[bucket_name] = {
                'count': len(bucket_imgs),
                'precision': p,
                'recall': r,
                'mAP50': map50,
                'mAP50-95': map_all
            }
        else:
            bucket_results[bucket_name] = {
                'count': len(bucket_imgs),
                'precision': float('nan'),
                'recall': float('nan'),
                'mAP50': float('nan'),
                'mAP50-95': float('nan')
            }

    # 7. Print Comparison Table
    print("\n" + "=" * 80)
    print("                    YOLOv11m DENSITY-TIER EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Density Tier':<18} | {'Image Count':<11} | {'Precision':<9} | {'Recall':<8} | {'mAP50':<7} | {'mAP50-95':<8}")
    print("-" * 80)
    
    for tier, res in bucket_results.items():
        if res['count'] > 0:
            print(f"{tier:<18} | {res['count']:<11} | {res['precision']:<9.3f} | {res['recall']:<8.3f} | {res['mAP50']:<7.3f} | {res['mAP50-95']:<8.3f}")
        else:
            print(f"{tier:<18} | {res['count']:<11} | {'N/A':<9} | {'N/A':<8} | {'N/A':<7} | {'N/A':<8}")
            
    print("=" * 80 + "\n")

    # 8. Print Conclusion comparing LOW density vs blended average
    low_res = bucket_results.get('LOW (<30)')
    if low_res and low_res['count'] > 0:
        p_diff = low_res['precision'] - BLENDED_PRECISION
        r_diff = low_res['recall'] - BLENDED_RECALL
        m50_diff = low_res['mAP50'] - BLENDED_MAP50
        m95_diff = low_res['mAP50-95'] - BLENDED_MAP50_95
        
        # Decide if low density is meaningfully better
        # A metric is considered "meaningfully better" if mAP50 increases by at least 0.05
        is_meaningfully_better = m50_diff >= 0.05
        
        print("=" * 80)
        print("                               CONCLUSION")
        print("=" * 80)
        print(f"Comparing LOW Density (< 30 people) to the Blended Average (Epoch 60):")
        print(f"  - Precision: {low_res['precision']:.3f} vs {BLENDED_PRECISION:.3f} ({'+' if p_diff >= 0 else ''}{p_diff:+.3f})")
        print(f"  - Recall:    {low_res['recall']:.3f} vs {BLENDED_RECALL:.3f} ({'+' if r_diff >= 0 else ''}{r_diff:+.3f})")
        print(f"  - mAP50:     {low_res['mAP50']:.3f} vs {BLENDED_MAP50:.3f} ({'+' if m50_diff >= 0 else ''}{m50_diff:+.3f})")
        print(f"  - mAP50-95:  {low_res['mAP50-95']:.3f} vs {BLENDED_MAP50_95:.3f} ({'+' if m95_diff >= 0 else ''}{m95_diff:+.3f})")
        print("-" * 80)
        
        if is_meaningfully_better:
            print("STATUS: LOW density performance is MEANINGFULLY BETTER than the blended average.")
            print("  This validates the hybrid design decision to trust YOLOv11m for low-density (sparse)")
            print("  scenes. The significantly higher localization accuracy (mAP50/mAP50-95) shows that YOLO")
            print("  is highly competent at individual detections when crowd occlusion is low.")
        else:
            # Let's check if it is still at least slightly better or similar
            if m50_diff > 0:
                print("STATUS: LOW density performance is SLIGHTLY BETTER than the blended average.")
                print(f"  mAP50 improved by {m50_diff:.3f}, but did not reach the +0.05 threshold.")
            else:
                print("STATUS: LOW density performance is NOT better than the blended average.")
                print("  The metrics on sparse images are comparable to or lower than the general average.")
                print("  Consider retraining YOLO or adjusting confidence thresholds for sparse settings.")
        print("=" * 80)
    else:
        print("ERROR: Could not compute conclusion because the LOW density bucket was empty or failed validation.")

if __name__ == "__main__":
    main()
