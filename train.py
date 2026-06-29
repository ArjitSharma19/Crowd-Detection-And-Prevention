import os
import sys
import shutil
import torch
import random
import glob
import yaml
from ultralytics import YOLO

# =====================================================================
# GLOBAL CONFIGURATION & FINE-TUNING PARAMETERS
# =====================================================================
TARGET_COUNT = 800  # Default target count for low-light subsampling
SEED = 42

# Base model weights to fine-tune. Default to the best daytime model weights.
MODEL_NAME = "models/yolo11m_best.pt"
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = "yolo11m.pt"

SOURCE_IMAGE_DIR = r"E:\Crowd Detection &  Prevention\Night Vision.v3i.yolov11\train\images"
SOURCE_LABEL_DIR = r"E:\Crowd Detection &  Prevention\Night Vision.v3i.yolov11\train\labels"

# Location to store the subsampled dataset
SUBSAMPLED_DIR = r"E:\Crowd Detection &  Prevention\Night Vision Subsampled"

# Location of the existing daytime dataset
DAYTIME_DATASET_DIR = r"E:\Crowd Detection &  Prevention\crowd density.v1-v1.yolov8"

# Location where the merged dataset will be created
MERGED_DATASET_DIR = r"E:\Crowd Detection &  Prevention\crowd_density_merged"


def subsample_dataset(source_image_dir, source_label_dir, target_count=800, seed=42):
    """
    Randomly selects target_count image+label pairs from the source dataset.
    Keeps image/label pairs together, ensuring we never select an image without its matching label file.
    Copies the selected subset into a new folder so the original dataset stays untouched.
    """
    # 1. Identify all image files in source_image_dir
    supported_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_paths = []
    for ext in supported_exts:
        # Search for both lowercase and uppercase extensions
        image_paths.extend(glob.glob(os.path.join(source_image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(source_image_dir, ext.upper())))
    
    # Remove duplicates if any
    image_paths = list(set(image_paths))
    
    # 2. Check for matching label files
    valid_pairs = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        label_path = os.path.join(source_label_dir, f"{base_name}.txt")
        if os.path.exists(label_path):
            valid_pairs.append((img_path, label_path))
            
    available_count = len(valid_pairs)
    
    # 3. Random selection
    random.seed(seed)
    selected_count = min(target_count, available_count)
    selected_pairs = random.sample(valid_pairs, selected_count)
    
    # 4. Print how many images were available vs how many were selected
    print("=" * 70)
    print("            DATASET SUBSAMPLING SUMMARY")
    print("=" * 70)
    print(f"Source Image Directory: {source_image_dir}")
    print(f"Total Available Pairs:  {available_count}")
    print(f"Target Selection Count: {target_count}")
    print(f"Actually Selected:      {selected_count}")
    print("=" * 70)
    
    # 5. Copy selected subset into new folders
    dest_image_dir = os.path.join(SUBSAMPLED_DIR, "images")
    dest_label_dir = os.path.join(SUBSAMPLED_DIR, "labels")
    
    # Clear existing subsampled folder if any to avoid mixing old/new runs
    if os.path.exists(SUBSAMPLED_DIR):
        print(f"Clearing existing subsampled directory: {SUBSAMPLED_DIR}")
        shutil.rmtree(SUBSAMPLED_DIR)
        
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    for img_path, lbl_path in selected_pairs:
        shutil.copy2(img_path, os.path.join(dest_image_dir, os.path.basename(img_path)))
        shutil.copy2(lbl_path, os.path.join(dest_label_dir, os.path.basename(lbl_path)))
        
    print(f"Subsampled subset successfully copied to: {SUBSAMPLED_DIR}")
    print("-" * 70 + "\n")
    return dest_image_dir, dest_label_dir


def merge_and_prepare_dataset(daytime_dataset_dir, subsampled_image_dir, subsampled_label_dir, merged_dataset_dir):
    """
    Merges the subsampled low-light dataset with the existing daytime dataset.
    Maps class 5 ('person') in the low-light dataset to class 0, and filters out other classes.
    Writes the merged dataset to merged_dataset_dir.
    Generates a new data.yaml config.
    """
    print("=" * 70)
    print("            MERGING DAYTIME & LOW-LIGHT DATASETS")
    print("=" * 70)
    
    # Define merged paths
    merged_train_img_dir = os.path.join(merged_dataset_dir, "train", "images")
    merged_train_lbl_dir = os.path.join(merged_dataset_dir, "train", "labels")
    merged_val_img_dir = os.path.join(merged_dataset_dir, "val", "images")
    merged_val_lbl_dir = os.path.join(merged_dataset_dir, "val", "labels")
    
    # Clear existing merged folder
    if os.path.exists(merged_dataset_dir):
        print(f"Clearing existing merged directory: {merged_dataset_dir}")
        shutil.rmtree(merged_dataset_dir)
        
    os.makedirs(merged_train_img_dir, exist_ok=True)
    os.makedirs(merged_train_lbl_dir, exist_ok=True)
    os.makedirs(merged_val_img_dir, exist_ok=True)
    os.makedirs(merged_val_lbl_dir, exist_ok=True)
    
    # 1. Copy original daytime training dataset
    daytime_train_img_src = os.path.join(daytime_dataset_dir, "train", "images")
    daytime_train_lbl_src = os.path.join(daytime_dataset_dir, "train", "labels")
    
    daytime_train_imgs = os.listdir(daytime_train_img_src)
    for img_file in daytime_train_imgs:
        shutil.copy2(os.path.join(daytime_train_img_src, img_file), os.path.join(merged_train_img_dir, img_file))
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(daytime_train_lbl_src, lbl_file)
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(merged_train_lbl_dir, lbl_file))
            
    print(f"Copied {len(daytime_train_imgs)} daytime training images + labels.")
    
    # 2. Copy and map subsampled low-light training dataset
    lowlight_imgs = os.listdir(subsampled_image_dir)
    mapped_count = 0
    for img_file in lowlight_imgs:
        shutil.copy2(os.path.join(subsampled_image_dir, img_file), os.path.join(merged_train_img_dir, img_file))
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(subsampled_label_dir, lbl_file)
        if os.path.exists(lbl_path):
            # Read, map class 5 to class 0, discard others
            mapped_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id == 5:  # 'person'
                            parts[0] = "0"
                            mapped_lines.append(" ".join(parts) + "\n")
            # Write mapped label file
            with open(os.path.join(merged_train_lbl_dir, lbl_file), "w", encoding="utf-8") as f:
                f.writelines(mapped_lines)
            mapped_count += 1
            
    print(f"Copied and mapped {mapped_count} low-light training images + labels (mapped class 5 -> 0).")
    
    # 3. Copy daytime validation dataset to merged validation folder
    daytime_val_img_src = os.path.join(daytime_dataset_dir, "valid", "images")
    daytime_val_lbl_src = os.path.join(daytime_dataset_dir, "valid", "labels")
    
    daytime_val_imgs = os.listdir(daytime_val_img_src)
    for img_file in daytime_val_imgs:
        shutil.copy2(os.path.join(daytime_val_img_src, img_file), os.path.join(merged_val_img_dir, img_file))
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(daytime_val_lbl_src, lbl_file)
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(merged_val_lbl_dir, lbl_file))
            
    print(f"Copied {len(daytime_val_imgs)} daytime validation images + labels.")
    
    # 4. Generate merged data.yaml
    orig_yaml_path = os.path.join(daytime_dataset_dir, "data.yaml")
    if os.path.exists(orig_yaml_path):
        with open(orig_yaml_path, "r") as f:
            orig_data = yaml.safe_load(f)
        nc = orig_data.get("nc", 1)
        names = orig_data.get("names", ["person"])
    else:
        nc = 1
        names = ["person"]
        
    merged_yaml_data = {
        "path": merged_dataset_dir.replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": nc,
        "names": names
    }
    
    merged_yaml_path = os.path.join(merged_dataset_dir, "data.yaml")
    with open(merged_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(merged_yaml_data, f)
        
    print(f"Generated merged data.yaml at: {merged_yaml_path}")
    print("-" * 70 + "\n")
    return merged_yaml_path


def main():
    # 0. Subsample and merge datasets
    subsampled_img_dir, subsampled_lbl_dir = subsample_dataset(
        source_image_dir=SOURCE_IMAGE_DIR,
        source_label_dir=SOURCE_LABEL_DIR,
        target_count=TARGET_COUNT,
        seed=SEED
    )
    
    DATA_YAML = merge_and_prepare_dataset(
        daytime_dataset_dir=DAYTIME_DATASET_DIR,
        subsampled_image_dir=subsampled_img_dir,
        subsampled_label_dir=subsampled_lbl_dir,
        merged_dataset_dir=MERGED_DATASET_DIR
    )

    # =====================================================================
    # CONFIGURATION & PARAMETERS
    # =====================================================================
    # Training hyperparameters
    EPOCHS = 100
    
    # PARAMETER EXPLANATION: imgsz
    # We set imgsz=960 (higher resolution than the default 640) because crowd scenes 
    # typically contain small, distant, and densely packed people. A higher resolution 
    # provides more pixel detail to help the model resolve these fine features.
    IMGSZ = 960
    
    # PARAMETER EXPLANATION: batch
    # We use batch=2 (reduced from 4) to prevent Out-Of-Memory (OOM) 
    # errors on the 8GB RTX 4060 Laptop GPU. The 960px image size significantly increases 
    # the GPU memory footprint during forward and backward passes.
    BATCH_SIZE = 2
    
    # PARAMETER EXPLANATION: patience
    # We set patience=20 to stop training early if the validation loss plateaus for 
    # 20 consecutive epochs, preventing overfitting and saving training time.
    PATIENCE = 20
    
    DEVICE = 0       # Use GPU device 0
    WORKERS = 0      # Set to 0 to avoid multi-processing deadlocks/leaks on Windows
    PROJECT = "runs/detect"
    RUN_NAME = "train_yolo11m_960px"
    # =====================================================================

    print("=" * 70)
    print("            YOLOv11m CROWD DETECTOR TRAINING STARTUP")
    print("=" * 70)
    print(f"Model Name / Size:  {MODEL_NAME} (Medium)")
    print(f"Image Resolution:   {IMGSZ} px")
    print(f"Batch Size:         {BATCH_SIZE}")
    print(f"Dataset YAML Path:  {DATA_YAML}")
    print(f"Total Max Epochs:   {EPOCHS}")
    print(f"Patience:           {PATIENCE} epochs")
    print("-" * 70)

    # 1. Verify CUDA / GPU status
    cuda_available = torch.cuda.is_available()
    print(f"CUDA GPU Available: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(DEVICE)
        total_mem = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
        print(f"GPU Name:           {gpu_name}")
        print(f"Total VRAM:         {total_mem:.2f} GB")
    else:
        print("WARNING: CUDA GPU is not available! Training on CPU will be extremely slow.")
    print("=" * 70 + "\n")

    # Double check dataset path exists
    if not os.path.exists(DATA_YAML):
        print(f"CRITICAL ERROR: Dataset YAML not found at '{DATA_YAML}'.")
        print("Please verify the folder path matches your local setup.")
        sys.exit(1)

    # 2. Resume vs Fresh Run Check
    # Check whether a previous training run checkpoint 'last.pt' exists
    possible_lasts = [
        os.path.join(PROJECT, RUN_NAME, "weights", "last.pt"),
        os.path.join(PROJECT, PROJECT, RUN_NAME, "weights", "last.pt"),
        os.path.join("runs", "detect", RUN_NAME, "weights", "last.pt"),
        os.path.join("runs", "detect", "runs", "detect", RUN_NAME, "weights", "last.pt")
    ]
    
    last_pt_path = None
    for path in possible_lasts:
        if os.path.exists(path):
            last_pt_path = path
            break
            
    resume_training = False
    if last_pt_path:
        print("=" * 70)
        print("                 EXISTING CHECKPOINT DETECTED")
        print("=" * 70)
        print(f"Found previous run checkpoint at: {last_pt_path}")
        print("Options:")
        print("  [R] Resume training from this checkpoint (continues from last saved epoch)")
        print("  [F] Start fresh (renames the existing run directory to prevent overwriting)")
        print("=" * 70)
        
        choice = ""
        while choice not in ['r', 'f']:
            try:
                choice = input("Enter choice [R/F]: ").strip().lower()
            except KeyboardInterrupt:
                print("\nTraining aborted.")
                sys.exit(0)
                
        if choice == 'r':
            resume_training = True
            # Print epoch details from checkpoint
            try:
                ckpt = torch.load(last_pt_path, map_location='cpu')
                last_completed_epoch = ckpt.get('epoch', -1)
                resuming_epoch = last_completed_epoch + 2
                print(f"\nResuming training from Epoch {resuming_epoch} (after completed Epoch {last_completed_epoch + 1}).")
            except Exception as e:
                print(f"\nCould not parse epoch details from checkpoint: {e}. Resuming training...")
        else:
            # START FRESH: rename the existing run directory to avoid overwriting it
            run_dir = os.path.dirname(os.path.dirname(last_pt_path))
            counter = 1
            new_run_dir = f"{run_dir}_old"
            while os.path.exists(new_run_dir):
                new_run_dir = f"{run_dir}_old_{counter}"
                counter += 1
                
            print(f"\nRenaming existing run directory from '{run_dir}' to '{new_run_dir}'...")
            try:
                shutil.move(run_dir, new_run_dir)
                print("Directory renamed successfully.")
            except Exception as e:
                print(f"CRITICAL ERROR renaming directory: {e}")
                sys.exit(1)
            last_pt_path = None

    # NOTE ON RESUMABILITY (Requirement 5):
    # YOLOv11m saves the 'last.pt' checkpoint at the end of each completed epoch.
    # To pause training safely, you can interrupt the process (Ctrl+C in terminal).
    # Letting the current epoch finish before stopping is recommended, as interrupting
    # mid-epoch loses the current in-progress epoch's work, but the previous completed
    # epoch's last.pt remains safe.
    print("\nNOTE ON RESUMABILITY:")
    print("  YOLOv11m saves 'last.pt' at the end of each completed epoch.")
    print("  To pause training safely, interrupt the process (Ctrl+C).")
    print("  Interruption mid-epoch loses the current epoch's progress,")
    print("  but the previous completed epoch's progress remains safe in last.pt.")
    print("-" * 70)

    # 3. Load/Initialize model
    if resume_training and last_pt_path:
        print(f"Loading checkpoint weights from '{last_pt_path}' to resume...")
        try:
            model = YOLO(last_pt_path)
        except Exception as e:
            print(f"ERROR loading checkpoint model: {e}")
            sys.exit(1)
    else:
        print(f"Initializing fresh base model '{MODEL_NAME}'...")
        try:
            model = YOLO(MODEL_NAME)
        except Exception as e:
            print(f"ERROR: Failed to initialize model {MODEL_NAME}: {e}")
            sys.exit(1)

    # 4. Train the model
    print("\nStarting training loop...")
    try:
        if resume_training and last_pt_path:
            # Ultralytics resume=True restores all original args (imgsz, batch, epochs, etc.) automatically.
            model.train(resume=True)
        else:
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH_SIZE,
                device=DEVICE,
                workers=WORKERS,
                patience=PATIENCE,
                project=PROJECT,
                name=RUN_NAME,
                exist_ok=True
            )
        
        # 4. Copy best checkpoint and log completion
        best_weights_dest = os.path.join("models", "yolo11m_best.pt")
        
        # Check multiple possible locations due to ultralytics path concatenation behavior on Windows
        possible_sources = [
            os.path.join(PROJECT, RUN_NAME, "weights", "best.pt"),
            os.path.join(PROJECT, PROJECT, RUN_NAME, "weights", "best.pt"),
            os.path.join("runs", "detect", RUN_NAME, "weights", "best.pt"),
            os.path.join("runs", "detect", "runs", "detect", RUN_NAME, "weights", "best.pt")
        ]
        
        best_weights_source = None
        for src in possible_sources:
            if os.path.exists(src):
                best_weights_source = src
                break
                
        print("\n" + "=" * 70)
        print("                 YOLOv11m TRAINING COMPLETED")
        print("=" * 70)
        
        if best_weights_source:
            print(f"Best model checkpoint: {best_weights_source}")
            os.makedirs("models", exist_ok=True)
            shutil.copy(best_weights_source, best_weights_dest)
            print(f"Copied weights to:     {best_weights_dest}")
        else:
            print(f"WARNING: Could not locate the best.pt weights file to copy. Searched: {possible_sources}")
        print("=" * 70)

    except Exception as e:
        # 5. Out-of-memory and general error handling
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print("\n" + "!" * 70)
            print("                CUDA OUT-OF-MEMORY (OOM) ERROR")
            print("!" * 70)
            print("The GPU ran out of memory while allocating tensors at 960px resolution.")
            print("Suggestions to resolve this OOM error:")
            print("  1. Reduce the batch size further (e.g. set batch=2).")
            print("  2. Reduce the image size (e.g. set imgsz=800 or imgsz=640).")
            print("  3. Close other applications using GPU memory.")
            print("!" * 70)
        else:
            print(f"\nERROR occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
