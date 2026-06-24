import os
import sys
import shutil
import torch
from ultralytics import YOLO

def main():
    # =====================================================================
    # CONFIGURATION & PARAMETERS
    # =====================================================================
    # Path to the Roboflow-exported crowd density dataset definition
    DATA_YAML = r"E:\Crowd Detection &  Prevention\crowd density.v1-v1.yolov8\data.yaml"
    
    # Model configuration
    MODEL_NAME = "yolo11m.pt"  # Medium YOLOv11 model (auto-downloads on first use)
    
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
