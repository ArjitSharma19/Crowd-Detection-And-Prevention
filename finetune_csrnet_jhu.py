import os
import re
import csv
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader

# Use Agg backend for matplotlib to prevent graphical issues in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom JHU and ShanghaiTech datasets
from data.jhu_dataset import JHUCrowdDataset, ShanghaiTechDataset
from src.csrnet_model import load_csrnet_model, CSRNet

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet on JHU-CROWD++ or ShanghaiTech datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['jhu', 'shanghai_a', 'shanghai_b'],
                        help="The dataset type: 'jhu', 'shanghai_a', or 'shanghai_b'.")
    parser.add_argument('--train_img_dir', type=str, required=True, help="Path to training images directory.")
    parser.add_argument('--train_gt_dir', type=str, required=True, help="Path to training ground-truth directory.")
    parser.add_argument('--val_img_dir', type=str, default=None, help="Path to validation images directory (optional).")
    parser.add_argument('--val_gt_dir', type=str, default=None, help="Path to validation ground-truth directory (optional).")
    
    parser.add_argument('--epochs', type=int, default=150, help="Number of epochs to train (default: 150).")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size (default: 2).")
    parser.add_argument('--lr', type=float, default=1e-4, help="Starting learning rate (default: 1e-4).")
    parser.add_argument('--max_size', type=int, default=1024, help="Maximum image dimension during evaluation (default: 1024).")
    parser.add_argument('--crop_size', type=int, default=512, help="Random crop size for training (default: 512).")
    parser.add_argument('--filter_blur', action='store_true', help="Filter out heavily blurred images from the JHU training set.")
    parser.add_argument('--patience', type=int, default=15, help="Patience epochs for early stopping (default: 15).")
    parser.add_argument('--weights', type=str, default="models/csrnet_shanghaitech.pth",
                        help="Path to baseline pretrained weights (default: models/csrnet_shanghaitech.pth).")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'.")
    return parser.parse_args()

def align_shanghaitech_files(img_dir, gt_dir):
    all_imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_pairs = []
    for img_path in all_imgs:
        img_name = os.path.basename(img_path)
        match = re.search(r'IMG_\d+', img_name)
        if match:
            img_id = match.group(0)
            gt_filename = f"GT_{img_id}.mat"
            gt_path = os.path.join(gt_dir, gt_filename)
            if os.path.exists(gt_path):
                valid_pairs.append((img_path, gt_path))
    return valid_pairs

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()

def main():
    args = get_args()
    
    # Setup seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Build Datasets & Loaders
    if args.dataset == 'jhu':
        # Train loader
        train_dataset = JHUCrowdDataset(
            images_dir=args.train_img_dir,
            gt_dir=args.train_gt_dir,
            max_size=args.max_size,
            filter_blur=args.filter_blur,
            is_train=True,
            crop_size=(args.crop_size, args.crop_size)
        )
        
        # Validation loader
        val_img_dir = args.val_img_dir if args.val_img_dir else args.train_img_dir
        val_gt_dir = args.val_gt_dir if args.val_gt_dir else args.train_gt_dir
        val_dataset = JHUCrowdDataset(
            images_dir=val_img_dir,
            gt_dir=val_gt_dir,
            max_size=args.max_size,
            filter_blur=False,
            is_train=False
        )
    else:  # shanghai_a or shanghai_b
        # Gather all pairs
        train_pairs = align_shanghaitech_files(args.train_img_dir, args.train_gt_dir)
        if not train_pairs:
            raise FileNotFoundError(f"No matching ShanghaiTech images and annotations found in {args.train_img_dir}")
            
        if args.val_img_dir and args.val_gt_dir:
            val_pairs = align_shanghaitech_files(args.val_img_dir, args.val_gt_dir)
        else:
            # Deterministic split (80% Train, 20% Val)
            indices = list(range(len(train_pairs)))
            np.random.shuffle(indices)
            val_size = int(0.2 * len(train_pairs))
            train_size = len(train_pairs) - val_size
            
            val_pairs = [train_pairs[i] for i in indices[train_size:]]
            train_pairs = [train_pairs[i] for i in indices[:train_size]]
            
        train_dataset = ShanghaiTechDataset(
            pairs=train_pairs,
            max_size=args.max_size,
            is_train=True,
            crop_size=(args.crop_size, args.crop_size)
        )
        val_dataset = ShanghaiTechDataset(
            pairs=val_pairs,
            max_size=args.max_size,
            is_train=False
        )
        
    print(f"Dataset selected: {args.dataset.upper()}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. Load Model
    if os.path.exists(args.weights):
        print(f"Loading baseline pretrained weights from '{args.weights}'...")
        model = load_csrnet_model(args.weights, device)
    else:
        print(f"Pretrained weights not found at '{args.weights}'. Initializing default CSRNet.")
        model = CSRNet(load_weights=False).to(device)
        
    # 4. Set Loss & Optimizers
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 5. Output configurations and files
    os.makedirs("models", exist_ok=True)
    best_weights_path = os.path.join("models", f"csrnet_{args.dataset}_best.pth")
    log_csv_path = os.path.join("models", f"csrnet_{args.dataset}_log.csv")
    plot_path = os.path.join("models", f"csrnet_{args.dataset}_loss_curve.png")
    
    # Initialize Log CSV
    with open(log_csv_path, 'w', newline='', encoding='utf-8') as lf:
        writer = csv.writer(lf)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'learning_rate'])
        
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }
    
    print("\n" + "="*70)
    print("                     STARTING TRAINING LOOP")
    print("="*70)
    print(f"Starting LR:    {args.lr}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Total Epochs:   {args.epochs}")
    print(f"Patience:       {args.patience} epochs")
    print(f"Weights path:   {best_weights_path}")
    print("*" * 70 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_batches = len(train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            print_progress_bar(batch_idx, train_batches, 
                               prefix=f"Epoch {epoch}/{args.epochs} [Train]", 
                               suffix=f"Loss: {loss.item():.6f}")
                               
        avg_train_loss = train_loss / train_batches
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Compute MAE
                for b in range(inputs.size(0)):
                    pred_count = outputs[b].sum().item()
                    true_count = targets[b].sum().item()
                    val_mae += abs(pred_count - true_count)
                    
                print_progress_bar(batch_idx, val_batches, 
                                   prefix=f"Epoch {epoch}/{args.epochs} [Val]  ", 
                                   suffix=f"Loss: {loss.item():.6f}")
                                   
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / len(val_dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_mae)
        
        epoch_time = time.time() - epoch_start
        print(f"Summary -> Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val MAE: {avg_val_mae:.2f} | LR: {current_lr:.1e} | Time: {epoch_time:.1f}s")
        
        # Save metrics to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        # Log to CSV
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as lf:
            writer = csv.writer(lf)
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{avg_val_mae:.2f}", f"{current_lr:.2e}"])
            
        # Plot Loss Curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('MSE Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_mae'], label='Val MAE', color='orange')
        plt.title('Validation MAE Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        # Early Stopping check
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"-> *** NEW BEST MODEL (MAE: {best_val_mae:.2f}) *** saved to {best_weights_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs. No validation MAE improvement for {args.patience} epochs.")
                break
                
        print("-" * 70)
        
    print("\n" + "="*70)
    print("                    TRAINING COMPLETED")
    print("="*70)
    print(f"Best Validation MAE: {best_val_mae:.2f}")
    print(f"Best Weights Saved:  {best_weights_path}")
    print(f"Log CSV Saved:       {log_csv_path}")
    print(f"Curves Plot Saved:   {plot_path}")
    print("="*70)

if __name__ == "__main__":
    main()
