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
import sys

# Append the project workspace root to the python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use Agg backend for matplotlib to prevent graphical issues in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom JHU and ShanghaiTech datasets
from data.jhu_dataset import JHUCrowdDataset, ShanghaiTechDataset
from src.csrnet_model import load_csrnet_model, CSRNet
from losses.dm_count_loss import DMCountLoss

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet on JHU-CROWD++ or ShanghaiTech datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['jhu', 'shanghai_a', 'shanghai_b'],
                        help="The dataset type: 'jhu', 'shanghai_a', or 'shanghai_b'.")
    parser.add_argument('--train_img_dir', type=str, required=True, help="Path to training images directory.")
    parser.add_argument('--train_gt_dir', type=str, required=True, help="Path to training ground-truth directory.")
    parser.add_argument('--val_img_dir', type=str, default=None, help="Path to validation images directory (optional).")
    parser.add_argument('--val_gt_dir', type=str, default=None, help="Path to validation ground-truth directory (optional).")
    
    parser.add_argument('--epochs', type=int, default=60, help="Number of epochs to train (default: 60).")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size (default: 2).")
    parser.add_argument('--lr', type=float, default=1e-4, help="Starting learning rate (default: 1e-4).")
    parser.add_argument('--max_size', type=int, default=1024, help="Maximum image dimension during evaluation (default: 1024).")
    parser.add_argument('--crop_size', type=int, default=384, help="Random crop size for training (default: 384).")
    parser.add_argument('--filter_blur', action='store_true', help="Filter out heavily blurred images from the JHU training set.")
    parser.add_argument('--patience', type=int, default=30, help="Patience epochs for early stopping (default: 30).")
    parser.add_argument('--weights', type=str, default="models/csrnet_shanghaitech.pth",
                        help="Path to baseline pretrained weights (default: models/csrnet_shanghaitech.pth).")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument('--loss', type=str, default='dm_count', choices=['mse', 'dm_count'],
                        help="Loss function to use: 'mse' or 'dm_count' (default: 'dm_count').")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau'],
                        help="Learning rate scheduler: 'cosine' (with 5-epoch warmup) or 'plateau'.")
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

def dm_count_collate(batch):
    transposed = list(zip(*batch))
    images = torch.stack(transposed[0], 0)
    points = transposed[1] # list of tensors
    counts = torch.tensor(transposed[2], dtype=torch.float32)
    return images, points, counts

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
            crop_size=(args.crop_size, args.crop_size),
            return_points=(args.loss == 'dm_count')
        )
        
        # Validation loader
        val_img_dir = args.val_img_dir if args.val_img_dir else args.train_img_dir
        val_gt_dir = args.val_gt_dir if args.val_gt_dir else args.train_gt_dir
        val_dataset = JHUCrowdDataset(
            images_dir=val_img_dir,
            gt_dir=val_gt_dir,
            max_size=args.max_size,
            filter_blur=False,
            is_train=False,
            return_points=(args.loss == 'dm_count')
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
            crop_size=(args.crop_size, args.crop_size),
            return_points=(args.loss == 'dm_count')
        )
        val_dataset = ShanghaiTechDataset(
            pairs=val_pairs,
            max_size=args.max_size,
            is_train=False,
            return_points=(args.loss == 'dm_count')
        )
        
    print(f"Dataset selected: {args.dataset.upper()}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    
    collate = dm_count_collate if args.loss == 'dm_count' else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
    
    # 3. Load Model
    if os.path.exists(args.weights):
        print(f"Loading baseline pretrained weights from '{args.weights}'...")
        model = load_csrnet_model(args.weights, device)
    else:
        print(f"Pretrained weights not found at '{args.weights}'. Initializing default CSRNet.")
        model = CSRNet(load_weights=False).to(device)
        
    # Freezing first 3 conv blocks of the VGG16 backbone frontend
    print("Freezing first 3 conv blocks of the VGG16 backbone frontend...")
    pool_count = 0
    for layer in model.frontend:
        if isinstance(layer, nn.MaxPool2d):
            pool_count += 1
        if pool_count < 3:
            for param in layer.parameters():
                param.requires_grad = False
                
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_frozen = sum(p.numel() for p in frozen_params)
    print(f"Model parameters: Trainable = {num_trainable:,} | Frozen = {num_frozen:,}")

    # 4. Set Loss & Optimizers
    if args.loss == 'dm_count':
        criterion = DMCountLoss(lambda_ot=0.1, lambda_tv=0.01, device=device)
    else:
        criterion = nn.MSELoss(reduction='sum')
        
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    if args.scheduler == 'cosine':
        warmup_epochs = 5
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return float(ep + 1) / float(warmup_epochs)
            progress = float(ep - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 5. Output configurations and files
    os.makedirs("models", exist_ok=True)
    suffix_path = "_dmcount" if args.loss == 'dm_count' else "_v2" if args.dataset == 'jhu' else ""
    best_weights_path = os.path.join("models", f"csrnet_{args.dataset}{suffix_path}_best.pth")
    log_csv_path = os.path.join("models", f"csrnet_{args.dataset}{suffix_path}_log.csv")
    plot_path = os.path.join("models", f"csrnet_{args.dataset}{suffix_path}_loss_curve.png")
    
    # Initialize Log CSV
    with open(log_csv_path, 'w', newline='', encoding='utf-8') as lf:
        writer = csv.writer(lf)
        if args.loss == 'dm_count':
            writer.writerow(['epoch', 'train_loss', 'train_cnt_loss', 'train_ot_loss', 'train_tv_loss', 'val_loss', 'val_mae', 'learning_rate'])
        else:
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
        train_cnt_loss = 0.0
        train_ot_loss = 0.0
        train_tv_loss = 0.0
        train_batches = len(train_loader)
        
        for batch_idx, batch_data in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            if args.loss == 'dm_count':
                inputs, points, counts = batch_data
                inputs = inputs.to(device)
                points = [pt.to(device) for pt in points]
                counts = counts.to(device)
                
                outputs = model(inputs)
                loss, loss_cnt, loss_ot, loss_tv = criterion(outputs, points, counts)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_cnt_loss += loss_cnt.item()
                train_ot_loss += loss_ot.item()
                train_tv_loss += loss_tv.item()
                
                suffix_str = f"Loss: {loss.item():.4f} (Cnt: {loss_cnt.item():.2f}, OT: {loss_ot.item():.3f}, TV: {loss_tv.item():.3f})"
            else:
                inputs, targets = batch_data
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                suffix_str = f"Loss: {loss.item():.6f}"
                
            print_progress_bar(batch_idx, train_batches, 
                               prefix=f"Epoch {epoch}/{args.epochs} [Train]", 
                               suffix=suffix_str)
                               
        avg_train_loss = train_loss / train_batches
        avg_train_cnt = train_cnt_loss / train_batches
        avg_train_ot = train_ot_loss / train_batches
        avg_train_tv = train_tv_loss / train_batches
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader, start=1):
                if args.loss == 'dm_count':
                    inputs, points, counts = batch_data
                    inputs = inputs.to(device)
                    points = [pt.to(device) for pt in points]
                    counts = counts.to(device)
                    
                    outputs = model(inputs)
                    loss, loss_cnt, loss_ot, loss_tv = criterion(outputs, points, counts)
                    val_loss += loss.item()
                    
                    for b in range(inputs.size(0)):
                        pred_count = outputs[b].sum().item()
                        true_count = counts[b].item()
                        val_mae += abs(pred_count - true_count)
                        
                    suffix_str = f"Loss: {loss.item():.4f}"
                else:
                    inputs, targets = batch_data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    for b in range(inputs.size(0)):
                        pred_count = outputs[b].sum().item()
                        true_count = targets[b].sum().item()
                        val_mae += abs(pred_count - true_count)
                        
                    suffix_str = f"Loss: {loss.item():.6f}"
                    
                print_progress_bar(batch_idx, val_batches, 
                                   prefix=f"Epoch {epoch}/{args.epochs} [Val]  ", 
                                   suffix=suffix_str)
                                   
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / len(val_dataset)
        
        old_lr = optimizer.param_groups[0]['lr']
        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_val_mae)
            if optimizer.param_groups[0]['lr'] < old_lr:
                print(f"Learning rate decayed from {old_lr:.1e} to {optimizer.param_groups[0]['lr']:.1e}. Resetting early stopping counter.")
                epochs_no_improve = 0
        new_lr = optimizer.param_groups[0]['lr']
            
        epoch_time = time.time() - epoch_start
        print(f"Summary -> Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val MAE: {avg_val_mae:.2f} | LR: {new_lr:.1e} | Time: {epoch_time:.1f}s")
        
        # Save metrics to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        # Log to CSV
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as lf:
            writer = csv.writer(lf)
            if args.loss == 'dm_count':
                writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_train_cnt:.6f}", f"{avg_train_ot:.6f}", f"{avg_train_tv:.6f}", f"{avg_val_loss:.6f}", f"{avg_val_mae:.2f}", f"{new_lr:.2e}"])
            else:
                writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{avg_val_mae:.2f}", f"{new_lr:.2e}"])
            
        # Plot Loss Curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
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
