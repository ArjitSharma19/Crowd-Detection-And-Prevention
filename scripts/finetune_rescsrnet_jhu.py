import os
import re
import csv
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys

# Append the project workspace root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Headless matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Custom modules
from data.jhu_dataset import JHUCrowdDataset
from src.rescsrnet_model import ResCSRNet, load_rescsrnet_model
from losses.dm_count_loss import DMCountLoss

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResCSRNet (ResNet-50 backbone) on JHU-CROWD++ dataset.")
    parser.add_argument('--train_img_dir', type=str, required=True, help="Path to training images directory.")
    parser.add_argument('--train_gt_dir', type=str, required=True, help="Path to training ground-truth directory.")
    parser.add_argument('--val_img_dir', type=str, default=None, help="Path to validation images directory (optional).")
    parser.add_argument('--val_gt_dir', type=str, default=None, help="Path to validation ground-truth directory (optional).")
    
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs to train (default: 40).")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size (default: 2).")
    parser.add_argument('--lr', type=float, default=1e-4, help="Starting learning rate (default: 1e-4).")
    parser.add_argument('--max_size', type=int, default=1024, help="Maximum image dimension during evaluation (default: 1024).")
    parser.add_argument('--crop_size', type=int, default=384, help="Random crop size for training (default: 384).")
    parser.add_argument('--patience', type=int, default=25, help="Patience epochs for early stopping (default: 25).")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'.")
    return parser.parse_args()

def dm_count_collate(batch):
    transposed = list(zip(*batch))
    images = torch.stack(transposed[0], 0)
    points = transposed[1] # list of tensors
    counts = torch.tensor(transposed[2], dtype=torch.float32)
    return images, points, counts

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()

def main():
    args = get_args()
    
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets & Loaders
    train_dataset = JHUCrowdDataset(
        images_dir=args.train_img_dir,
        gt_dir=args.train_gt_dir,
        max_size=args.max_size,
        filter_blur=False,
        is_train=True,
        crop_size=(args.crop_size, args.crop_size),
        return_points=True
    )
    
    val_img_dir = args.val_img_dir if args.val_img_dir else args.train_img_dir
    val_gt_dir = args.val_gt_dir if args.val_gt_dir else args.train_gt_dir
    val_dataset = JHUCrowdDataset(
        images_dir=val_img_dir,
        gt_dir=val_gt_dir,
        max_size=args.max_size,
        filter_blur=False,
        is_train=False,
        return_points=True
    )
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=dm_count_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=dm_count_collate)
    
    # Initialize ResCSRNet
    print("Initializing ResCSRNet with ResNet-50 ImageNet backbone...")
    model = ResCSRNet(load_weights=False).to(device)
    
    # Freeze stem (conv1, bn1, layer1) of ResNet-50 frontend to stabilize initial convergence
    print("Freezing stem and layer1 of ResNet-50 backbone...")
    stem_layers = [model.frontend[0], model.frontend[1], model.frontend[4]] # conv1, bn1, layer1
    for m in stem_layers:
        for p in m.parameters():
            p.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    print(f"Model Parameters: Trainable = {sum(p.numel() for p in trainable_params):,} | Frozen = {sum(p.numel() for p in frozen_params):,}")
    
    # Loss & Differential Optimizer Setup
    criterion = DMCountLoss(lambda_ot=0.1, lambda_tv=0.01, device=device)
    
    # Differential learning rates: 0.1x for pretrained ResNet-50 frontend, 1.0x for dilated backend & output layer
    frontend_params = [p for p in model.frontend.parameters() if p.requires_grad]
    backend_params = list(model.backend.parameters()) + list(model.output_layer.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': frontend_params, 'lr': args.lr * 0.1},
        {'params': backend_params, 'lr': args.lr}
    ], weight_decay=1e-4)
    
    # Scheduler with increased patience (8 epochs) to prevent premature LR decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    
    # Checkpoint paths
    os.makedirs("models", exist_ok=True)
    best_weights_path = os.path.join("models", "rescsrnet_jhu_dmcount_best.pth")
    log_csv_path = os.path.join("models", "rescsrnet_jhu_log.csv")
    plot_path = os.path.join("models", "rescsrnet_jhu_loss_curve.png")
    
    with open(log_csv_path, 'w', newline='', encoding='utf-8') as lf:
        writer = csv.writer(lf)
        writer.writerow(['epoch', 'train_loss', 'train_cnt_loss', 'train_ot_loss', 'train_tv_loss', 'val_loss', 'val_mae', 'learning_rate'])
        
    best_val_mae = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    print("\n" + "="*70)
    print("                 STARTING ResCSRNet TRAINING LOOP")
    print("="*70)
    print(f"Starting LR:    {args.lr}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Total Epochs:   {args.epochs}")
    print(f"Checkpoint:     {best_weights_path}")
    print("="*70 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_cnt, train_ot, train_tv = 0.0, 0.0, 0.0, 0.0
        train_batches = len(train_loader)
        
        for batch_idx, (inputs, points, counts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            points = [pt.to(device) for pt in points]
            counts = counts.to(device)
            
            outputs = model(inputs)
            loss, loss_cnt, loss_ot, loss_tv = criterion(outputs, points, counts)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_cnt += loss_cnt.item()
            train_ot += loss_ot.item()
            train_tv += loss_tv.item()
            
            suffix_str = f"Loss: {loss.item():.4f} (Cnt: {loss_cnt.item():.2f}, OT: {loss_ot.item():.3f}, TV: {loss_tv.item():.3f})"
            print_progress_bar(batch_idx, train_batches, prefix=f"Epoch {epoch}/{args.epochs} [Train]", suffix=suffix_str)
            
        avg_train_loss = train_loss / train_batches
        avg_train_cnt = train_cnt / train_batches
        avg_train_ot = train_ot / train_batches
        avg_train_tv = train_tv / train_batches
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, points, counts) in enumerate(val_loader, start=1):
                inputs = inputs.to(device)
                points = [pt.to(device) for pt in points]
                counts = counts.to(device)
                
                outputs = model(inputs)
                loss, _, _, _ = criterion(outputs, points, counts)
                val_loss += loss.item()
                
                for b in range(inputs.size(0)):
                    pred_count = outputs[b].sum().item()
                    true_count = counts[b].item()
                    val_mae += abs(pred_count - true_count)
                    
                print_progress_bar(batch_idx, val_batches, prefix=f"Epoch {epoch}/{args.epochs} [Val]  ", suffix=f"Loss: {loss.item():.4f}")
                
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / len(val_dataset)
        
        scheduler.step(avg_val_mae)
        new_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        print(f"Summary -> Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.2f} | LR: {new_lr:.1e} | Time: {epoch_time:.1f}s")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        with open(log_csv_path, 'a', newline='', encoding='utf-8') as lf:
            writer = csv.writer(lf)
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_train_cnt:.6f}", f"{avg_train_ot:.6f}", f"{avg_train_tv:.6f}", f"{avg_val_loss:.6f}", f"{avg_val_mae:.2f}", f"{new_lr:.2e}"])
            
        # Plot curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('ResCSRNet Loss Curve')
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
        
        # Save best model
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"-> *** NEW BEST ResCSRNet MODEL (MAE: {best_val_mae:.2f}) *** saved to {best_weights_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs. No validation MAE improvement for {args.patience} epochs.")
                break
                
        print("-" * 70)
        
    print("\n" + "="*70)
    print("               ResCSRNet TRAINING COMPLETED")
    print("="*70)
    print(f"Best Validation MAE: {best_val_mae:.2f}")
    print(f"Best Checkpoint:     {best_weights_path}")
    print("="*70)

if __name__ == "__main__":
    main()
