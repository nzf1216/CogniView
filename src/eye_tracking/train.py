import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import ADEyeTrackingDataset
from model import ADClassifierModel

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training AD Eye-Tracking Screening Model on device: {device}")
    
    dataset = ADEyeTrackingDataset(args.data_dir)
    train_sz = int(0.8 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_set, val_set = random_split(dataset, [train_sz, val_sz])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    model = ADClassifierModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    
    # Modern PyTorch 2.x AMP Scaler initialization
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Modern PyTorch 2.x autocast context
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(seqs)
                loss = criterion(preds, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                val_loss += criterion(outputs, labels).item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_ad_tracker.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("🛑 Early stopping triggered. Optimal convergence reached.")
                break
                
    print(f"✅ Training completed. Best model saved to {args.save_dir}/best_ad_tracker.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/ad_eyetracking/synthetic")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4) # Optimized learning rate for attention stability
    parser.add_argument("--patience", type=int, default=10) # Increased patience window
    args = parser.parse_args()
    train_model(args)