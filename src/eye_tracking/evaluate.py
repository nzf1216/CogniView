import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from dataset import ADEyeTrackingDataset
from model import ADClassifierModel

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ADEyeTrackingDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    model = ADClassifierModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    true_labels = []
    pred_probabilities = []
    
    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            true_labels.extend(labels.numpy())
            pred_probabilities.extend(probs)
            
    true_labels = np.array(true_labels)
    pred_probabilities = np.array(pred_probabilities)
    pred_classes = (pred_probabilities >= 0.5).astype(int)
    
    print("\n================ CLINICAL SCREENING REPORT ================")
    print(classification_report(true_labels, pred_classes, target_names=["Healthy Control", "MCI / AD Risk"]))
    
    auc = roc_auc_score(true_labels, pred_probabilities)
    print(f"Diagnostic ROC-AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(true_labels, pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'ad_screening_roc.png')
    
    # Generate publication-grade smooth ROC curve plot
    fpr, tpr, _ = roc_curve(true_labels, pred_probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Oculomotor Marker (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('CogniView: AD/MCI Eye-Tracking Screening Performance', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ High-resolution ROC curve successfully saved to '{save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/ad_eyetracking/synthetic")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_ad_tracker.pth")
    parser.add_argument("--output_dir", type=str, default="./results/plots", help="Directory to save evaluation plots")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate_model(args)