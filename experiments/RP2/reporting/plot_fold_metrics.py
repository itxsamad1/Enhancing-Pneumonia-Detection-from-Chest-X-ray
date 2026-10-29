import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

RESULTS_FILE = "kfold_results.json"
PLOTS_DIR = "kfold_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Loading results from {RESULTS_FILE}...")
with open(RESULTS_FILE) as f:
    results = json.load(f)

# Markdown summary table header
markdown_report = "# 5-Fold Cross-Validation: Detailed Per-Fold Metrics\n\n"
markdown_report += "This document lists the confusion matrix components (TN, FP, FN, TP) and Recall (Sensitivity) for each fold across all 10 architectures.\n\n"

for model_key, model_data in results.items():
    if model_key == "mcnemar" or "folds" not in model_data:
        continue
    
    display_name = model_data.get("display", model_key)
    print(f"\nProcessing model: {display_name}...")
    
    markdown_report += f"## {display_name}\n\n"
    markdown_report += "| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |\n"
    markdown_report += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    plt.figure(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(1, 6):
        fold_key = f"fold_{i}"
        if fold_key not in model_data["folds"]:
            print(f"  [WARN] Fold {i} missing for {model_key}")
            continue
            
        fold_data = model_data["folds"][fold_key]
        preds = np.array(fold_data["preds"])
        labels = np.array(fold_data["labels"])
        
        # Compute Confusion Matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Compute Recall (Sensitivity)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Format markdown row
        markdown_report += f"| Fold {i} | {tn} | {fp} | {fn} | {tp} | {recall:.4f} |\n"
        
        # Plot Binary ROC Curve
        # True Negative Rate (Specificity) = TN / (TN + FP)
        # False Positive Rate = FP / (FP + TN)
        # True Positive Rate (Sensitivity / Recall) = TP / (TP + FN)
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # ROC points: (0,0) -> (FPR, TPR) -> (1,1)
        plt.plot([0, fpr, 1], [0, tpr, 1], marker='o', linestyle='-', color=colors[i-1], 
                 label=f"Fold {i} (AUC: {fold_data.get('auc', 0.0):.3f}, Rec: {recall:.3f})")
                 
    # Set plot styling
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title(f"ROC Curves per Fold — {display_name}", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save ROC plot
    plot_path = os.path.join(PLOTS_DIR, f"{model_key}_kfold_roc.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    markdown_report += "\n"

# Write markdown report
report_path = "kfold_metrics_summary.md"
with open(report_path, "w") as f:
    f.write(markdown_report)
print(f"\nSummary report saved to: {report_path}")
print(f"Per-fold ROC plots saved in directory: {PLOTS_DIR}")
