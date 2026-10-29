import json

with open("graphs/training_history.json") as f:
    data = json.load(f)

epochs = len(data["train_loss"])
print("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Precision | Recall | F1 Score | Specificity | LR |")
print("|---|---|---|---|---|---|---|---|---|---|")
for i in range(epochs):
    pt = [
        i+1,
        data["train_loss"][i],
        data["train_acc"][i],
        data["val_loss"][i],
        data["val_acc"][i],
        data["precision"][i],
        data["recall"][i],
        data["f1"][i],
        data["specificity"][i],
        data["lr"][i]
    ]
    print(f"| {pt[0]:02d} | {pt[1]:.4f} | {pt[2]:.4f} | {pt[3]:.4f} | {pt[4]:.4f} | {pt[5]:.4f} | {pt[6]:.4f} | {pt[7]:.4f} | {pt[8]:.4f} | {pt[9]:.6g} |")
