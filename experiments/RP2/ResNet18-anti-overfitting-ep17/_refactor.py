import os

filepath = 'train_pneumonia.py'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('matplotlib.use("TkAgg")  # Use TkAgg backend so plt.show() opens a window', 
                    'matplotlib.use("Agg")  # Use Agg backend for background plotting without popups')
text = text.replace('EARLY_STOPPING_PATIENCE = 5', 'EARLY_STOPPING_PATIENCE = 30')

# Replacing plot_metrics
old_plot_metrics = '''def plot_metrics(history, save_dir):
    \"\"\"Precision + Recall + F1 + Specificity → same graph.\"\"\"
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["precision"]) + 1)
    ax.plot(epochs, history["precision"], "g-^", markersize=5, label="Precision")
    ax.plot(epochs, history["recall"], "m-s", markersize=5, label="Recall")
    ax.plot(epochs, history["f1"], "c-D", markersize=5, label="F1 Score")
    ax.plot(epochs, history["specificity"], "y-v", markersize=5, label="Specificity")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, F1 & Specificity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(save_dir, "metrics_plot.png")
    fig.savefig(path, dpi=150)
    print(f"  📊 Saved: {path}")
    return fig'''

new_plot_metrics = '''def plot_metric_single(history, metric_key, title, filename, color_marker, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history[metric_key]) + 1)
    ax.plot(epochs, history[metric_key], color_marker, markersize=5, label=title)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(title + " over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if metric_key != "lr":
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    print(f"  📊 Saved: {path}")
    plt.close(fig)
    return fig

def generate_individual_metrics(history, save_dir):
    plot_metric_single(history, "precision", "Precision", "precision_plot.png", "g-^", save_dir)
    plot_metric_single(history, "recall", "Recall", "recall_plot.png", "m-s", save_dir)
    plot_metric_single(history, "f1", "F1 Score", "f1_score_plot.png", "c-D", save_dir)
    plot_metric_single(history, "specificity", "Specificity", "specificity_plot.png", "y-v", save_dir)
    plot_metric_single(history, "lr", "Learning Rate", "learning_rate_plot.png", "k--.", save_dir)'''

text = text.replace(old_plot_metrics, new_plot_metrics)

old_graphs_call = '''    fig1 = plot_accuracy(history, GRAPHS_DIR)
    fig2 = plot_loss(history, GRAPHS_DIR)
    fig3 = plot_metrics(history, GRAPHS_DIR)
    fig4 = plot_confusion_mat(final_labels, final_preds, GRAPHS_DIR)'''

new_graphs_call = '''    plot_accuracy(history, GRAPHS_DIR)
    plot_loss(history, GRAPHS_DIR)
    generate_individual_metrics(history, GRAPHS_DIR)
    plot_confusion_mat(final_labels, final_preds, GRAPHS_DIR)'''

text = text.replace(old_graphs_call, new_graphs_call)

old_plt_show = '''    # Show plots (interactive window with save icon)
    print("\\n📊 Displaying plots (close windows to exit)...")
    plt.show()'''

new_plt_show = '''    print("\\n📊 8 Plots have been generated and saved silently in the background.")'''

text = text.replace(old_plt_show, new_plt_show)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("Refactoring complete.")
