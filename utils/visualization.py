import matplotlib.pyplot as plt

def plot_metrics(metrics, save_path_prefix):
    epochs = list(range(1, len(metrics['loss']) + 1))

    # Loss plot
    plt.figure()
    plt.plot(epochs, metrics['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(f"{save_path_prefix}_loss.png")
    plt.close()

    # Accuracy plot (if available)
    if 'clean_acc' in metrics and 'adv_acc' in metrics:
        plt.figure()
        plt.plot(epochs, metrics['clean_acc'], label='Clean Accuracy')
        plt.plot(epochs, metrics['adv_acc'], label='Adversarial Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path_prefix}_acc.png")
        plt.close()
