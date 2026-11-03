import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss

# Use system default font instead of Times New Roman
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def show_point(max_id: int, data_list: list, x_list: list = None):
    x_value = x_list[max_id] if x_list is not None else max_id + 1
    show_max = f"({x_value}, {round(data_list[max_id], 2)})"
    plt.annotate(
        show_max,
        xytext=(x_value, data_list[max_id]),
        xy=(x_value, data_list[max_id]),
        fontsize=6,
    )


def smooth(y: list):
    if 95 <= len(y):
        return ss.savgol_filter(y, 95, 3)

    return y


def plot_acc(tra_acc_list: list, val_acc_list: list, save_path: str, epoch_list: list = None):
    x_acc = epoch_list if epoch_list is not None else [i + 1 for i in range(len(tra_acc_list))]

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)
    plt.title("Accuracy of training and validation", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(x[max1], y1[max1], "r-o")
    plt.plot(x[max2], y2[max2], "r-o")
    show_point(max1, y1, x_acc)
    show_point(max2, y2, x_acc)
    plt.legend()
    plt.savefig(save_path + "/acc.pdf", bbox_inches="tight")
    plt.close()


def plot_loss(loss_list: list, save_path: str):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title("Loss curve", fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(x_loss, smooth(loss_list))
    plt.savefig(save_path + "/loss.pdf", bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels_name: list,
    save_path: str,
    title="Confusion matrix",
    filename="mat.pdf",
):
    cm = cm.astype("float")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")  # Use the Blues colormap.
    plt.title(title, fontweight="bold")  # Figure title.
    plt.colorbar()
    count = len(labels_name)
    num_local = np.array(range(count))
    if count <= 20:
        # Add values to confusion-matrix cells.
        for i in range(count):
            for j in range(count):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], ".2f"),
                    horizontalalignment="center",
                    color="black" if cm[i, j] <= 0.5 else "white",
                )  # Choose text color according to cell brightness.
        # Print labels on the x-axis.
        plt.xticks(num_local, labels_name, rotation=45)
        # Print labels on the y-axis.
        plt.yticks(num_local, labels_name)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path + f"/{filename}", bbox_inches="tight")
    plt.close()
