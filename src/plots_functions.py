import matplotlib.pyplot as plt
import numpy as np


def multiple_runs_with_uncertainty_band(metrics_list, title_accuracy, title_loss):

    plt.figure(figsize=(10, 6))

    # Prepare lists of train and test accuracies for uncertainty bands
    train_acc_list = [metrics["train_acc"] for metrics in metrics_list]
    test_acc_list = [metrics["test_acc"] for metrics in metrics_list]
    epochs = len(train_acc_list[0])
    num_of_runs = len(train_acc_list)

    avg_train_acc = np.mean(train_acc_list, axis=0)
    std_train_acc = np.std(train_acc_list, axis=0)

    avg_test_acc = np.mean(test_acc_list, axis=0)
    std_test_acc = np.std(test_acc_list, axis=0)

    plt.fill_between(
        range(epochs),
        avg_train_acc - std_train_acc,
        avg_train_acc + std_train_acc,
        color="blue",
        alpha=0.2,
        label="Train Accuracy Uncertainty",
    )

    plt.fill_between(
        range(epochs),
        avg_test_acc - std_test_acc,
        avg_test_acc + std_test_acc,
        color="red",
        alpha=0.2,
        label="Test Accuracy Uncertainty",
    )

    plt.plot(
        range(epochs),
        avg_train_acc,
        label=f"Avg Train Accuracy over {num_of_runs} runs",
        color="blue",
        linestyle="--",
        linewidth=4,
    )
    plt.plot(
        range(epochs),
        avg_test_acc,
        label=f"Avg Test Accuracy over {num_of_runs} runs",
        color="red",
        linestyle="--",
        linewidth=4,
    )

    plt.title(title_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()

    plt.figure(figsize=(10, 6))

    train_loss_list = [metrics["train_loss"] for metrics in metrics_list]
    test_loss_list = [metrics["test_loss"] for metrics in metrics_list]

    avg_train_loss = np.mean(train_loss_list, axis=0)
    std_train_loss = np.std(train_loss_list, axis=0)

    avg_test_loss = np.mean(test_loss_list, axis=0)
    std_test_loss = np.std(test_loss_list, axis=0)

    plt.fill_between(
        range(epochs),
        avg_train_loss - std_train_loss,
        avg_train_loss + std_train_loss,
        color="blue",
        alpha=0.2,
        label="Train Loss Uncertainty",
    )

    plt.fill_between(
        range(epochs),
        avg_test_loss - std_test_loss,
        avg_test_loss + std_test_loss,
        color="red",
        alpha=0.2,
        label="Test Loss Uncertainty",
    )

    plt.plot(
        range(epochs),
        avg_train_loss,
        label=f"Avg Train Loss over {num_of_runs} runs",
        color="blue",
        linestyle="--",
        linewidth=4,
    )
    plt.plot(
        range(epochs),
        avg_test_loss,
        label=f"Avg Test Loss over {num_of_runs} runs",
        color="red",
        linestyle="--",
        linewidth=4,
    )

    plt.title(title_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()


def multiple_runs_with_every_run(metrics_list, title_accuracy, title_loss):

    plt.figure(figsize=(10, 6))

    # Prepare lists of train and test accuracies for uncertainty bands
    train_acc_list = [metrics["train_acc"] for metrics in metrics_list]
    test_acc_list = [metrics["test_acc"] for metrics in metrics_list]
    epochs = len(train_acc_list[0])
    num_of_runs = len(train_acc_list)

    for i, metrics in enumerate(metrics_list):
        plt.plot(range(epochs), metrics["train_acc"], color="blue", alpha=0.2)
        plt.plot(range(epochs), metrics["test_acc"], color="red", alpha=0.2)

    # Calculate and plot average train accuracy
    avg_train_loss = np.mean([metrics["train_acc"] for metrics in metrics_list], axis=0)
    plt.plot(
        range(epochs),
        avg_train_loss,
        label=f"Avg Train Accuracy over {num_of_runs} runs",
        color="blue",
        linestyle="--",
        linewidth=4,
    )

    # Calculate and plot average test accuracy
    avg_test_loss = np.mean([metrics["test_acc"] for metrics in metrics_list], axis=0)
    plt.plot(
        range(epochs),
        avg_test_loss,
        label=f"Avg Test Accuracy over {num_of_runs} runs",
        color="red",
        linestyle="--",
        linewidth=4,
    )

    # Add title, labels, grid, legend
    plt.title(title_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()

    plt.figure(figsize=(10, 6))

    train_loss_list = [metrics["train_loss"] for metrics in metrics_list]
    test_loss_list = [metrics["test_loss"] for metrics in metrics_list]

    for i, metrics in enumerate(metrics_list):
        plt.plot(range(epochs), metrics["train_loss"], color="blue", alpha=0.2)
        plt.plot(range(epochs), metrics["test_loss"], color="red", alpha=0.2)

    avg_train_loss = np.mean(
        [metrics["train_loss"] for metrics in metrics_list], axis=0
    )
    plt.plot(
        range(epochs),
        avg_train_loss,
        label=f"Avg Train Loss over {num_of_runs} runs",
        color="blue",
        linestyle="--",
        linewidth=4,
    )

    avg_test_loss = np.mean([metrics["test_loss"] for metrics in metrics_list], axis=0)
    plt.plot(
        range(epochs),
        avg_test_loss,
        label=f"Avg Test Loss over {num_of_runs} runs",
        color="red",
        linestyle="--",
        linewidth=4,
    )

    # Add title, labels, grid, legend
    plt.title(title_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()
