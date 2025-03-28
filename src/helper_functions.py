import torch
from timeit import default_timer as timer


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device,
    silent=False,
):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y, y_pred=y_pred.argmax(dim=1)
        )  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    if not silent:
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return float(train_loss), train_acc


def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device,
    silent=False,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),  # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        if not silent:
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    return float(test_loss), test_acc


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None, silent=False):
    total_time = end - start
    if not silent:
        print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def run_model(
    cinic_train, cinic_test, model, loss_fn, optimizer, device, epochs=3, silent=True
):
    time_start = timer()
    metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Train and test model
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            data_loader=cinic_train,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
            silent=silent,
        )
        test_loss, test_acc = test_step(
            data_loader=cinic_test,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
            silent=silent,
        )

        # Append the metrics to the respective lists
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)

    time_end = timer()
    total_time = print_train_time(
        start=time_start, end=time_end, device=device, silent=silent
    )

    return metrics, total_time
