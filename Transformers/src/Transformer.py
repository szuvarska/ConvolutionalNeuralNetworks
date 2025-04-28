import sys
from Dataset import SpeechCommandsDataset
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchaudio.transforms as T
from tqdm import tqdm
from collections import Counter
import matplotlib
import random
# matplotlib.use('inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()


class SpeechCommandTransformer(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 64, embed_dim: int = 64, num_layers: int = 4, num_heads: int = 4,
                 device: torch.device = None, stride: int | list = 1, dim_feedforward: int = 512,
                 pos_embedding: bool = False, cnn_type: int = 1):
        super().__init__()

        self.device = device if device else torch.device("cpu")

        self.feature_extractor = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=n_mels
        )

        self.db = torchaudio.transforms.AmplitudeToDB()

        if isinstance(stride, int):
            stride1 = stride
            stride2 = stride
        else:
            stride1 = stride[0]
            stride2 = stride[1]

        if cnn_type == 1:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=stride1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=stride2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        elif cnn_type == 2:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=stride1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=stride2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=stride2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        elif cnn_type == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=stride1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=stride2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        elif cnn_type == 4:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=stride1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=stride2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.projection = nn.Linear(n_mels, embed_dim)  # Project mel bins to embed_dim

        if pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )  # Learnable [CLS] token

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, waveforms: torch.Tensor):
        """
        waveforms: List[Tensor] or Tensor shape (batch_size, samples)
        """

        if isinstance(waveforms, list):
            waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)

        features = self.feature_extractor(waveforms)  # (batch_size, n_mels, time)
        features = self.db(features)  # (batch_size, n_mels, time)
        features = self.cnn(features.unsqueeze(1))  # (batch_size, 64, n_mels, time)

        # Flatten the frequency and time dims into a sequence
        batch_size, channels, freq, time = features.shape
        features = features.permute(0, 2, 3, 1)  # (batch_size, freq, time, channels)
        features = features.reshape(
            batch_size, freq * time, channels
        )  # (batch_size, seq_len, channels)

        x = self.projection(features)  # (batch_size, seq_len, embed_dim)

        if hasattr(self, "pos_embedding"):
            x += self.pos_embedding[:, : x.size(1), :]  # Add positional embedding

        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + time, embed_dim)

        x = self.transformer_encoder(x)

        cls_output = x[:, 0]  # take output of CLS token

        logits = self.classifier(cls_output)

        return logits


def train_transformer(
    train_loader,
    test_loader,
    model,
    criterion=None,
    optimizer=None,
    scheduler=None,
    scheduling=True,
    num_epochs: int = 10,
    device: torch.device = None,
    verbose: bool = True,
    patience: int = 3,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if scheduler is None and scheduling:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_state = None
    best_train_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    train_true_labels = []
    train_pred_labels = []
    test_true_labels = []
    test_pred_labels = []

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for waveforms, labels in pbar:
            waveforms = waveforms.squeeze(1).to(device)  # (batch_size, samples)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(waveforms)  # (batch_size, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)

            # Compute accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=running_loss / total, acc=100.0 * correct / total)

            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

        if scheduling:
            scheduler.step()

        train_acc = correct / total
        train_loss = running_loss / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():
            for waveforms, labels in test_loader:
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)

                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                test_loss += loss.item() * labels.size(0)

                test_true_labels.extend(labels.cpu().numpy())
                test_pred_labels.extend(preds.cpu().numpy())

        test_acc = test_correct / test_total if test_total > 0 else 0
        test_loss = test_loss / test_total if test_total > 0 else 0

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {100. * train_acc:.2f}, Test Accuracy: {100. * test_acc:.2f}%"
            )

        # Early stopping
        if abs(best_test_acc - test_acc) > 1e-5:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if verbose:
        print(
            f"Best Test Accuracy: {100. * best_test_acc:.2f}%, Best Train Accuracy: {100. * best_train_acc:.2f}%"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return (
        train_losses,
        train_accuracies,
        test_losses,
        test_accuracies,
        train_true_labels,
        train_pred_labels,
        test_true_labels,
        test_pred_labels,
    )


def test_transformer(test_loader, model, device: torch.device = None, criterion=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.squeeze(1).to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {100. * accuracy:.2f}%")
    return accuracy, avg_loss


def calculate_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    label_counts = Counter(labels)
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_samples / label_counts[i] for i in range(len(dataset.class_to_idx))],
        dtype=torch.float,
    )
    class_weights = class_weights / class_weights.sum()
    return class_weights


def plot_confusion_matrix(true_labels, pred_labels, dataset, normalize=False):
    cm = confusion_matrix(true_labels, pred_labels)
    if normalize:
        cm = np.round(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], 3)

    label_names = list(dataset.class_to_idx.keys())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(include_values=True, xticks_rotation="vertical", ax=ax, cmap="Blues",
              values_format=".3f" if normalize else None)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()
    return plt


def plot_accuracy_loss(train_accuracies, train_losses, test_accuracies, test_losses):
    epochs = range(1, len(train_losses) + 1)

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlim(1, len(train_losses))
    plt.xticks(epochs)
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlim(1, len(train_accuracies))
    plt.ylim(
        max(0, min(train_accuracies + test_accuracies)),
        max(train_accuracies + test_accuracies),
    )
    plt.xticks(epochs)
    plt.show()


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)  # For CPU
    torch.cuda.manual_seed_all(seed_value)  # For GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    train_dataset = SpeechCommandsDataset("../data/train", mode="modified")
    indices = torch.randperm(len(train_dataset))[:1000]
    sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(
        train_dataset, batch_size=16, num_workers=6, sampler=sampler
    )

    test_dataset = SpeechCommandsDataset("../data/test", mode="modified")
    limited_test_dataset = Subset(test_dataset, range(1000))
    test_loader = DataLoader(
        limited_test_dataset, batch_size=16, shuffle=False, num_workers=6
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechCommandTransformer(
        num_classes=len(train_dataset.class_to_idx), device=device, stride=2
    ).to(device)

    class_weights = calculate_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    (
        train_losses,
        train_accuracies,
        test_losses,
        test_accuracies,
        train_true_labels,
        train_pred_labels,
        test_true_labels,
        test_pred_labels,
    ) = train_transformer(
        train_loader,
        test_loader,
        model=model,
        num_epochs=10,
        device=device,
        criterion=criterion,
    )

    plot_confusion_matrix(
        train_true_labels, train_pred_labels, train_dataset, normalize=False
    )

    plot_accuracy_loss(train_accuracies, train_losses, test_accuracies, test_losses)
