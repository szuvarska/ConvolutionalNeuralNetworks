import sys
from Dataset import SpeechCommandsDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchaudio.transforms as T
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()


class SpeechCommandTransformer(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 64, embed_dim: int = 64, num_layers: int = 4, num_heads: int = 4,
                 device: torch.device = None):
        super().__init__()

        self.device = device if device else torch.device("cpu")

        self.feature_extractor = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )

        self.db = torchaudio.transforms.AmplitudeToDB()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.projection = nn.Linear(n_mels, embed_dim)  # Project mel bins to embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable [CLS] token

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
        features = features.reshape(batch_size, freq * time, channels)  # (batch_size, seq_len, channels)

        x = self.projection(features)  # (batch_size, seq_len, embed_dim)

        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + time, embed_dim)

        x = self.transformer_encoder(x)

        cls_output = x[:, 0]  # take output of CLS token

        logits = self.classifier(cls_output)

        return logits


def train_transformer(train_loader, test_loader, model, criterion=None, optimizer=None, scheduler=None,
                      num_epochs: int = 10,
                      device: torch.device = None, verbose: bool = True, patience: int = 3):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_state = None
    best_train_acc = 0.0
    best_test_acc = 0.0
    patience_counter = 0

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

            pbar.set_postfix(loss=running_loss / total, acc=100. * correct / total)

        scheduler.step()

        train_acc = correct / total
        train_loss = running_loss / total

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for waveforms, labels in test_loader:
                waveforms = waveforms.squeeze(1).to(device)
                labels = labels.to(device)

                outputs = model(waveforms)
                preds = outputs.argmax(dim=1)

                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total

        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {100. * train_acc:.2f}, Test Accuracy: {100. * test_acc:.2f}%")

        # Early stopping
        if test_acc > best_test_acc:
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
        print(f"Best Test Accuracy: {100. * best_test_acc:.2f}%, Best Train Accuracy: {100. * best_train_acc:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)


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


if __name__ == "__main__":
    train_dataset = SpeechCommandsDataset("../data/train", mode="modified")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)

    test_dataset = SpeechCommandsDataset("../data/test", mode="modified")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeechCommandTransformer(num_classes=len(train_dataset.class_to_idx), device=device).to(device)

    train_transformer(train_loader, test_loader, model=model, num_epochs=10, device=device)
