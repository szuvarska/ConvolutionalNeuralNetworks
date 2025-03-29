import torch
from torch import nn


class EnhancedModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # Block 1
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 2),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 2),
            nn.MaxPool2d(2),
        )

        # Block 3
        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 4),
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units * 4),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_units * 4, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
