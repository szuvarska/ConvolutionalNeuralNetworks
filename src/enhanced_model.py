import torch 
from torch import nn


class EnhancedModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        # Block 1: First set of convolutional layers
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),  # Add batch normalization
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce spatial dimensions
        )
        
        # Block 2: Second set of convolutional layers
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),  # Increase filter size
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.MaxPool2d(2)
        )
        
        # Block 3: Third set of convolutional layers
        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1),  # Increase filter size again
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.Conv2d(hidden_units*4, hidden_units*4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(2)
        )
        
        # Global Average Pooling to reduce the number of parameters
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer (classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),  # Add dropout to prevent overfitting
            nn.Linear(in_features=hidden_units*4, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.global_pool(x)  # Global average pooling
        x = self.classifier(x)
        return x