import torch.nn as nn
import torch.nn.functional as F
import torch


# This file contains various neural network architectures for training models in PyTorch.
# For detailed explanation on the CNNResNetAttention architecture, refer to the Section 2.3 "Deep Model Architecture"
# and Section 2.4 "Implementation Details" in the paper.


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        attention_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, attention_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, attention_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()

        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)

        return out + x


class CNNResNetAttention(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int] = (1, 2000, 50), num_classes: int = 10) -> None:
        super().__init__()

        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.pool = nn.MaxPool2d(2, 2)

        self.resblock1 = ResidualBlock(1, 2, stride=2)
        self.resblock2 = ResidualBlock(2, 3, stride=2)
        self.attention = AttentionBlock(3)

        # Calculate the correct input size for the first fully connected layer
        self.fc_input_size = self._get_fc_input_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)


    def _get_fc_input_size(self, input_size: tuple[int, int, int]) -> int:
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            x = self.pool(F.relu(self.bn(self.conv(x))))
            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.attention(x)
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.attention(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Models for ablation studies introduced in the Section 3.2 Experiments with Real Data of the paper.
    

class CNNResNet_NoAttention(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int] = (1, 2000, 50), num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.pool = nn.MaxPool2d(2, 2)

        self.resblock1 = ResidualBlock(1, 2, stride=2)
        self.resblock2 = ResidualBlock(2, 3, stride=2)

        self.fc_input_size = self._get_fc_input_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _get_fc_input_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            x = self.pool(F.relu(self.bn(self.conv(x))))
            x = self.resblock1(x)
            x = self.resblock2(x)
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class ResNetAttention_NoCNN(nn.Module):
    def __init__(self, input_shape=(1, 2000, 50), num_classes=10):
        super().__init__()
        self.resblock1 = ResidualBlock(1, 2, stride=2)
        self.resblock2 = ResidualBlock(2, 3, stride=2)
        self.attention = AttentionBlock(3)

        self.fc_input_size = self._get_fc_input_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _get_fc_input_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.attention(x)
        return x.numel()

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class CNNAttention_NoResidual(nn.Module):
    def __init__(self, input_shape=(1, 2000, 50), num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.pool = nn.MaxPool2d(2, 2)
        self.down = nn.MaxPool2d(4, 4)

        self.attention = AttentionBlock(1)

        self.fc_input_size = self._get_fc_input_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _get_fc_input_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            x = self.pool(F.relu(self.bn(self.conv(x))))
            x = self.down(x)
            x = self.attention(x)
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = self.down(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN_Only(nn.Module):
    def __init__(self, input_shape=(1, 2000, 50), num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc_input_size = self._get_fc_input_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _get_fc_input_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            x = self.pool(F.relu(self.bn(self.conv(x))))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(
    input_shape: tuple[int, int, int] = (1, 2000, 50),
    model_type: str = "res_net",
    num_classes: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 7e-3
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    if model_type == "baseline_cnn":
        model = CNN_Only(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "res_net":
        model = CNNResNetAttention(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "no_att":
        model = CNNResNet_NoAttention(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "no_res":
        model = CNNAttention_NoResidual(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "no_cnn":
        model = ResNetAttention_NoCNN(input_shape=input_shape, num_classes=num_classes)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, loss_fn, optimizer
