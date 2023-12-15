import torch
import torch.nn as nn

import torch.nn.functional as F

import lightning.pytorch as pl



from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, \
ToPILImage, Resize, ColorJitter, RandomRotation, RandomAffine, RandomErasing

def activation_layer(activation: str="relu", alpha: float=0.1, inplace: bool=True):
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)

    Returns:
        torch.Tensor: activation layer
    """
    if activation == "relu":
        return nn.ReLU(inplace=inplace)

    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)


class ConvBlock(nn.Module):
    """ Convolutional block with batch normalization
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conv=True, stride=1, dropout=0.2, activation="leaky_relu"):
        super(ResidualBlock, self).__init__()
        self.convb1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act1 = activation_layer(activation)

        self.convb2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=dropout)

        self.shortcut = None
        if skip_conv:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act2 = activation_layer(activation)

    def forward(self, x):
        skip = x

        out = self.act1(self.convb1(x))
        out = self.convb2(out)

        if self.shortcut is not None:
            out += self.shortcut(skip)

        out = self.act2(out)
        out = self.dropout(out)

        return out

class CNNmodel(pl.LightningModule):
    def __init__(self, pad_val, num_chars: int, activation: str="leaky_relu", dropout: float=0.2):
        super(CNNmodel, self).__init__()

        self.pad_val = pad_val

        self.rb1 = ResidualBlock(3, 16, skip_conv=True, stride=1, activation=activation, dropout=dropout)
        self.rb2 = ResidualBlock(16, 16, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb3 = ResidualBlock(16, 16, skip_conv=False, stride=1, activation=activation, dropout=dropout)
        self.rb4 = ResidualBlock(16, 32, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb5 = ResidualBlock(32, 32, skip_conv=False, stride=1, activation=activation, dropout=dropout)
        self.rb6 = ResidualBlock(32, 64, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb7 = ResidualBlock(64, 64, skip_conv=True, stride=1, activation=activation, dropout=dropout)
        self.rb8 = ResidualBlock(64, 64, skip_conv=False, stride=1, activation=activation, dropout=dropout)
        self.rb9 = ResidualBlock(64, 64, skip_conv=False, stride=1, activation=activation, dropout=dropout)

        self.lstm = nn.LSTM(64, 128, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout)

        self.output = nn.Linear(256, num_chars + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images_float = images / 255.0
        #images_float = images_float.permute(0, 3, 1, 2)

        #print(f"Shape after initial processing: {images_float.shape}")

#------------------------------------------------------------------------------
        x = self.rb1(images_float)
        #print(f"After rb1 Shape: {x.shape}")
        x = self.rb2(x)
        #print(f"After rb2 Shape: {x.shape}")
        x = self.rb3(x)
        #print(f"After rb3 Shape: {x.shape}")
        x = self.rb4(x)
        #print(f"After rb4 Shape: {x.shape}")
        x = self.rb5(x)
        #print(f"After rb5 Shape: {x.shape}")
        x = self.rb6(x)
        #print(f"After rb6 Shape: {x.shape}")
        x = self.rb7(x)
        #print(f"After rb7 Shape: {x.shape}")
        x = self.rb8(x)
        #print(f"After rb8 Shape: {x.shape}")
        x = self.rb9(x)
        #print(f"After rb9 Shape: {x.shape}")

        x = x.reshape(x.size(0), -1, x.size(1))
        #print(f"After Reshape Shape: {x.shape}")

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        #print(f"After LSTM Shape: {x.shape}")

        x = self.output(x)
        x = F.log_softmax(x, 2)
        #print('X from soft_max:', x)
        #print(f"Final Output Shape: {x.shape}")
#------------------------------------------------------------------------------

        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch

        outputs = self(images)

        target_lengths = torch.sum(targets != self.pad_val, dim=1)

        targets_unpadded = targets[targets != self.pad_val].view(-1)

        outputs = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        outputs_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)

        #print('Input lenghts tr', len(outputs_lengths))
        loss = F.ctc_loss(outputs, targets_unpadded, outputs_lengths, target_lengths, blank = self.pad_val)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        outputs = self(images)

        target_lengths = torch.sum(targets != self.pad_val, dim=1)

        targets_unpadded = targets[targets != self.pad_val].view(-1)

        outputs = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        outputs_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)

        #print('Input lenghts val', len(outputs_lengths))
        loss = F.ctc_loss(outputs, targets_unpadded, outputs_lengths, target_lengths, blank = self.pad_val)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer