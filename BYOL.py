# Amit Barilant, Alon Finestein
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(UNetEncoder, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 32)  # Reduced filters
        self.encoder2 = conv_block(32, 64)           # Reduced filters
        self.encoder3 = conv_block(64, 128)          # Reduced filters
        self.encoder4 = conv_block(128, 256)         # Reduced filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional pooling layer to further downsample the output
        self.bottleneck = conv_block(256, 512)
        self.additional_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the output dimension of the encoder
        self.output_dim = 512 * 32 * 32 # Assuming input size is divisible by 16 due to 4 pooling layers

    def forward(self, x):
        enc1 = self.encoder1(x)
        # print("enc1 shape is: ",enc1.shape)
        enc2 = self.encoder2(self.pool(enc1))
        # print("enc2 shape is: ",enc2.shape)
        enc3 = self.encoder3(self.pool(enc2))
        # print("enc3 shape is: ",enc3.shape)
        enc4 = self.encoder4(self.pool(enc3))
        # print("enc4 shape is: ",enc4.shape)
        bottleneck = self.bottleneck(self.additional_pool(enc4))  # Apply additional pooling to reduce size
        # print("bottleneck shape is: ",bottleneck.shape)
        return bottleneck.view(bottleneck.size(0), -1)




class BYOL(nn.Module):
    def __init__(self, base_encoder, projector_dim=256, predictor_dim=4096, moving_average_decay=0.99):
        super(BYOL, self).__init__()
        # Online network
        self.online_encoder = base_encoder()
        self.online_projector = self._build_projector(self.online_encoder.output_dim, projector_dim)
        self.online_predictor = self._build_predictor(projector_dim, predictor_dim)

        # Target network
        self.target_encoder = base_encoder()
        self.target_projector = self._build_projector(self.target_encoder.output_dim, projector_dim)
        self.moving_average_decay = moving_average_decay

        # Initialize target network with online network weights
        self._initialize_target_network()

    def _build_projector(self, input_dim, projector_dim):
        return nn.Sequential(
            nn.Linear(input_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )

    def _build_predictor(self, projector_dim, predictor_dim):
        return nn.Sequential(
            nn.Linear(projector_dim, predictor_dim),
            nn.BatchNorm1d(predictor_dim),
            nn.ReLU(),
            nn.Linear(predictor_dim, projector_dim)
        )

    def _initialize_target_network(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def update_target_network(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.moving_average_decay * param_t.data + (1 - self.moving_average_decay) * param_o.data

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data = self.moving_average_decay * param_t.data + (1 - self.moving_average_decay) * param_o.data

    def forward(self, x1, x2):
        # Online network forward pass
        z1_online = self.online_projector(self.online_encoder(x1))
        z2_online = self.online_projector(self.online_encoder(x2))
        p1_online = self.online_predictor(z1_online)
        p2_online = self.online_predictor(z2_online)

        # Target network forward pass
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))

        # Loss calculation
        loss = 0.5 * (F.mse_loss(p1_online, z1_target) + F.mse_loss(p2_online, z2_target))
        return loss

    def get_representation(self, x):
        # Extract representations from the online encoder
        return self.online_encoder(x)


class CustomImageDatasetBYOL(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale if needed
        if self.transform:
            image = self.transform(image)
        return image


