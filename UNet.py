# Amit Barilant, Alon Finestein

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import label
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import os
import numpy as np
from scipy.ndimage import label, generic_filter
from scipy.ndimage import label, convolve
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        enc1, enc2, enc3, enc4, bottleneck = self.encoder(x)  # Adjusted to include bottleneck
        output = self.decoder(enc1, enc2, enc3, enc4, bottleneck)
        return output

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
        return enc1, enc2, enc3, enc4, bottleneck  # Return bottleneck for decoder


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNetDecoder, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # New upconv layer
        self.decoder5 = conv_block(512, 256)  # New decoder layer

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = conv_block(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = conv_block(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = conv_block(64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the channel dimension

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        # Decoder layers with the additional upsampling step
        dec5 = self.upconv5(bottleneck)
        # print("dec5 shape is: ",dec5.shape)
        dec5 = torch.cat((dec5, enc4), dim=1)
        # print("dec5 after cat with enc4 shape is: ",dec5.shape)
        dec5 = self.decoder5(dec5)
        # print("dec5 after decoder5 shape is: ",dec5.shape)

        dec4 = self.upconv4(dec5)
        # print("dec4 shape is: ",dec4.shape)
        dec4 = torch.cat((dec4, enc3), dim=1)
        # print("dec4 after cat with enc3 shape is: ",dec4.shape)
        dec4 = self.decoder4(dec4)
        # print("dec4 after decoder4 shape is: ",dec4.shape)

        dec3 = self.upconv3(dec4)
        # print("dec3 shape is: ",dec3.shape)
        dec3 = torch.cat((dec3, enc2), dim=1)
        # print("dec3 after cat with enc2 shape is: ",dec3.shape)
        dec3 = self.decoder3(dec3)
        # print("dec3 after decoder3 shape is: ",dec3.shape)

        dec2 = self.upconv2(dec3)
        # print("dec2 shape is: ",dec2.shape)
        dec2 = torch.cat((dec2, enc1), dim=1)
        # print("dec2 after cat with enc1 shape is: ",dec2.shape)
        dec2 = self.decoder2(dec2)
        # print("dec2 after decoder2 shape is: ",dec2.shape)

        # Resize dec2 to match enc1 size before concatenation
        dec1 = self.upconv1(dec2)
        # print("dec1 shape is: ",dec1.shape)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # print("dec1 after cat with enc1 shape is: ",dec1.shape)
        dec1 = self.decoder1(dec1)
        # print("dec1 after decoder1 shape is: ",dec1.shape)

        # Final output layer
        output = self.final_conv(dec1)
        # print("output shape is: ",output.shape)

        # Apply softmax along the channel dimension
        output = self.softmax(output)

        return output

def grayscale_to_instance_encoding(gray_image):
    # Label connected components in the binary mask
    instance_map, num_instances = label(gray_image)

    # Edge detection kernel
    edge_kernel = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

    # Convolve the labeled instance map with the edge kernel
    edges = convolve(instance_map.astype(np.int32), edge_kernel, mode='constant', cval=0)

    # Initialize the classified map:
    # - Set edges (detected by non-zero convolution result) to 2
    # - Set inside (where the instance map is positive and no edge was detected) to 1
    classified_map = np.zeros_like(instance_map)
    classified_map[(instance_map > 0) & (edges == 0)] = 1  # Inside pixels
    classified_map[edges != 0] = 2                          # Edge pixels

    return classified_map

class CellImageDatasetUNet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        # Load image and mask
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
            # Resize the mask to match the image size
            resize_transform = T.Resize((512, 512))
            mask = resize_transform(mask)

        # Convert mask to numpy array and apply binary threshold
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # Ensure binary

        # Convert the binary mask to instance-encoded mask
        instance_encoded_mask = grayscale_to_instance_encoding(mask)

        # Convert instance-encoded mask to a tensor
        instance_encoded_mask_tensor = torch.from_numpy(instance_encoded_mask).long()
        # print(instance_encoded_mask_tensor.shape)
        return image, instance_encoded_mask_tensor