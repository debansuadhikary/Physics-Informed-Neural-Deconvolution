import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LightweightUNetRGB(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ENCODER
        self.down1 = DoubleConv(3, 32)      # Upgraded from 16 to 32
        self.pool1 = nn.MaxPool2d(2)    
        
        self.down2 = DoubleConv(32, 64)     # Upgraded from 32 to 64
        self.pool2 = nn.MaxPool2d(2)
        
        # BOTLENECK
        self.bottleneck = DoubleConv(64, 128) # Upgraded from 64 to 128
        
        # DECODER
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = DoubleConv(128, 64)        # 64 (upconv) + 64 (skip) = 128
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up1 = DoubleConv(64, 32)         # 32 (upconv) + 32 (skip) = 64
        
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, x):
        x1 = self.down1(x)         
        x = self.pool1(x1)
        x2 = self.down2(x)         
        x = self.pool2(x2)
        x = self.bottleneck(x)
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1) 
        x = self.up2(x)
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1) 
        x = self.up1(x)
        return self.final_conv(x)

if __name__ == "__main__":
    # Create a dummy tensor matching dataloader output
    dummy_input = torch.randn(4, 3, 128, 128) 
    
    # Initialising
    model = LightweightUNetRGB()
    
    # Passing the dummy data through the model
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("If they match exactly, the AI architecture is sound")