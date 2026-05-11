import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from dataset import AstroPatchDatasetRGB, generate_airy_psf
from model import LightweightUNetRGB

from astropy.io import fits

def load_and_normalize_fits(fits_path):
    """Loads a raw JWST FITS file with aggressive NaN-cleansing and Asinh Stretch."""
    # 1. Read the array
    raw_data = np.squeeze(fits.getdata(fits_path, memmap=True))
    
    # 2. Aggressive NaN Wiping
    valid_data = raw_data[np.isfinite(raw_data)]
    median_val = np.median(valid_data)
    
    # Replace the NaNs and Infinities with the smooth background median
    clean_data = np.nan_to_num(raw_data, nan=median_val, posinf=median_val, neginf=median_val)
    
    # 3. Asinh Stretch
    shifted_data = clean_data - median_val
    
    # The '0.1' is a scaling factor to soften the stretch
    stretched_data = np.arcsinh(shifted_data * 0.1)
    
    # 4. Percentile Normalisation
    vmin = np.percentile(stretched_data, 1.0)
    vmax = np.percentile(stretched_data, 99.5)
    
    normalized_data = (stretched_data - vmin) / (vmax - vmin)
    
    # 5. Strict Clipping to 0.0 - 1.0 for the AI
    normalized_data = np.clip(normalized_data, 0.0, 1.0)
    
    return normalized_data.astype(np.float32)

def build_rgb_fits_cube(fits_red, fits_green, fits_blue):
    """Maps the 3 bands to Red, Green, and Blue channels and forces matching shapes."""
    print("Extracting Photons and stacking Hyperspectral Cube...")
    band_R = load_and_normalize_fits(fits_red) 
    band_G = load_and_normalize_fits(fits_green) 
    band_B = load_and_normalize_fits(fits_blue) 
    
    # 1. Finding the smallest height and width across all three images
    min_h = min(band_R.shape[0], band_G.shape[0], band_B.shape[0])
    min_w = min(band_R.shape[1], band_G.shape[1], band_B.shape[1])
    
    print(f"Aligning dimensions... Cropping all bands to {min_h} x {min_w}")
    
    # 2. Helper function to center-crop an image to the minimum target size
    def center_crop_2d(img, target_h, target_w):
        h, w = img.shape
        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        return img[start_y:start_y+target_h, start_x:start_x+target_w]

    # 3. Apply the crop so they are mathematically identical in shape
    band_R = center_crop_2d(band_R, min_h, min_w)
    band_G = center_crop_2d(band_G, min_h, min_w)
    band_B = center_crop_2d(band_B, min_h, min_w)

    
    # Stack into [Height, Width, 3]
    rgb_cube = np.stack([band_R, band_G, band_B], axis=-1)
    
    # Transpose for PyTorch to [3, Height, Width]
    return rgb_cube.transpose(2, 0, 1)

class PhysicsInformedLossRGB(nn.Module):
    def __init__(self, psf_R, psf_G, psf_B, lambda_phys=0.5):
        super().__init__()
        psf_stack = np.stack([psf_R, psf_G, psf_B], axis=0)[:, np.newaxis, :, :]
        self.psf_tensor = torch.tensor(psf_stack, dtype=torch.float32)
        self.lambda_phys = lambda_phys
        
        self.l1 = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss()

    def forward(self, pred_clean, target_clean, input_noisy):
        # 1. THE SATURATION SHIELD (Luminance Masking)
        saturation_threshold = 0.95
        mask = (target_clean < saturation_threshold).float()
        
        # 2. Masked L1 Data Loss
        raw_l1_loss = self.l1(pred_clean, target_clean)
        loss_data = (raw_l1_loss * mask).mean() # Apply mask and then take the average
        
        # 3. Physics Convolution Loss 
        pad = self.psf_tensor.shape[-1] // 2
        re_blurred = F.conv2d(pred_clean, self.psf_tensor, padding=pad, groups=3)
        loss_phys = self.mse(re_blurred, input_noisy)
        
        return loss_data + (self.lambda_phys * loss_phys)


def process_full_image_in_tiles(model, full_tensor, tile_size=1024):
    """
    Chops a massive tensor into tiles, runs AI deconvolution on each, 
    and stitches them back together seamlessly.
    """
    _, c, h, w = full_tensor.shape
    
    # Create an empty canvas of the exact same size to hold the stitched image
    output_tensor = torch.zeros_like(full_tensor)
    
    print(f"Starting Tile Processing for massive {w}x{h} image...")
    
    # Slide a window across the image, row by row, column by column
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Calculate the edges of the current tile
            end_y = min(y + tile_size, h)
            end_x = min(x + tile_size, w)
            
            # Extract the small tile from the massive image
            tile = full_tensor[:, :, y:end_y, x:end_x]
            
            # Neural Networks need dimensions to be multiples of 16 (due to pooling layers)
            # If the edge pieces are odd sizes, we skip them to avoid crashes
            if tile.shape[2] % 16 != 0 or tile.shape[3] % 16 != 0:
                output_tensor[:, :, y:end_y, x:end_x] = tile # Leave edges raw
                continue

            with torch.no_grad():
                clean_tile = model(tile)
            
            # Paste the cleaned tile onto the blank canvas
            output_tensor[:, :, y:end_y, x:end_x] = clean_tile
            print(f"   -> Stitched Tile at Y:{y}-{end_y}, X:{x}-{end_x}")
            
    print("Mosaicking Complete!")
    return output_tensor


# EXECUTION BLOCK
if __name__ == "__main__":
    IMAGE_FILE_NAME = "image_00.jpg"
    print(f"Initializing RGB Data and Model using {IMAGE_FILE_NAME}...")
    
    # 1. Setup Data
    dataset = AstroPatchDatasetRGB(image_path=IMAGE_FILE_NAME, patch_size=128, num_samples=200)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Setup RGB Model & Optimiser
    model = LightweightUNetRGB()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 3. Setup Hyperspectral Physics Loss
    psf_R = generate_airy_psf(size=31, base_radius=3.0, wavelength_nm=650.0) 
    psf_G = generate_airy_psf(size=31, base_radius=3.0, wavelength_nm=550.0) 
    psf_B = generate_airy_psf(size=31, base_radius=3.0, wavelength_nm=450.0) 
    criterion = PhysicsInformedLossRGB(psf_R, psf_G, psf_B, lambda_phys=0.5)
    
    # 4. THE TRAINING LOOP
    epochs = 20
    print("\nStarting Hyperspectral Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (noisy_inputs, clean_targets) in enumerate(dataloader):
            optimizer.zero_grad()
            predictions = model(noisy_inputs)
            loss = criterion(predictions, clean_targets, noisy_inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.6f}")
                
        print(f"... Epoch {epoch+1} Complete | Average Loss: {running_loss/len(dataloader):.6f} ...")

    print("\nSaving trained model weights to disk...")
    torch.save(model.state_dict(), 'pbnd_model.pth')

    # 5. VISUALIsATION (Full Color)
    print("\nGenerating RGB AI Reconstructions... ")
    model.eval() 
    with torch.no_grad():
        test_noisy, test_clean = next(iter(dataloader))
        prediction = model(test_noisy)
        
        img_noisy = np.clip(test_noisy[0].numpy().transpose(1, 2, 0), 0, 1)
        img_pred = np.clip(prediction[0].numpy().transpose(1, 2, 0), 0, 1)
        img_clean = np.clip(test_clean[0].numpy().transpose(1, 2, 0), 0, 1)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("1. Noisy Observation (RGB)")
        plt.imshow(img_noisy)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("2. AI Reconstruction (RGB)")
        plt.imshow(img_pred)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("3. Ground Truth (RGB)")
        plt.imshow(img_clean)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    # 6. MOSAIKING
    print("\nGenerating High-Definition AI Mosaic... ")
    
    FILE_RED = "jwst_red.fits"     
    FILE_GREEN = "jwst_green.fits" 
    FILE_BLUE = "jwst_blue.fits"   
    
    try:
        # 1. Building the massive RGB Tensor
        real_rgb_data = build_rgb_fits_cube(FILE_RED, FILE_GREEN, FILE_BLUE)
        real_tensor = torch.tensor(real_rgb_data).unsqueeze(0)
        
        # 2. Runing the Sliding Window Stitcher
        model.eval()
        ai_reconstruction = process_full_image_in_tiles(model, real_tensor, tile_size=1024)
            
        # 3. Format for saving [H, W, C]
        # We multiply by 255 to save it as a standard High-Def image file
        final_image_array = np.clip(ai_reconstruction[0].numpy().transpose(1, 2, 0), 0, 1)
        final_image_8bit = (final_image_array * 255).astype(np.uint8)
        
        # 4. Saving the massive image
        import matplotlib.image as mpimg
        mpimg.imsave("PBND_HighDef_Orion_Mosaic.png", final_image_8bit)
        
        print("\nSUCCESS! Massive High-Definition Image saved as 'PBND_HighDef_Orion_Mosaic.png'.")
        print("Open it in your computer's image viewer and zoom in!")
        
    except FileNotFoundError:
        print("Waiting for FITS files! Make sure the names match exactly.")