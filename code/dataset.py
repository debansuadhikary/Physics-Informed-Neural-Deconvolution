import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import j1

def generate_airy_psf(size, base_radius, wavelength_nm, ref_wavelength_nm=550.0):
    # Creates an Airy Disk where the blur size scales dynamically with the color of light.
    # Scale the radius based on wavelength (Red = wider, Blue = tighter)
    scaling_factor = wavelength_nm / ref_wavelength_nm
    effective_radius = base_radius * scaling_factor
    
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2) + 1e-10
    
    arg = (np.pi * r) / effective_radius
    psf = (2 * j1(arg) / arg)**2
    return psf / psf.sum()

def apply_physics_model_rgb(clean_img_rgb, psfs, snr_linear=20):
    """
    Applies Convolution and Noise to each color channel independently.
    clean_img_rgb: shape (H, W, 3)
    psfs: tuple of (psf_R, psf_G, psf_B)
    """
    noisy_rgb = np.zeros_like(clean_img_rgb)
    
    # Loop through Red (0), Green (1), and Blue (2) channels
    for c in range(3):
        # 1. Forward Convolution
        blurred = fftconvolve(clean_img_rgb[:, :, c], psfs[c], mode='same')
        
        # 2. Add Poisson Noise
        scaled_img = np.clip(blurred * snr_linear, 0, None)
        noisy = np.random.poisson(scaled_img).astype(float) / snr_linear
        
        noisy_rgb[:, :, c] = noisy
        
    return noisy_rgb

class AstroPatchDatasetRGB(Dataset):
    def __init__(self, image_path, patch_size=128, num_samples=1000):
        img = Image.open(image_path).convert('RGB')
        self.full_image = np.array(img, dtype=float) / 255.0
        self.patch_size = patch_size
        self.num_samples = num_samples
        
        psf_R = generate_airy_psf(size=31, base_radius=3.0, wavelength_nm=650.0) # Red
        psf_G = generate_airy_psf(size=31, base_radius=3.0, wavelength_nm=550.0) # Green
        psf_B = generate_airy_psf(size=31, base_radius=3.0, wavelength_nm=450.0) # Blue
        self.psfs = (psf_R, psf_G, psf_B)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h, w, _ = self.full_image.shape
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        
        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)
        
        # Clean patch is now (128, 128, 3)
        clean_patch = self.full_image[y:y+self.patch_size, x:x+self.patch_size, :]
        noisy_patch = apply_physics_model_rgb(clean_patch, self.psfs, snr_linear=20)
        
        # CHANGE 3: PyTorch expects channels first [Channels, H, W], so we transpose the arrays
        clean_tensor = torch.tensor(clean_patch.transpose(2, 0, 1), dtype=torch.float32)
        noisy_tensor = torch.tensor(noisy_patch.transpose(2, 0, 1), dtype=torch.float32)
        
        return noisy_tensor, clean_tensor

if __name__ == "__main__":
    IMAGE_FILE_NAME = "image_00.jpg" 
    print(f"Loading hyperspectral data from: {IMAGE_FILE_NAME}...")
    
    dataset = AstroPatchDatasetRGB(image_path=IMAGE_FILE_NAME, patch_size=128, num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    noisy_batch, clean_batch = next(iter(dataloader))
    
    # Expected output: [4, 3, 128, 128]
    print(f"Success! RGB Batch Shape: {noisy_batch.shape}")