import torch
import numpy as np
from pytorch_msssim import ssim
import math
from torch.utils.data import DataLoader

from dataset import AstroPatchDatasetRGB
from model import LightweightUNetRGB

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def run_benchmark():
    print("... PB-ND Quantitative Benchmarking ...")
    
    # 1. Loading the Model
    model = LightweightUNetRGB()
    model.load_state_dict(torch.load('pbnd_model.pth'))
    model.eval()
    
    # 2. Loading 100 Validation Patches
    dataset = AstroPatchDatasetRGB(image_path="image_00.jpg", patch_size=128, num_samples=100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    psnr_raw_list, psnr_ai_list = [], []
    ssim_raw_list, ssim_ai_list = [], []
    
    with torch.no_grad():
        for noisy_inputs, clean_targets in dataloader:
            predictions = model(noisy_inputs)
            
            # Clip tensors to valid image range
            predictions = torch.clamp(predictions, 0.0, 1.0)
            noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)
            
            # Calculate PSNR
            psnr_raw = calculate_psnr(noisy_inputs, clean_targets)
            psnr_ai = calculate_psnr(predictions, clean_targets)
            
            # Calculate SSIM
            ssim_raw = ssim(noisy_inputs, clean_targets, data_range=1.0, size_average=True).item()
            ssim_ai = ssim(predictions, clean_targets, data_range=1.0, size_average=True).item()
            
            psnr_raw_list.append(psnr_raw)
            psnr_ai_list.append(psnr_ai)
            ssim_raw_list.append(ssim_raw)
            ssim_ai_list.append(ssim_ai)
            
    avg_psnr_raw = np.mean(psnr_raw_list)
    avg_psnr_ai = np.mean(psnr_ai_list)
    avg_ssim_raw = np.mean(ssim_raw_list)
    avg_ssim_ai = np.mean(ssim_ai_list)
    
    # Exposure time required scales quadratically with SNR
    snr_gain = 10 ** ((avg_psnr_ai - avg_psnr_raw) / 20)
    exposure_multiplier = snr_gain ** 2
    
    print(f"Raw Data Average PSNR:  {avg_psnr_raw:.2f} dB")
    print(f"PB-ND Average PSNR:     {avg_psnr_ai:.2f} dB (Higher is better)")
    print(f"Raw Data Average SSIM:  {avg_ssim_raw:.4f}")
    print(f"PB-ND Average SSIM:     {avg_ssim_ai:.4f} (Closer to 1.0 is better)\n")
    print(f"EFFECTIVE EXPOSURE MULTIPLIER: {exposure_multiplier:.1f}x")
    print(f"This means a 45-second observation cleaned by PB-ND contains the ")
    print(f"signal clarity of a {45 * exposure_multiplier / 60:.1f}-minute continuous exposure.")

if __name__ == "__main__":
    run_benchmark()