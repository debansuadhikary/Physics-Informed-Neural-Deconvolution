import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.special import j1
from PIL import Image

def generate_airy_psf(size, radius):
    
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2) + 1e-10  # Avoid division by zero
    
    arg = (np.pi * r) / radius
    psf = (2 * j1(arg) / arg)**2
    return psf / psf.sum()

def apply_physics_model(clean_img, psf, snr_linear=50):
    
    # 1. Forward Convolution (The Blur)
    blurred = fftconvolve(clean_img, psf, mode='same')
    
    # 2. Add Poisson Noise (Shot noise from photons)
    scaled_img = blurred * snr_linear
    
    # Safety catch: FFT math can sometimes result in tiny negative numbers (e.g., -1e-15).
    # Poisson distributions cannot take negative inputs, so we clip at 0.
    scaled_img = np.clip(scaled_img, 0, None)
    
    noisy = np.random.poisson(scaled_img).astype(float) / snr_linear
    
    return noisy

# Execution:
# 1. Loading the images
image_path = "image_00.jpg"
img = Image.open(image_path)

# 2. Convert to Grayscale and Normalise
img_gray = img.convert('L')
img_array = np.array(img_gray, dtype=float) / 255.0  # Normalise pixel values to 0.0 - 1.0

# 3. Croping a 512x512 patch from the center to save memory/CPU
h, w = img_array.shape
crop_size = 512
start_y = h // 2 - crop_size // 2
start_x = w // 2 - crop_size // 2
clean_image = img_array[start_y:start_y+crop_size, start_x:start_x+crop_size]

# 4. Generating the PSF
psf = generate_airy_psf(size=64, radius=3.0)

# 5. Generating the Observation
# We set SNR to 20 to simulate a short exposure / noisy capture
observed = apply_physics_model(clean_image, psf, snr_linear=20)

# 6. Visualisation
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Ground Truth (JWST Center Crop)")
plt.imshow(clean_image, cmap='magma', origin='lower')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Telescope PSF (Airy Disk)")
plt.imshow(psf, cmap='viridis', origin='lower')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Observation (Blurred + Noisy)")
plt.imshow(observed, cmap='magma', origin='lower')
plt.axis('off')

plt.tight_layout()
plt.show()