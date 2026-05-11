import matplotlib.pyplot as plt

def generate_benchmark_graph():

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Data
    labels = ['Raw Observation', 'PB-ND Reconstruction']
    psnr_vals = [19.98, 29.53]
    ssim_vals = [0.2333, 0.8223]
    
    colors = ['#7f8c8d', '#2980b9'] 

    # PSNR
    # Added edgecolors and slightly thinner bars for a cleaner look
    bars1 = ax1.bar(labels, psnr_vals, color=colors, width=0.4, edgecolor='black', linewidth=1)
    ax1.set_title('Peak Signal-to-Noise Ratio (PSNR)', fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('Decibels (dB)', fontsize=11)
    ax1.set_ylim(0, 35)
    ax1.grid(axis='y', linestyle='--', alpha=0.7) # Added subtle horizontal grid lines

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval} dB', 
                 ha='center', va='bottom', fontsize=11, color='black')

    # SSIM
    bars2 = ax2.bar(labels, ssim_vals, color=colors, width=0.4, edgecolor='black', linewidth=1)
    ax2.set_title('Structural Similarity Index (SSIM)', fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylabel('Index (0.0 to 1.0)', fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval}', 
                 ha='center', va='bottom', fontsize=11, color='black')

    # Formatting
    plt.suptitle('PB-ND Engine: Quantitative Validation', fontsize=14, fontweight='bold', y=1.02)

    fig.text(0.5, -0.02, 'Effective Exposure Multiplier: 9.0x', ha='center', fontsize=12, style='italic')
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('benchmark_graph_professional.png', bbox_inches='tight', dpi=300, facecolor='white')
    print("Graph saved as 'benchmark_graph_professional.png'")
    plt.show()

if __name__ == "__main__":
    generate_benchmark_graph()