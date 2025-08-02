
!pip install torch-fidelity torchmetrics --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import seaborn as sns


nz = 100       
ngf = 32       
ndf = 32       
nc = 1         
image_size = 64
batch_size = 64   
num_epochs = 20    
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


full_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

subset_size = 5000  # Reduced from 20000
subset_indices = list(range(subset_size))
dataset = torch.utils.data.Subset(full_dataset, subset_indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Generator & Discriminator Definitions
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorDropout(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorNoBN(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# GAN Training Class
class GAN:
    def __init__(self, netG, netD, device, latent_dim=100, lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0):
        self.device = device
        self.netG = netG.to(device)
        self.netD = netD.to(device)
        self.latent_dim = latent_dim
        self.criterion = nn.BCELoss()
        self.optG = optim.Adam(netG.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optD = optim.Adam(netD.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.fixed_noise = torch.randn(64, latent_dim, device=device)
        self.history = {"G_losses": [], "D_losses": [], "images": []}

    def train(self, dataloader, num_epochs=50, log_interval=50, verbose=True):
        if verbose:
            print("Starting Training Loop...")
        for epoch in range(1, num_epochs + 1):
            for i, (real_imgs, _) in enumerate(dataloader, 1):
                b_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                # Train Discriminator
                self.netD.zero_grad()
                label = torch.full((b_size,), 1.0, device=self.device)
                output = self.netD(real_imgs).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_imgs = self.netG(noise)
                label.fill_(0.0)
                output = self.netD(fake_imgs.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optD.step()

                # Train Generator
                self.netG.zero_grad()
                label.fill_(1.0)
                output = self.netD(fake_imgs).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optG.step()

                self.history["D_losses"].append(errD.item())
                self.history["G_losses"].append(errG.item())

                if verbose and i % log_interval == 0:
                    print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                          f"Loss_D: {errD:.4f} Loss_G: {errG:.4f} "
                          f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

            # Save sample grid
            with torch.no_grad():
                fake = self.netG(self.fixed_noise).cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                self.history["images"].append(grid)

        if verbose:
            print("Training Complete.")
        return self.history

# ================================
# PAIRWISE COMPARISON FUNCTIONS
# ================================

def smooth_curve(data, window_length=21, polyorder=3):
    """Apply Savitzky-Golay smoothing"""
    if len(data) < window_length:
        window_length = len(data) // 2 * 2 + 1
        if window_length < 3:
            return data
    return savgol_filter(data, window_length, polyorder)

def create_pairwise_comparison(baseline_history, variant_history, baseline_label, variant_label):
    """Create detailed pairwise comparison between baseline and one variant"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'🔍 Analysis: {baseline_label} vs {variant_label}', 
                 fontsize=18, fontweight='bold')
    
    # Extract and smooth data
    baseline_g = np.array(baseline_history['G_losses'])
    baseline_d = np.array(baseline_history['D_losses'])
    variant_g = np.array(variant_history['G_losses'])
    variant_d = np.array(variant_history['D_losses'])
    
    baseline_g_smooth = smooth_curve(baseline_g)
    variant_g_smooth = smooth_curve(variant_g)
    
    # Colors
    baseline_color = '#2E86C1'  # Blue
    variant_color = '#E74C3C'   # Red
    
    # 1. Generator Loss Comparison (Top Left)
    ax1.plot(baseline_g_smooth, color=baseline_color, linewidth=3, 
             label=f'{baseline_label}', alpha=0.8)
    ax1.plot(variant_g_smooth, color=variant_color, linewidth=3, 
             label=f'{variant_label}', alpha=0.8)
    ax1.set_title('📈 Generator Loss Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Generator Loss')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add improvement metrics
    final_baseline = baseline_g_smooth[-1]
    final_variant = variant_g_smooth[-1]
    improvement = ((final_baseline - final_variant) / final_baseline) * 100
    
    ax1.text(0.02, 0.98, 
             f'📊 Final Losses:\n{baseline_label}: {final_baseline:.3f}\n{variant_label}: {final_variant:.3f}\n💡 Improvement: {improvement:.1f}%', 
             transform=ax1.transAxes, va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Discriminator Loss Comparison (Top Right)
    baseline_d_smooth = smooth_curve(baseline_d)
    variant_d_smooth = smooth_curve(variant_d)
    
    ax2.plot(baseline_d_smooth, color=baseline_color, linewidth=3, 
             linestyle='--', label=f'{baseline_label}', alpha=0.8)
    ax2.plot(variant_d_smooth, color=variant_color, linewidth=3, 
             linestyle='--', label=f'{variant_label}', alpha=0.8)
    ax2.set_title('🛡️ Discriminator Loss Comparison', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Discriminator Loss')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    final_baseline_d = baseline_d_smooth[-1]
    final_variant_d = variant_d_smooth[-1]
    d_change = ((final_baseline_d - final_variant_d) / final_baseline_d) * 100
    
    ax2.text(0.02, 0.98, 
             f'📊 Final D Losses:\n{baseline_label}: {final_baseline_d:.3f}\n{variant_label}: {final_variant_d:.3f}\n📈 Change: {d_change:.1f}%', 
             transform=ax2.transAxes, va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. Loss Difference Analysis (Bottom Left)
    min_len = min(len(baseline_g), len(variant_g))
    diff = variant_g[:min_len] - baseline_g[:min_len]
    diff_smooth = smooth_curve(diff)
    
    ax3.plot(diff_smooth, color='purple', linewidth=2.5, label='Difference')
    ax3.fill_between(range(len(diff_smooth)), 0, diff_smooth, 
                     where=(diff_smooth <= 0), color='green', alpha=0.3, 
                     label=f'{variant_label} Better')
    ax3.fill_between(range(len(diff_smooth)), 0, diff_smooth, 
                     where=(diff_smooth > 0), color='red', alpha=0.3, 
                     label=f'{baseline_label} Better')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title(f'⚖️ Advantage Analysis\n({variant_label} - {baseline_label})', 
                  fontweight='bold', fontsize=14)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Loss Difference')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Calculate percentage of time variant is better
    better_percentage = (np.sum(diff_smooth <= 0) / len(diff_smooth)) * 100
    ax3.text(0.02, 0.02, f'✅ {variant_label} better\n{better_percentage:.1f}% of time', 
             transform=ax3.transAxes, va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 4. Statistical Analysis (Bottom Right)
    final_portion = int(0.8 * len(baseline_g))
    baseline_final = baseline_g[final_portion:]
    variant_final = variant_g[final_portion:]
    
    # Box plot
    box_data = [baseline_final, variant_final]
    bp = ax4.boxplot(box_data, labels=[baseline_label, variant_label], 
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(baseline_color)
    bp['boxes'][1].set_facecolor(variant_color)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    
    ax4.set_title('📊 Final Performance Distribution\n(Last 20% of Training)', 
                  fontweight='bold', fontsize=14)
    ax4.set_ylabel('Generator Loss')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(baseline_final, variant_final)
    significance = "Significant ✅" if p_value < 0.05 else "Not Significant ❌"
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(baseline_final)-1)*np.var(baseline_final) + 
                         (len(variant_final)-1)*np.var(variant_final)) / 
                        (len(baseline_final)+len(variant_final)-2))
    cohens_d = (np.mean(variant_final) - np.mean(baseline_final)) / pooled_std
    
    ax4.text(0.5, 0.95, 
             f'🔬 Statistical Test:\np-value: {p_value:.4f}\n{significance}\n📏 Effect Size: {abs(cohens_d):.2f}', 
             transform=ax4.transAxes, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📋 ANALYSIS SUMMARY: {baseline_label} vs {variant_label}")
    print(f"{'='*60}")
    print(f"🎯 Generator Performance:")
    print(f"   • {baseline_label} final loss: {final_baseline:.4f}")
    print(f"   • {variant_label} final loss: {final_variant:.4f}")
    print(f"   • Improvement: {improvement:.2f}%")
    print(f"\n🔬 Statistical Validation:")
    print(f"   • p-value: {p_value:.4f}")
    print(f"   • Significance: {significance}")
    print(f"   • Effect size (Cohen's d): {abs(cohens_d):.3f}")
    print(f"\n💡 Recommendation:")
    if improvement > 5 and p_value < 0.05:
        print(f"   ✅ STRONG: Use {variant_label} - significant improvement!")
    elif improvement > 0 and p_value < 0.05:
        print(f"   ✅ MODERATE: {variant_label} is better, but modest gains")
    elif improvement < -5:
        print(f"   ❌ AVOID: {variant_label} performs worse than {baseline_label}")
    else:
        print(f"   ➖ NEUTRAL: No clear advantage for either method")
    print(f"{'='*60}\n")
    
    return {
        'improvement_percent': improvement,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'better_percentage': better_percentage
    }

def analyze_all_variants(histories, labels, baseline_idx=0):
    """Run pairwise analysis for all variants against baseline"""
    baseline_history = histories[baseline_idx]
    baseline_label = labels[baseline_idx]
    
    results = {}
    
    print(f"🔬 COMPREHENSIVE ANALYSIS AGAINST {baseline_label.upper()}")
    print(f"{'='*80}")
    
    for i, (variant_history, variant_label) in enumerate(zip(histories, labels)):
        if i == baseline_idx:
            continue
            
        print(f"\n🔍 Analyzing: {baseline_label} vs {variant_label}")
        print("-" * 50)
        
        result = create_pairwise_comparison(
            baseline_history, variant_history,
            baseline_label, variant_label
        )
        results[variant_label] = result
    
    return results

# ================================
# TRAINING EXPERIMENTS
# ================================

print("🚀 Starting GAN Experiments with Reduced Parameters...")

# Baseline
print("\n1️⃣ Training Baseline...")
netG_base = Generator(nz, ngf, nc)
netD_base = Discriminator(nc, ndf)
gan_base = GAN(netG_base, netD_base, device)
history_base = gan_base.train(dataloader, num_epochs, verbose=False)

# L2 Regularization
print("\n2️⃣ Training L2 Regularization...")
netG_l2 = Generator(nz, ngf, nc)
netD_l2 = Discriminator(nc, ndf)
gan_l2 = GAN(netG_l2, netD_l2, device, weight_decay=0.01)
history_l2 = gan_l2.train(dataloader, num_epochs, verbose=False)

# Dropout
print("\n3️⃣ Training Dropout...")
netG_dropout = Generator(nz, ngf, nc)
netD_dropout = DiscriminatorDropout(nc, ndf)
gan_dropout = GAN(netG_dropout, netD_dropout, device)
history_dropout = gan_dropout.train(dataloader, num_epochs, verbose=False)

# No BatchNorm
print("\n4️⃣ Training No BatchNorm...")
netG_bn = Generator(nz, ngf, nc)
netD_bn = DiscriminatorNoBN(nc, ndf)
gan_bn = GAN(netG_bn, netD_bn, device)
history_bn = gan_bn.train(dataloader, num_epochs, verbose=False)

# ================================
# PAIRWISE ANALYSIS
# ================================

histories = [history_base, history_l2, history_dropout, history_bn]
labels = ["Baseline", "L2", "Dropout", "NoBatchNorm"]

print("\n" + "="*80)
print("🎯 RUNNING COMPREHENSIVE PAIRWISE ANALYSIS")
print("="*80)

results = analyze_all_variants(histories, labels, baseline_idx=0)

# ================================
# FID EVALUATION
# ================================

print("\n🎨 Computing FID Scores...")
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
import torch.nn.functional as F

# Prepare real images for FID
denorm = T.Normalize((-1,), (2,))

def prep_images(imgs):
    imgs = denorm(imgs).clamp(0, 1)
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    return imgs.repeat(1, 3, 1, 1)

# Collect real images
real_list = []
count = 0
for imgs, _ in dataloader:
    real_list.append(imgs)
    count += imgs.size(0)
    if count >= 64:  # Reduced for speed
        break
real_imgs = torch.cat(real_list, dim=0)[:64].to(device)
real_imgs = prep_images(real_imgs)

def compute_fid(models_dict, num_samples=64, step=16):
    fid_scores = {}
    for label, netG in models_dict.items():
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        fid.update(real_imgs, real=True)
        
        netG.eval()
        seen = 0
        with torch.no_grad():
            while seen < num_samples:
                b = min(step, num_samples - seen)
                noise = torch.randn(b, nz, device=device)
                fake = netG(noise)
                fake = prep_images(fake)
                fid.update(fake, real=False)
                seen += b
        
        score = fid.compute().item()
        fid_scores[label] = round(score, 2)
    
    return fid_scores

models_to_eval = {
    "Baseline": netG_base,
    "L2": netG_l2,
    "Dropout": netG_dropout,
    "NoBatchNorm": netG_bn
}

fid_results = compute_fid(models_to_eval)

print("\n📊 FID SCORES COMPARISON:")
print("="*40)
baseline_fid = fid_results["Baseline"]
for method, fid_score in fid_results.items():
    if method == "Baseline":
        print(f"🔵 {method}: {fid_score}")
    else:
        improvement = ((baseline_fid - fid_score) / baseline_fid) * 100
        symbol = "✅" if improvement > 0 else "❌"
        print(f"{symbol} {method}: {fid_score} ({improvement:+.1f}%)")

print(f"\n🏆 BEST FID SCORE: {min(fid_results, key=fid_results.get)} ({min(fid_results.values())})")

# ================================
# FINAL SUMMARY
# ================================

print("\n" + "="*80)
print("🏁 FINAL RECOMMENDATIONS")
print("="*80)

best_loss_method = min(results.keys(), key=lambda x: results[x]['improvement_percent'])
best_fid_method = min(fid_results.keys(), key=fid_results.get)

print(f"🎯 Best Loss Performance: {best_loss_method}")
print(f"🎨 Best FID Score: {best_fid_method}")

if best_loss_method == best_fid_method.replace('Baseline', ''):
    print(f"🎉 CLEAR WINNER: {best_fid_method} excels in both metrics!")
else:
    print(f"🤔 Mixed results - consider your priority: loss vs. image quality")

print("="*80)
