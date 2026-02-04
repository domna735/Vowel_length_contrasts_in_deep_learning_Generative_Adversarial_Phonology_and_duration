"""
Explore the ciwGAN latent space to understand duration encoding.

Questions to answer:
1. Can we interpolate between short/long in latent space?
2. Does VOT change smoothly during interpolation?
3. Are duration classes linearly separable in latent space?

Usage:
    python tools/explore_latent_space.py --checkpoint runs/train/ciwgan_20251109T071313Z/ckpt-100
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import librosa
import soundfile as sf

# Add wavegan to path
sys.path.insert(0, str(Path(__file__).parent.parent / "wavegan-master"))

from tools.compute_vot import compute_vot_from_audio

# Griffin-Lim for audio reconstruction
def spectrogram_to_audio(spec, sr=16000, n_iter=32):
    """Convert mel-spectrogram to audio using Griffin-Lim."""
    # spec: (128, 128) in [-1, 1]
    spec_db = spec * 40  # Scale to ~dB range
    spec_linear = librosa.db_to_power(spec_db)
    
    # Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        spec_linear,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_iter=n_iter
    )
    return audio

def load_generator(checkpoint_path):
    """Load generator from checkpoint."""
    # Assuming checkpoint saved with tf.train.Checkpoint
    # You may need to adjust based on actual checkpoint format
    
    # For now, return None and print instructions
    print("⚠ Generator loading not yet implemented")
    print("Need to:")
    print("  1. Load model architecture from tools/train_ciwgan.py")
    print("  2. Restore weights from checkpoint")
    print("  3. Return generator model")
    return None

def interpolate_latent(z1, z2, steps=10):
    """Linear interpolation between two latent vectors."""
    alphas = np.linspace(0, 1, steps)
    return np.array([alpha * z2 + (1 - alpha) * z1 for alpha in alphas])

def analyze_interpolation(generator, class_short=0, class_long=1, n_steps=10, output_dir="runs/latent_analysis"):
    """
    Generate samples along latent interpolation path and measure VOT.
    
    Hypothesis: If duration is smoothly encoded, VOT should change gradually.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("LATENT SPACE INTERPOLATION ANALYSIS")
    print("="*70)
    
    # Sample two random latent vectors
    z1 = np.random.randn(1, 128).astype(np.float32)
    z2 = np.random.randn(1, 128).astype(np.float32)
    
    # Interpolate
    z_interp = interpolate_latent(z1, z2, steps=n_steps)
    
    results = []
    
    # Test with short class label
    print(f"\nGenerating {n_steps} samples with SHORT class label (class={class_short})...")
    for i, z in enumerate(z_interp):
        # Generate spectrogram
        class_label = np.array([[class_short]], dtype=np.int32)
        spec = generator([z[None, :], class_label], training=False)[0].numpy()
        
        # Convert to audio
        audio = spectrogram_to_audio(spec.squeeze())
        
        # Save audio
        audio_path = output_dir / f"interp_short_{i:02d}.wav"
        sf.write(audio_path, audio, 16000)
        
        # Compute VOT
        vot_ms, conf, _, _ = compute_vot_from_audio(audio, sr=16000)
        
        results.append({
            'interpolation_step': i,
            'alpha': i / (n_steps - 1),
            'class': 'short',
            'vot_ms': vot_ms,
            'confidence': conf
        })
        
        print(f"  Step {i}: VOT = {vot_ms:.2f} ms (conf={conf:.2f})")
    
    # Test with long class label
    print(f"\nGenerating {n_steps} samples with LONG class label (class={class_long})...")
    for i, z in enumerate(z_interp):
        class_label = np.array([[class_long]], dtype=np.int32)
        spec = generator([z[None, :], class_label], training=False)[0].numpy()
        audio = spectrogram_to_audio(spec.squeeze())
        
        audio_path = output_dir / f"interp_long_{i:02d}.wav"
        sf.write(audio_path, audio, 16000)
        
        vot_ms, conf, _, _ = compute_vot_from_audio(audio, sr=16000)
        
        results.append({
            'interpolation_step': i,
            'alpha': i / (n_steps - 1),
            'class': 'long',
            'vot_ms': vot_ms,
            'confidence': conf
        })
        
        print(f"  Step {i}: VOT = {vot_ms:.2f} ms (conf={conf:.2f})")
    
    # Plot results
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in ['short', 'long']:
        subset = df[df['class'] == cls]
        ax.plot(subset['alpha'], subset['vot_ms'], 'o-', label=f'Class={cls}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Interpolation α (0=z1, 1=z2)', fontsize=12)
    ax.set_ylabel('VOT (ms)', fontsize=12)
    ax.set_title('VOT Along Latent Interpolation Path\n(Does VOT change smoothly?)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "latent_interpolation_vot.png", dpi=150)
    print(f"\n✓ Saved plot to {output_dir / 'latent_interpolation_vot.png'}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    short_vots = df[df['class'] == 'short']['vot_ms'].values
    long_vots = df[df['class'] == 'long']['vot_ms'].values
    
    short_variance = np.var(short_vots)
    long_variance = np.var(long_vots)
    
    print(f"\nVOT variance along interpolation path:")
    print(f"  Short class: {short_variance:.2f} ms²")
    print(f"  Long class:  {long_variance:.2f} ms²")
    
    if short_variance < 50 and long_variance < 50:
        print("\n✓ LOW VARIANCE → Class label dominates VOT")
        print("  → Latent vector z has minimal effect on timing")
        print("  → Duration is strongly controlled by class label")
    else:
        print("\n⚠ HIGH VARIANCE → Latent vector affects VOT")
        print("  → Duration is entangled with other latent factors")
        print("  → Class label provides weak duration control")
    
    mean_short = np.mean(short_vots)
    mean_long = np.mean(long_vots)
    
    print(f"\nMean VOT:")
    print(f"  Short class: {mean_short:.2f} ms")
    print(f"  Long class:  {mean_long:.2f} ms")
    print(f"  Difference:  {abs(mean_long - mean_short):.2f} ms")
    
    if abs(mean_long - mean_short) < 3:
        print("\n⚠ CLASSES NOT SEPARATED")
        print("  → Short and long produce similar VOT")
        print("  → Class label may not control duration effectively")
    else:
        print("\n✓ CLASSES SEPARATED")
        print("  → Short and long produce different VOT")
        print("  → Class label successfully controls duration")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (e.g., runs/train/.../ckpt-100)")
    parser.add_argument("--steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--output-dir", default="runs/latent_analysis", help="Output directory")
    args = parser.parse_args()
    
    print(f"Loading generator from {args.checkpoint}...")
    generator = load_generator(args.checkpoint)
    
    if generator is None:
        print("\n" + "="*70)
        print("IMPLEMENTATION STATUS: INCOMPLETE")
        print("="*70)
        print("\nTo complete this tool, you need to:")
        print("1. Implement load_generator() to restore the trained model")
        print("2. Ensure the generator architecture matches train_ciwgan.py")
        print("3. Handle TensorFlow checkpoint format correctly")
        print("\nOnce implemented, this tool will:")
        print("✓ Interpolate between random latent vectors")
        print("✓ Generate audio for each interpolation step")
        print("✓ Measure VOT along the interpolation path")
        print("✓ Determine if duration is smoothly encoded in latent space")
        return
    
    analyze_interpolation(generator, n_steps=args.steps, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
