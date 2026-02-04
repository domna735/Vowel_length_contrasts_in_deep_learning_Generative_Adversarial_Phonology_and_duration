"""
Analyze spectral differences between Vietnamese short and long vowels.
Check if duration correlates with vowel quality (formant patterns).

Usage:
    python tools/analyze_vowel_qualities.py
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

# Paths
DATA_DIR = Path("processed_data")
MANIFEST = Path("manifest/manifest.csv")
OUTPUT_DIR = Path("runs/vowel_quality_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_manifest():
    """Load manifest with duration labels."""
    df = pd.read_csv(MANIFEST)
    # Assuming columns: relative_path, duration_class (0=short, 1=long)
    return df

def compute_spectral_centroid(spec):
    """
    Compute spectral centroid (center of mass of spectrum).
    Higher centroid = brighter sound (higher formants).
    """
    # spec: (n_mels, n_frames)
    mel_bins = np.arange(spec.shape[0])
    # Weight each mel bin by its magnitude
    centroid = np.sum(spec * mel_bins[:, None], axis=0) / np.sum(spec, axis=0)
    return np.mean(centroid)  # Average across time

def compute_spectral_rolloff(spec, rolloff_pct=0.85):
    """
    Frequency below which rolloff_pct of energy is contained.
    Higher rolloff = more high-frequency energy.
    """
    cumsum = np.cumsum(spec, axis=0)
    total = np.sum(spec, axis=0)
    rolloff_idx = np.argmax(cumsum >= rolloff_pct * total[None, :], axis=0)
    return np.mean(rolloff_idx)

def compute_formant_proxy(spec):
    """
    Rough proxy for formant structure using peak mel bins.
    Returns mean of top 3 peak positions (F1, F2, F3 approximation).
    """
    # Average over time, find peaks
    avg_spec = np.mean(spec, axis=1)
    # Find top 3 peaks
    top3_indices = np.argsort(avg_spec)[-3:]
    return np.mean(top3_indices)

def analyze_file(npy_path):
    """Extract spectral features from .npy spectrogram."""
    spec = np.load(npy_path)  # Shape: (n_mels, n_frames)
    
    # Ensure non-negative (if log-mel, exponentiate)
    if np.min(spec) < 0:
        spec = np.exp(spec)  # Convert log-mel to linear
    
    features = {
        'spectral_centroid': compute_spectral_centroid(spec),
        'spectral_rolloff': compute_spectral_rolloff(spec),
        'formant_proxy': compute_formant_proxy(spec),
        'mean_energy': np.mean(spec),
        'energy_variance': np.var(spec),
    }
    return features

def main():
    print("Loading manifest...")
    manifest = load_manifest()
    
    # Filter for Vietnamese only (if manifest has language column)
    if 'language' in manifest.columns:
        manifest = manifest[manifest['language'] == 'Vietnamese']
    
    print(f"Found {len(manifest)} Vietnamese files")
    
    # Collect features by duration class
    results = []
    
    for _, row in manifest.iterrows():
        rel_path = row['rel_path']
        # Convert length_class (long/short) to binary
        duration_class = 1 if row['length_class'] == 'long' else 0
        
        # Convert to .npy path
        npy_path = DATA_DIR / Path(rel_path).with_suffix('.npy')
        
        if not npy_path.exists():
            continue
        
        features = analyze_file(npy_path)
        features['duration_class'] = duration_class
        features['file'] = rel_path
        results.append(features)
    
    df = pd.DataFrame(results)
    print(f"\nAnalyzed {len(df)} files")
    print(f"  Short vowels: {len(df[df.duration_class == 0])}")
    print(f"  Long vowels: {len(df[df.duration_class == 1])}")
    
    # Save raw results
    df.to_csv(OUTPUT_DIR / "vowel_quality_features.csv", index=False)
    print(f"\n✓ Saved features to {OUTPUT_DIR / 'vowel_quality_features.csv'}")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON: Short vs Long Vowels")
    print("="*70)
    
    short = df[df.duration_class == 0]
    long = df[df.duration_class == 1]
    
    for feature in ['spectral_centroid', 'spectral_rolloff', 'formant_proxy', 'mean_energy']:
        short_vals = short[feature]
        long_vals = long[feature]
        
        t_stat, p_val = ttest_ind(short_vals, long_vals)
        
        print(f"\n{feature.upper()}:")
        print(f"  Short: mean={short_vals.mean():.2f}, std={short_vals.std():.2f}")
        print(f"  Long:  mean={long_vals.mean():.2f}, std={long_vals.std():.2f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}", end="")
        
        if p_val < 0.001:
            print("  *** HIGHLY SIGNIFICANT ***")
        elif p_val < 0.01:
            print("  ** SIGNIFICANT **")
        elif p_val < 0.05:
            print("  * SIGNIFICANT *")
        else:
            print("  (not significant)")
    
    # Visualization
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    features_to_plot = ['spectral_centroid', 'spectral_rolloff', 'formant_proxy', 'mean_energy']
    
    for ax, feature in zip(axes.flat, features_to_plot):
        # Violin plot
        parts = ax.violinplot(
            [short[feature], long[feature]],
            positions=[0, 1],
            showmeans=True,
            showextrema=True
        )
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Short', 'Long'])
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{feature.replace("_", " ").title()}\n(t-test p={ttest_ind(short[feature], long[feature])[1]:.4f})')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vowel_quality_comparison.png", dpi=150)
    print(f"✓ Saved plot to {OUTPUT_DIR / 'vowel_quality_comparison.png'}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    centroid_diff = long['spectral_centroid'].mean() - short['spectral_centroid'].mean()
    formant_diff = long['formant_proxy'].mean() - short['formant_proxy'].mean()
    
    if abs(centroid_diff) > 2 and ttest_ind(short['spectral_centroid'], long['spectral_centroid'])[1] < 0.05:
        print("\n✓ DURATION-QUALITY COUPLING DETECTED!")
        print("  → Short and long vowels have significantly different spectral centroids")
        print("  → This confirms Vietnamese uses 'Strategy 2' (duration + quality)")
        print("  → The GAN's 15ms short-vowel VOT may reflect learned quality differences")
    else:
        print("\n⚠ NO CLEAR QUALITY DIFFERENCE DETECTED")
        print("  → Short and long vowels have similar spectral profiles")
        print("  → This suggests Vietnamese uses 'Strategy 1' (pure duration)")
        print("  → The GAN's 15ms short-vowel VOT remains unexplained")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
