"""
Quick spectral analysis: Do short and long vowels have different qualities?

This bypasses the manifest and analyzes all .npy files in processed_data/
by detecting duration from filename patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind
import pandas as pd

DATA_DIR = Path("processed_data")
OUTPUT_DIR = Path("runs/vowel_quality_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_spectral_features(spec):
    """Extract spectral features from mel-spectrogram."""
    # Assume spec is log-mel, shape (128, 128)
    # Convert to linear if needed
    if np.min(spec) < 0:
        spec_linear = np.exp(spec)
    else:
        spec_linear = spec
    
    # Average over time to get spectral profile
    avg_spec = np.mean(spec_linear, axis=1)  # Shape: (128,)
    
    # Spectral centroid (center of mass)
    mel_bins = np.arange(len(avg_spec))
    centroid = np.sum(avg_spec * mel_bins) / np.sum(avg_spec)
    
    # Spectral spread (variance around centroid)
    spread = np.sqrt(np.sum(((mel_bins - centroid) ** 2) * avg_spec) / np.sum(avg_spec))
    
    # Spectral rolloff (85% energy threshold)
    cumsum = np.cumsum(avg_spec)
    rolloff_idx = np.argmax(cumsum >= 0.85 * cumsum[-1])
    
    # Peak frequencies (top 3 bins = rough F1, F2, F3 proxy)
    top3 = np.argsort(avg_spec)[-3:]
    formant_proxy = np.mean(top3)
    
    # Energy
    energy = np.mean(spec_linear)
    
    return {
        'centroid': centroid,
        'spread': spread,
        'rolloff': rolloff_idx,
        'formant_proxy': formant_proxy,
        'energy': energy
    }

def infer_duration_class(filename):
    """
    Infer if file is long or short from filename patterns.
    Vietnamese files have patterns like:
    - Long: aːe55.npy, eːm55.npy, etc. (contains ː)
    - Short: ab55.npy, ad55.npy, etc. (no ː)
    
    Also check for (女) / (男) gender markers (both can be long/short)
    """
    stem = filename.stem  # e.g., 'aːe55' or 'ab55'
    
    # Check for length marker (ː)
    if 'ː' in stem:
        return 'long'
    
    # Check for common Vietnamese long vowel patterns
    if any(v in stem for v in ['iː', 'eː', 'aː', 'oː', 'uː', 'ɛː', 'ɔː', 'əː']):
        return 'long'
    
    # Check for short vowel endings (consonants indicate closed syllables = typically short)
    # Vietnamese short vowels often end in: b, d, g, k, m, n, ng, p, t
    if any(stem.endswith(c) for c in ['55', '213', '35', '21', '24']):
        # These are tone markers - need to check what comes before
        base = stem[:-2]  # Remove tone
        if any(c in base for c in ['p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'ŋ']):
            # Closed syllable → likely short
            return 'short'
    
    # Default: unclear
    return 'unknown'

def main():
    print("Scanning processed_data/ for .npy files...")
    npy_files = list(DATA_DIR.glob("*.npy"))
    print(f"Found {len(npy_files)} files")
    
    results = []
    
    for npy_path in npy_files:
        # Load spectrogram
        try:
            spec = np.load(npy_path)
        except Exception as e:
            print(f"  ⚠ Skipping {npy_path.name}: {e}")
            continue
        
        # Infer duration class
        duration_class = infer_duration_class(npy_path)
        
        # Extract features
        features = compute_spectral_features(spec)
        features['file'] = npy_path.name
        features['duration_class'] = duration_class
        results.append(features)
    
    df = pd.DataFrame(results)
    
    # Filter out unknown
    df = df[df['duration_class'] != 'unknown']
    
    print(f"\nClassified {len(df)} files:")
    print(f"  Long:  {len(df[df.duration_class == 'long'])}")
    print(f"  Short: {len(df[df.duration_class == 'short'])}")
    
    if len(df[df.duration_class == 'long']) == 0 or len(df[df.duration_class == 'short']) == 0:
        print("\n⚠ Cannot perform comparison - need both long and short vowels")
        print("Saving all detected files for manual inspection...")
        df.to_csv(OUTPUT_DIR / "vowel_quality_features_all.csv", index=False)
        return
    
    # Save features
    df.to_csv(OUTPUT_DIR / "vowel_quality_features.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR / 'vowel_quality_features.csv'}")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON: Short vs Long Vowels")
    print("="*70)
    
    short = df[df.duration_class == 'short']
    long = df[df.duration_class == 'long']
    
    for feature in ['centroid', 'spread', 'rolloff', 'formant_proxy', 'energy']:
        short_vals = short[feature]
        long_vals = long[feature]
        
        t_stat, p_val = ttest_ind(short_vals, long_vals)
        
        print(f"\n{feature.upper()}:")
        print(f"  Short: mean={short_vals.mean():.2f}, std={short_vals.std():.2f}")
        print(f"  Long:  mean={long_vals.mean():.2f}, std={long_vals.std():.2f}")
        print(f"  Difference: {abs(long_vals.mean() - short_vals.mean()):.2f}")
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
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features_to_plot = ['centroid', 'spread', 'rolloff', 'formant_proxy', 'energy']
    
    for ax, feature in zip(axes.flat[:5], features_to_plot):
        # Box plot
        data_to_plot = [short[feature], long[feature]]
        bp = ax.boxplot(data_to_plot, labels=['Short', 'Long'], patch_artist=True)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{feature.replace("_", " ").title()}\n(p={ttest_ind(short[feature], long[feature])[1]:.4f})')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide the last subplot
    axes.flat[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vowel_quality_comparison.png", dpi=150)
    print(f"✓ Saved plot to {OUTPUT_DIR / 'vowel_quality_comparison.png'}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    centroid_diff = abs(long['centroid'].mean() - short['centroid'].mean())
    formant_diff = abs(long['formant_proxy'].mean() - short['formant_proxy'].mean())
    
    _, centroid_p = ttest_ind(short['centroid'], long['centroid'])
    _, formant_p = ttest_ind(short['formant_proxy'], long['formant_proxy'])
    
    print(f"\nSpectral Centroid Difference: {centroid_diff:.2f} bins (p={centroid_p:.4f})")
    print(f"Formant Proxy Difference: {formant_diff:.2f} bins (p={formant_p:.4f})")
    
    if centroid_diff > 3 and centroid_p < 0.05:
        print("\n✓ **DURATION-QUALITY COUPLING DETECTED!**")
        print("  → Short and long vowels have significantly different spectral profiles")
        print("  → Vietnamese uses 'Strategy 2': duration + quality")
        print("  → The GAN learned coupled representations")
        print("\nIMPLICATION FOR 15ms SHORT VOT:")
        print("  → The model may be generating a different vowel *quality* for short vowels")
        print("  → This different quality happens to have 15ms VOT timing")
        print("  → The 2× VOT error reflects learned quality differences, not pure duration")
    else:
        print("\n⚠ **NO CLEAR QUALITY DIFFERENCE DETECTED**")
        print("  → Short and long vowels have similar spectral profiles")
        print("  → Vietnamese may use 'Strategy 1': pure duration")
        print("\nIMPLICATION FOR 15ms SHORT VOT:")
        print("  → The 2× VOT error remains unexplained by quality differences")
        print("  → Possible causes:")
        print("     1. Class imbalance in training data")
        print("     2. Discriminator bias toward long vowels")
        print("     3. Training not converged for short vowel timing")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
