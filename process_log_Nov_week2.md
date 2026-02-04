# Process Log Nov 2025 Week2.

## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:

## Must run before anything (activate env)

Before running any training or analysis scripts, activate the project's Python environment in PowerShell. Two venvs are present in this repository; choose the one you normally use:


```powershell
.\.venv_cpu\Scripts\Activate.ps1
# confirm
python -V
```


```powershell
.\.venv_gpu\Scripts\Activate.ps1
# or, if you use WSL, activate your WSL venv inside Ubuntu instead
# source ~/specgan-venv/bin/activate
```

Notes:

### APPEND RULE
All new daily entries for November Week 2 must be appended below this line (bottom of file). Do not insert edits in earlier dated sections; instead add a fresh block:

## 2025-11-11 | Investigating the Vietnamese Short Vowel VOT Mystery

**Context**: The 100-epoch model shows asymmetric VOT performance:
- Long vowels: 7.5ms real → 7.5ms generated (100% accuracy) ✅
- Short vowels: 7.5ms real → 15ms generated (50% accuracy) ⚠️

**Research Question**: Why do generated short vowels have 2× the expected VOT?

### Hypothesis Testing: Duration-Quality Coupling

**Initial Hypothesis** (from COMPLETE_EVALUATION_REPORT.md):
- Vietnamese might use "Strategy 2" (duration + vowel quality coupling)
- Short vowels = different vowel quality (e.g., /ɛ/) + shorter duration
- Long vowels = different vowel quality (e.g., /eː/) + longer duration
- The GAN learned these as fundamentally different phonemes, not just duration variants

**Test**: Analyze spectral differences between short and long vowels in the **real** Vietnamese dataset.

**Tool Created**: `tools/analyze_vowel_qualities_simple.py`
- Analyzes 680 `.npy` spectrograms in `processed_data/`
- Infers duration class from filename patterns (presence of `ː` marker)
- Computes spectral features: centroid, spread, rolloff, formant proxy, energy
- Performs t-tests to detect significant quality differences

**Results** (runs/vowel_quality_analysis/):
```
Classified 390 files:
  Long:  286
  Short: 104

STATISTICAL COMPARISON:
                   Short         Long          Difference    Significance
Spectral Centroid  16.90 ± 8.70  18.53 ± 9.44  1.63 bins     NOT SIGNIFICANT (p=0.1249)
Spectral Spread    2.43 ± 2.74   4.47 ± 3.88   2.04 bins     *** HIGHLY SIGNIFICANT (p<0.001)
Spectral Rolloff   18.35 ± 10.10 21.86 ± 11.13 3.52 bins     ** SIGNIFICANT (p=0.0049)
Formant Proxy      16.90 ± 8.79  17.92 ± 9.07  1.02 bins     NOT SIGNIFICANT (p=0.3233)
```

**Interpretation**:
- ⚠️ **Hypothesis REJECTED**: No clear duration-quality coupling detected
  - Spectral centroid (brightness): NOT significantly different
  - Formant proxy (F1/F2/F3 approximation): NOT significantly different
  - Energy: NOT significantly different
- ✓ **Spectral spread and rolloff ARE different**:
  - Long vowels have wider spectral spread (4.47 vs 2.43 bins, p<0.001)
  - Long vowels have higher rolloff (21.86 vs 18.35 bins, p<0.01)
  - **Interpretation**: Long vowels may have more **spectral complexity** (richer harmonics) rather than different **vowel quality**
  - This supports **"Strategy 1" (pure duration)** more than "Strategy 2" (duration + quality)

**Conclusion**:
- Vietnamese short and long vowels appear to have **similar formant structures** (same vowel quality)
- Difference is primarily **temporal** (duration) with some **spectral complexity** differences
- The 15ms short-vowel VOT error is **NOT explained by quality coupling**

### Remaining Questions

1. **Why 15ms specifically?**
   - Is there a Vietnamese phoneme in the training data with ~15ms VOT?
   - Is 15ms the "default" VOT the generator produces when uncertain?

2. **Class Imbalance?**
   - Analysis found: Long=286 files, Short=104 files (2.75:1 ratio)
   - **Hypothesis**: Discriminator saw 2.75× more long vowels during training
   - Model may be biased toward long-vowel-like features

3. **Latent Space Structure**
   - Does the latent vector `z` control duration smoothly?
   - Or does the class label `c` provide only weak duration control?
   - **Tool needed**: `tools/explore_latent_space.py` (skeleton created, needs generator loading)

4. **Training Dynamics**
   - Did short-vowel VOT improve from epoch 30 to 100?
     - Epoch 30: 3.75ms error (from earlier logs)
     - Epoch 100: 7.50ms error (current)
     - **This got WORSE!** Suggests mode collapse or discriminator bias

### Proposed Next Steps

**Investigation 1: Check Training Data VOT Distribution**
```python
# Analyze real Vietnamese audio VOT by duration class
# Question: Do short vowels in training data actually have ~15ms VOT?
# Or is 15ms an artifact of the generator?
```

**Investigation 2: Latent Space Interpolation** (requires checkpoint loading)
```python
# Generate samples along latent interpolation paths
# Measure VOT at each step
# Question: Does VOT change smoothly, or is it controlled only by class label?
```

**Investigation 3: Discriminator Attention Analysis**
```python
# Visualize what the discriminator focuses on
# Question: Does it primarily look at duration? Or other features?
```

**Investigation 4: Retrain with Balanced Dataset**
```python
# Oversample short vowels or undersample long vowels to 1:1 ratio
# Question: Does balanced training fix the 15ms short-vowel VOT?
```

**Files Created**:
- `tools/analyze_vowel_qualities_simple.py` — Spectral analysis (COMPLETE)
- `tools/explore_latent_space.py` — Latent space interpolation (SKELETON ONLY, needs generator loading)
- `runs/vowel_quality_analysis/vowel_quality_features.csv` — Raw spectral features (390 files)
- `runs/vowel_quality_analysis/vowel_quality_comparison.png` — Statistical visualization

**Key Finding**: Class imbalance (2.75:1 long:short ratio) is the leading candidate explanation for 15ms short-vowel VOT error. The initial "duration-quality coupling" hypothesis is rejected by spectral analysis.

