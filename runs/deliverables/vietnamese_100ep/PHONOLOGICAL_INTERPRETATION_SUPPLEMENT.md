# Phonological Interpretation Supplement — Vietnamese Vowel Length ciwGAN

**Bundle**: `vietnamese_100ep`  
**Date**: 2026-02-04  

This note is a short, interpretation-focused complement to the main quantitative reports. It is written to be honest about what the current artifacts can and cannot support.

## 1) What the model is actually controlling

The ciwGAN is trained with an explicit conditioning variable `class_id ∈ {0,1}` intended to represent **short vs long**. In the current implementation, conditioning is injected into the generator input and also encouraged via an auxiliary classifier head on the critic.

Important nuance: the **acoustic representation is fixed-length** (128 frames). That makes the “short vs long” label function more like a *coarse duration/timing style* cue than literal waveform duration control.

## 2) What the evaluation suggests (and what it doesn’t)

### 2.1 VOT

- Generated VOT distributions become plausible by 30 epochs.
- At 100 epochs, the **long-class** median VOT matches the stored “real” median in this repo snapshot.

Caveat: the repository’s current `runs/vot_real.csv` is known to be incomplete in this lightweight workspace snapshot (some comparison summaries show `real_n = 1`). That means the *shape* of the real VOT distribution cannot be estimated from this repo alone.

Interpretation (conservative):
- The generator can produce **stop-like onsets** whose burst→voicing delays fall in a reasonable range.
- The long-class result is encouraging, but not yet a robust claim about matching real Vietnamese VOT distributions.

### 2.2 Intensity

The generated audio is consistently quieter than real audio even after normalization. This is most consistent with Griffin–Lim mel inversion behavior rather than a “phonological” property of the model.

Interpretation:
- Intensity mismatch is primarily a **signal reconstruction bottleneck**, not strong evidence for or against phonological realism.

## 3) Why short-vowel VOT might differ

A persistent observation is that **short-class** generated VOT remains shifted upward in the stored summaries.

Plausible contributors:
- **Class imbalance**: long:short imbalance can cause more reliable learning in the majority class.
- **Heuristic measurement bias**: VOT detection may latch onto synthetic transients differently.
- **Entanglement**: the class signal may not cleanly isolate the specific cue used by the VOT estimator.

## 4) Recommended next analyses (high value)

1. **Regenerate a full real VOT reference CSV** from the full audio dataset and re-run comparisons.
2. **Balance the dataset** (sampling/weighting) and re-train a small ablation to see if short-class VOT shifts.
3. **Latent-space probing**: interpolate `z` within a class and track whether VOT changes smoothly; this tests whether timing cues are encoded in `z` versus dominated by the class label.
4. **Swap Griffin–Lim for a neural vocoder** (e.g., HiFi-GAN) to remove intensity/phase artifacts that can confuse both human listening and heuristic metrics.

## 5) Bottom line

The current artifacts support the claim that the system learns **structured onset timing** and improves dramatically with training, but they do not yet support a strong phonological claim about matching real Vietnamese VOT distributions—primarily due to incomplete real-reference VOT data inside this code-only snapshot.
