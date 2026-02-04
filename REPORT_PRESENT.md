# Presentation Report (Outline) — ciwGAN Vietnamese Vowel Length

**Goal**: a slide-ready narrative for presenting the project (demo + results + limitations).

## Slide 1 — Title
- ciwGAN for Vietnamese vowel length contrasts (short vs long)
- Author, date, repo link

## Slide 2 — Motivation
- Why vowel length matters
- Why generative models are interesting for phonology

## Slide 3 — Research question
- Can a conditional GAN generate short/long vowel classes with measurable phonetic structure (VOT, intensity) without direct supervision on those cues?

## Slide 4 — Data & labels
- Speech tokens organized into folders
- Class label inferred from path: contains `long` vs `short`
- Note dataset not committed to Git; local `--data-root` required

## Slide 5 — Representation
- Fixed-size log-mel patches: 128×128
- Pros: stable batching, simple conv nets
- Cons: not explicit variable-duration waveform modeling

## Slide 6 — Model
- WGAN-GP + auxiliary class head
- Conditioning: one-hot class concatenated with latent z

## Slide 7 — Training setup
- critic_steps=5, GP λ=10
- Adam lr=2e-4, β1=0.5, β2=0.9
- Checkpoints + TensorBoard logging

## Slide 8 — Generation
- Generate log-mel → invert with mel_to_audio (Griffin–Lim style)
- Known reconstruction limitations

## Slide 9 — Metrics
- VOT heuristic: burst → voicing onset
- Intensity (RMS dB)
- Compare real vs generated distributions

## Slide 10 — Key results (trend)
- 1 epoch: VOT unrealistic (hundreds of ms)
- 30 epochs: plausible VOT range
- 100 epochs: long-class median matches stored real reference median; short-class remains shifted

## Slide 11 — Intensity findings
- Generated audio quieter than real
- Normalization helps but residual ~15–18 dB gap remains
- Likely driven by Griffin–Lim inversion

## Slide 12 — What’s unresolved
- Short-class VOT mismatch
- Real VOT reference inside repo snapshot may be incomplete

## Slide 13 — Next steps
- Regenerate real VOT CSV at scale
- Balance classes, run ablations
- Finish latent-space probing
- Replace Griffin–Lim with a neural vocoder (HiFi-GAN)

## Slide 14 — Demo
- Train command
- Generate samples
- Compute metrics
- Package deliverables

## Slide 15 — Takeaways
- End-to-end reproducible pipeline
- Clear learning progress
- Honest limitations + concrete next experiments
