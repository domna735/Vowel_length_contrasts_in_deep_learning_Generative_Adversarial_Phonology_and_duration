#Vowel Length Contrasts in Deep Learning: Generative Adversarial Phonology and Duration#

Is a research-oriented project exploring the use of GANs to model phonological contrasts in vowel duration. The system is designed to generate synthetic speech with controllable vowel length, supporting downstream linguistic analysis and phonological modeling.

🧠 Research Motivation
Traditional phonological analysis often relies on handcrafted features and limited datasets. This project aims to bridge the gap between generative modeling and phonological theory by using deep learning to simulate vowel length contrasts across languages and dialects.

🧪 Data & Preprocessing
The system uses curated speech datasets annotated for vowel duration. Spectrograms are extracted and normalized to serve as input for GAN-based generation. The pipeline supports both raw waveform and spectrogram-based modeling, with optional phoneme alignment for linguistic interpretability.

🧱 Modeling Approach
A multi-stage GAN framework is adopted:

- Generator: SpecGAN-based architecture with controllable duration parameters  
- Discriminator: CNN-based classifier trained to distinguish natural vs synthetic vowel contrasts  
- Loss Functions: WGAN-GP for stability, with auxiliary duration loss to enforce phonological control  
- Architecture Enhancements:  
  - Upsampling layers optimized for temporal resolution  
  - Conditional inputs for vowel class and target duration  
  - tf.keras migration from legacy TF1 codebase for reproducibility

🔧 Training Strategy
Training involves staged curriculum learning, gradually increasing contrast difficulty. The model is evaluated on both perceptual quality and phonological accuracy, using metrics such as:

- Spectral convergence  
- Duration deviation  
- Phoneme classification accuracy  
- GAN stability indicators (gradient norms, discriminator loss)

📊 Evaluation & Findings
- The system successfully generates vowel contrasts with controllable duration shifts  
- WGAN-GP improves convergence and reduces mode collapse  
- TF2 migration enables reproducible experiments and easier deployment  
- Generated samples support phonological analysis of contrastive length in synthetic speech

🚀 Outcome
- A controllable GAN-based speech generation system  
- A reproducible TF2 pipeline for phonological modeling  
- Co-authored work presented at UC Berkeley Annual Phonology Meeting (2025)  
- Targeted for submission to UMass (Jan 2026)

This project demonstrates the feasibility of using deep generative models for phonological contrast modeling and lays the groundwork for future work in generative phonology, linguistic conditioning, and cross-linguistic contrast simulation.
