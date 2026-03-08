# Transformer-based Soccer Video Captioning

## Overview

This project implements a **Transformer-based video captioning system for soccer events** using the **SoccerNet-Caption dataset**.

The system generates natural language descriptions for short soccer event clips by combining:

- **Vision Transformer (ViT)** for frame-level visual feature extraction
- **Transformer Decoder** for autoregressive caption generation

Each event clip is represented as **16 sampled frames**, which are encoded into embeddings using a pretrained Vision Transformer. These embeddings serve as **visual memory** for a Transformer decoder that generates captions describing the event.

The model extends the baseline architecture with **multi-head attention, temporal frame positional embeddings, beam search decoding, and a custom multi-component loss function**, significantly improving caption quality.

---

# Dataset

**SoccerNet-Caption**

- 36,894 temporally anchored captions
- 471 broadcast soccer matches
- ~716 hours of video content

Each caption corresponds to a **5–10 second event clip** and contains:

- Match ID
- Timestamp
- Event label
- Caption text

---

# Input Representation

For each event clip:

1. **16 evenly spaced frames** are sampled from the video.
2. Frames are resized to **224 × 224**.
3. Each frame is passed through:


google/vit-base-patch16-224


4. The **CLS token embedding** is extracted.

Output per frame:


768-dimensional feature vector


Final representation for a clip:


(16, 768) feature tensor


These features are **precomputed and stored**, avoiding end-to-end vision training.

---

# Caption Processing Pipeline

## Caption Cleaning

Text preprocessing steps include:

- Removing punctuation
- Converting text to lowercase
- Replacing tokens like `[PLAYER]` with `player`
- Preserving score patterns such as `2-1`

---

## Vocabulary Construction

Vocabulary is built **only from the training set**.

Steps:

1. Count word frequencies
2. Keep words with **frequency ≥ 2**

Special tokens added:

<PAD> <SOS> <EOS> <UNK> ```

Final vocabulary size:

~1,130 words

Mappings created:

word_to_idx

idx_to_word

Tokenization

Each caption is processed as follows:

Tokenize words

Replace unseen words with <UNK>

Add <SOS> at the beginning

Add <EOS> at the end

Pad or truncate to maximum length = 30

Final caption tensor shape:

(30)

Training setup:

Input to decoder: tokens[:, :-1]
Target tokens:   tokens[:, 1:]

This enables next-token prediction training.

Dataset and DataLoader

Each training batch returns:

video_features : (B, 16, 768)
caption_tokens : (B, 30)

Both are stored as PyTorch tensors.

Model Architecture
Architecture Type

Decoder-only Transformer with cross-attention to visual memory

Feature Projection

Original video features:

(16, 768)

Projected using a linear layer:

768 → 512

New representation:

(16, 512)
Frame Positional Encoding

A learnable positional embedding is added to represent the temporal order of the 16 frames.

This allows the model to understand event progression.

Token Embedding

Each token index is converted into:

512-dimensional embedding

Token positional embeddings are added to encode sequence order.

Final token tensor:

(30, 512)
Decoder Masking

A causal mask ensures:

token_t cannot attend to tokens > t

This enforces autoregressive generation.

Transformer Decoder

Configuration:

Layers: 6
Attention Heads: 8
Model Dimension: 512

Each layer contains:

Masked self-attention (language modeling)

Cross-attention to video frame embeddings

Feedforward network

Cross-attention allows the model to align generated words with visual features from frames.

Output Projection

Decoder outputs:

(30, 512)

Final linear layer:

512 → vocab_size

Produces logits:

(30, vocab_size)

Softmax is applied internally during cross-entropy loss computation.

Loss Function

The model uses a multi-component loss:

Total Loss = CE + 0.1 * Rep + 0.1 * Cov − 0.05 * Div
1. Cross Entropy Loss

Standard next-token prediction loss.

Ignores <PAD> tokens

Drives grammatical correctness

2. Repetition Penalty

Measures similarity between predicted token distributions across nearby timesteps.

High similarity is penalized.

Goal:

Reduce repeated phrases

Avoid mode collapse

3. Coverage Loss

Aggregates attention across all frames.

If certain frames receive low cumulative attention, a penalty is applied.

Goal:

Encourage the model to use information from the entire video clip

4. Diversity Loss

Computes cosine similarity between attention heads.

If heads focus on identical patterns:

Apply penalty

Goal:

Encourage attention head specialization

Training Strategy

Optimizer:

AdamW

Learning Rate:

1e-4

Scheduler:

Linear warmup

Regularization:

Dropout = 0.1
Weight Decay = 0.01

Stability Techniques:

Gradient clipping

Mixed precision training (autocast + GradScaler)

Memory Optimization:

Gradient accumulation

Early Stopping:

Based on validation BLEU score

Best model checkpoint is saved automatically.

Inference

Two decoding strategies are used.

Greedy Decoding

Used during validation for speed.

Beam Search Decoding

Used during final evaluation.

Configuration:

Beam size = 3

Includes repetition penalty to prevent repeating the last 3 tokens.

Baseline Limitations

Initial baseline model had several issues:

Single attention head → mode collapse

No temporal encoding

High dropout (0.4) → underfitting

Greedy decoding only

Only 5 training epochs

Limited vocabulary (961 words)

Improvements Introduced

Eight major improvements were implemented:

Increased attention heads 1 → 8 (removed mode collapse)

Increased decoder depth 4 → 6 layers

Added learnable temporal positional encoding

Reduced dropout 0.4 → 0.1

Expanded vocabulary 961 → 1,130 words

Implemented beam search decoding

Introduced custom multi-component loss

Extended training 5 → 50 epochs with early stopping

Best validation performance was reached at epoch 12.

Evaluation Metrics

Model performance was evaluated using:

BLEU-4

METEOR

ROUGE-L

CIDEr

Perplexity

Perplexity is computed as:

exp(total_cross_entropy / total_valid_tokens)
Results

Performance improvements compared to baseline:

Metric	Baseline	Final Model
BLEU-4	0.0202	0.0468
METEOR	+66.9%	
Perplexity	64.4	4.0

Additional observations:

73.2% of samples improved

Gain/Loss ratio: 7.41×

Largest improvements occurred in:

Cross events (+0.050)

Corner events (+0.043)

The final model produces more diverse, temporally grounded captions with significantly better linguistic quality.
