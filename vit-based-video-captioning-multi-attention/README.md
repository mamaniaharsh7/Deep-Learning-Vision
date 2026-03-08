# Transformer-Based Soccer Video Captioning

## Overview

This project implements a Transformer-based soccer video captioning system using the SoccerNet-Caption dataset. The system generates natural language descriptions for short soccer event clips by combining visual features extracted from video frames with a Transformer decoder that generates captions autoregressively.

Each event clip is represented as 16 sampled frames. A pretrained Vision Transformer (ViT) extracts a 768-dimensional embedding per frame, which serves as the visual memory for the caption generation model.

The caption generator is a 6-layer, 8-head Transformer decoder that uses:

- Masked self-attention for language modeling
- Cross-attention to align words with visual frame features

Several improvements were introduced beyond the baseline model, including multi-head attention, temporal positional embeddings, beam search decoding, and a custom multi-component loss function. These changes significantly improved caption quality.

---

# Dataset

**SoccerNet-Caption**

- 36,894 temporally anchored captions
- 471 broadcast soccer matches
- Approximately 716 hours of video

Each caption describes a 5 to 10 second event clip and includes:

- Match ID
- Timestamp
- Event label
- Caption text

---

# Input Representation

For each event clip:

1. 16 evenly spaced frames are sampled from the event window.
2. Frames are resized to 224 × 224.
3. Each frame is passed through the pretrained model:

`google/vit-base-patch16-224`

4. The CLS token embedding is extracted.

Each frame produces a:

768-dimensional feature vector

Final representation for an event clip:

(16, 768) tensor

These visual features are precomputed and stored on disk so that the vision encoder does not need to be trained end-to-end.

---

# Caption Processing Pipeline

## Caption Cleaning

Text preprocessing includes:

- Removing punctuation
- Converting text to lowercase
- Replacing tokens like `[PLAYER]` with `player`
- Preserving score patterns such as `2-1`

---

## Vocabulary Construction

Vocabulary is built using only the training split.

Steps:

1. Count word frequencies
2. Keep words with frequency ≥ 2

Special tokens added:

```
<PAD>
<SOS>
<EOS>
<UNK>
```

Final vocabulary size:

Approximately 1,130 words.

Mappings created:

- `word_to_idx`
- `idx_to_word`

---

## Tokenization

Each caption is processed as follows:

1. Tokenize caption into words
2. Replace unseen words with `<UNK>`
3. Add `<SOS>` at the start
4. Add `<EOS>` at the end
5. Pad or truncate to maximum length = 30

Final caption tensor shape:

(30)

Training uses next-token prediction:

Input to decoder:

tokens[:, :-1]

Target tokens:

tokens[:, 1:]

This setup allows the model to learn autoregressive caption generation.

---

# Dataset and DataLoader

Each training batch returns:

video_features : (B, 16, 768)  
caption_tokens : (B, 30)

Both are stored as PyTorch tensors.

---

# Model Architecture

## Architecture Type

Decoder-only Transformer with cross-attention to visual memory.

---

## Feature Projection

Original video feature tensor:

(16, 768)

Projected using a linear layer:

768 → 512

Resulting representation:

(16, 512)

---

## Frame Positional Encoding

A learnable positional embedding is added to represent the temporal order of the 16 frames.

This gives the model awareness of event progression.

---

## Token Embedding

Each token index is mapped to a 512-dimensional embedding vector.

Token positional embeddings are added to represent sequence order.

Final token tensor shape:

(30, 512)

---

## Decoder Masking

A causal mask ensures:

token_t cannot attend to tokens after t

This enforces autoregressive caption generation.

---

## Transformer Decoder

Configuration:

- 6 decoder layers
- 8 attention heads
- d_model = 512

Each decoder layer performs:

1. Masked self-attention over tokens
2. Cross-attention to video frame embeddings
3. Feedforward network

Cross-attention allows the model to align generated words with visual features from frames.

---

## Output Projection

Decoder output:

(30, 512)

Final projection layer:

512 → vocab_size

Produces logits:

(30, vocab_size)

Softmax is applied internally when computing cross-entropy loss.

---

# Loss Function

The model uses a multi-component loss:

Total Loss = CE + 0.1 * Rep + 0.1 * Cov − 0.05 * Div

### Cross Entropy Loss

Standard next-token prediction loss.

- Ignores `<PAD>` tokens
- Ensures grammatical correctness

### Repetition Penalty

Measures similarity between token distributions across nearby timesteps.

High similarity is penalized to reduce repeated phrases and prevent mode collapse.

### Coverage Loss

Aggregates attention across frames.

If some frames receive very little attention, a penalty is applied.  
This encourages the model to use information from the entire clip.

### Diversity Loss

Computes cosine similarity between attention heads.

If multiple heads attend to the same patterns, a penalty is applied.  
This encourages attention head specialization.

---

# Training Strategy

Optimizer:

AdamW

Learning Rate:

1e-4

Scheduler:

Linear warmup

Regularization:

- Dropout = 0.1
- Weight decay = 0.01

Stability techniques:

- Gradient clipping
- Mixed precision training using autocast and GradScaler

Memory optimization:

- Gradient accumulation

Early stopping is based on validation BLEU score.

The best model checkpoint is saved automatically.

---

# Inference

Two decoding strategies are used.

## Greedy Decoding

Used during validation for faster evaluation.

## Beam Search

Used during final evaluation.

Beam size = 3

A repetition penalty is applied during beam search to prevent repeating the last few tokens.

---

# Baseline Limitations

The initial baseline model had several issues:

- Only 1 attention head leading to mode collapse
- No temporal positional encoding
- High dropout (0.4) causing underfitting
- Greedy decoding only
- Only 5 training epochs
- Limited vocabulary (961 words)

---

# Improvements Introduced

Eight improvements were implemented:

1. Increased attention heads from 1 to 8
2. Increased decoder layers from 4 to 6
3. Added learnable temporal positional encoding
4. Reduced dropout from 0.4 to 0.1
5. Expanded vocabulary from 961 to 1,130 words
6. Added beam search decoding
7. Introduced custom multi-component loss
8. Increased training from 5 to 50 epochs with early stopping

Best validation performance occurred at epoch 12.

---

# Evaluation Metrics

The model is evaluated using:

- BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- Perplexity

Perplexity is computed as:

exp(total_cross_entropy / total_valid_tokens)

---

# Results

Performance improvements compared to the baseline:

| Metric | Baseline | Final Model |
|------|------|------|
| BLEU-4 | 0.0202 | 0.0468 |
| METEOR | - | +66.9% |
| Perplexity | 64.4 | 4.0 |

Additional observations:

- 73.2% of samples improved
- Gain/Loss ratio: 7.41x

Largest improvements occurred in:

- Cross events (+0.050 BLEU)
- Corner events (+0.043 BLEU)

The final model produces more diverse, temporally grounded captions with significantly improved linguistic quality.
