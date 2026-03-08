# Multimodal Fashion Recommendation System (Multimodal RAG)

## Overview

A **Multimodal Retrieval-Augmented Generation (RAG)** architecture extends traditional RAG systems by incorporating multiple data types such as **text and images**, enabling AI systems to understand and respond to queries using diverse information sources.

The system encodes different modalities into a **shared embedding space**, stores them in a **vector database**, and retrieves relevant information to augment the response generation process.

This project builds a **Multimodal Shopping Assistant** capable of recommending men’s fashion items using a combination of **visual input (image)** and **natural language queries**.

---

## Core Tasks Addressed

This system combines several multimodal AI tasks:

- **Cross-Modal Retrieval**  
  Retrieve images using text queries or retrieve text-based items using image queries.

- **Multi-Modal Classification**  
  Use both image and text signals to improve prediction quality.

- **Visual Question Answering (VQA)**  
  Answer questions about visual inputs.

---

## Problem Statement

Build a **Multimodal Shopping Assistant** that recommends fashion products using both an image and a text prompt.

### Inputs
- An image of a clothing item (example: a shirt)
- A natural language prompt specifying preferences  
  Example:  
  `"Something like this but with a collar and no buttons under $50"`

### Outputs
The system returns:

- A **ranked list of matching product recommendations**
- Product details including:
  - Images
  - Price
  - Description
- A brief explanation for each recommendation

---

# System Architecture

## 1. Data Collection

Product data is collected using **web scraping from e-commerce platforms** such as:

- Uniqlo
- H&M
- Zara

### Collected Fields

- Product image
- Title
- Description
- Price
- Tags / categories
- Reviews (if available)

### Storage Format

- JSONL
- Pandas DataFrame

---

## 2. Feature Extraction

### Image Modality

Encoder:
- Pretrained **CLIP** or **BLIP image encoder**

Output:
- Image embeddings of **512 to 1024 dimensions**

These embeddings capture visual attributes such as:

- Color
- Style
- Texture
- Clothing structure

---

### Text Modality

Encoder options:

- **BERT**
- **DistilBERT**
- **Sentence-BERT**

Applied to:

- Product title + description
- User query

Output:

- Dense text embeddings representing semantic meaning

---

### Fusion Strategy

Visual and text features are combined using one of the following strategies:

- **Concatenation**
- **Weighted Sum**
- **Cross-Attention** (optional advanced approach)

The fused representation creates a **single multimodal product embedding**.

---

## 3. Vector Database and Similarity Search

Libraries used:

- **FAISS**
- **ChromaDB**

Each product is stored as a **combined embedding vector**.

### Retrieval Process

1. Encode the user's image and text query
2. Generate a multimodal query embedding
3. Compute similarity with stored embeddings using **cosine similarity**
4. Retrieve **top-k most similar products**

---

## 4. Query Interface

### Inputs

- User-uploaded clothing image
- Natural language prompt specifying constraints

Example query:
"Something like this but with a collar and no buttons under $50"


### Outputs

Top-K recommended products including:

- Image preview
- Product title
- Price
- Product description

---

## 5. Explanation Module (Optional)

The system can generate explanations for recommendations.

Approaches:

- Template-based explanations
- LLM-powered explanations using:
  - **T5-small**
  - **GPT APIs**

### Example Explanation

> Selected for its similarity in fabric and design, addition of collar, and price under $50.

This improves **user trust and interpretability** of recommendations.

---

# Technologies Used

- Python
- CLIP / BLIP
- BERT / Sentence-BERT
- FAISS / ChromaDB
- Pandas
- Web Scraping
- Multimodal Embedding Models
- Retrieval-Augmented Generation

---

# Key Capabilities

- Multimodal retrieval using **image + text**
- Fashion recommendation with semantic similarity
- Vector database powered search
- Explainable AI recommendations
- Cross-modal reasoning

---

# Future Improvements

- Cross-attention based multimodal fusion
- Personalized recommendation modeling
- Style clustering and trend analysis
- User feedback loops
- Large-scale product catalog support
