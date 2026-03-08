# Multimodal Retrieval and Structured Representation Fusion

Multimodal Shopping Assistant capable of recommending men’s fashion items based on a 
combination of visual input (image) and natural language queries. 
 
Given: 
  •  An image of a clothing item (e.g., a shirt) 
  •  A textual prompt specifying the user’s preferences (e.g., “something like this but with a 
collar and no buttons under $50”) 
 
The system returns: 
  •  A ranked list of matching product recommendations 
  •  Details including images, price, and descriptions 
  •  A brief explanation for why each product was recommended 
 
 
 
System Architecture  
 
1. Data Collection 
  •  Web scraping from e-commerce platforms (e.g., Uniqlo, H&M, Zara) 
  •  Collected fields: 
  •  Product image 
  •  Title 
  •  Description 
  •  Price 
  •  Tags/categories 
  •  Reviews (if available) 
  •  Storage format: JSONL or Pandas DataFrame 
 
2. Feature Extraction 
 
Image Modality 
  •  Encoder: Pretrained CLIP or BLIP image encoder 
  •  Converts images into 512–1024 dimensional embeddings 
 
Text Modality 
  •  Encoder: Pretrained BERT, DistilBERT, or Sentence-BERT 
  •  Applied to: 
  •  Product title + description (for items) 
  •  User’s query (for search) 
 
Fusion Strategy 
  •  Combine visual and text features via: 
  •  Concatenation 
  •  Weighted sum 
  •  Cross-attention (optional stretch goal) 
 
3. Vector Database & Similarity Search 
  •  Library: FAISS or ChromaDB 
  •  Each product is stored as a combined embedding vector 
  •  User query is encoded into a query vector, compared via cosine similarity 
  •  Top-k results are returned 
 
4. Query Interface 
 
Inputs: 
  •  User-uploaded clothing image 
  •  Prompt text: constraints/preferences 
 
Outputs: 
  •  Top k recommended products with: 
  •  Image preview 
  •  Title, price, description 
  •  Optional explanation (“Matches the requested collar style, full sleeves, and price 
range”) 
 
5. Explanation Module (Optional) 
  •  Template-based or LLM-powered (T5-small or GPT API) 
  •  Example: 
“Selected for its similarity in fabric and design, addition of collar, and price under $50.”
