# CLIP and Image Retrieval

CLIP (Contrastive Language-Image Pre-Training) is a neural model from OpenAI that
learns a shared embedding space for images and text by training on hundreds of
millions of image-text pairs from the internet. In this shared space, an image of
a dog and the text "a photograph of a dog" produce nearby vectors while unrelated
images and texts produce distant vectors. CLIP enables cross-modal retrieval -
finding images given a text query, finding text given an image query, and ranking
images by semantic similarity to natural language descriptions - without any
task-specific fine-tuning. It is the foundational model behind most modern image
retrieval systems and multimodal RAG pipelines.

## Intuition

Text retrieval works by encoding both documents and queries in the same vector space
and measuring similarity. Image retrieval needs the same capability but with images
instead of documents. The challenge is that images and text are fundamentally
different modalities - a pixel grid and a sequence of tokens have no obvious common
representation.

CLIP solves this with a simple but powerful idea: train two separate encoders -
one for images, one for text - with a shared output space. The training objective
forces the image encoder and text encoder to agree on what is similar. An image of
a cat and the caption "a cat sitting on a windowsill" must produce nearby vectors.
The same image paired with "a dog running in a park" must produce distant vectors.

After training on 400 million image-text pairs, the encoders learn to map both
modalities into a space where semantic similarity translates to geometric proximity.
The resulting embeddings support:

```
Text → Image retrieval:  "a red sports car" → find matching images
Image → Text retrieval:  [image of car] → find matching captions
Image → Image retrieval: [image of car] → find visually similar images
Zero-shot classification: [image] → "is this a cat or a dog?" without training
```

## CLIP Architecture

CLIP has two encoder components:

### Image encoder

A Vision Transformer (ViT) or ResNet that encodes images into fixed-size vectors.

```
Input:  224×224 RGB image (or larger for ViT-L)
Process:
  ViT variant: split image into 16×16 patches, encode as sequence with transformer
  ResNet variant: convolutional feature extraction + attention pooling
Output: 512-dim (ViT-B/32) or 768-dim (ViT-L/14) embedding vector
```

Common CLIP image encoder variants:

```
ViT-B/32:  patch_size=32, 512-dim output, fastest
ViT-B/16:  patch_size=16, 512-dim output, better quality
ViT-L/14:  patch_size=14, 768-dim output, highest quality
ViT-H/14:  patch_size=14, 1024-dim output, OpenCLIP large variant
```

### Text encoder

A transformer (similar to GPT-2) that encodes text into vectors of the same
dimension as the image encoder output.

```
Input:  text string (up to 77 tokens for standard CLIP)
Process: transformer encoder → take [EOS] token representation
Output: 512-dim or 768-dim embedding vector (same dim as image encoder)
```

The 77-token limit is a significant constraint for retrieval tasks involving
long captions or descriptions. Models like Long-CLIP extend this to 248 tokens.

### Contrastive training objective

Given a batch of N (image, text) pairs, CLIP trains both encoders simultaneously
to maximize cosine similarity for matching pairs and minimize it for non-matching:

```
Batch: [(I₁, T₁), (I₂, T₂), ..., (Iₙ, Tₙ)]

Similarities:
  sim(Iᵢ, Tⱼ) = cosine_similarity(image_encoder(Iᵢ), text_encoder(Tⱼ))

Loss (InfoNCE):
  L_I = cross_entropy over image → text direction
  L_T = cross_entropy over text → image direction
  L   = (L_I + L_T) / 2
```

The loss pushes diagonal elements (matching pairs) high and off-diagonal elements
(non-matching pairs) low. With large batches (CLIP uses batch_size=32768), this
creates 32767 negative examples per positive - very strong contrastive signal.

## CLIP vs Text-Only Retrieval

| Property                                 | CLIP                    | Text retrieval (BM25/dense)  |
| ---------------------------------------- | ----------------------- | ---------------------------- |
| Modalities                               | Image + text            | Text only                    |
| Query types                              | Text or image           | Text only                    |
| Document types                           | Images, text, or both   | Text only                    |
| Shared embedding space Yes (cross-modal) | No (text only)          |
| Zero-shot capability                     | Strong                  | Depends on pretraining       |
| Context window                           | 77 tokens (text)        | 512+ tokens (BERT-based)     |
| Vocabulary mismatch                      | Handled via visual sem. | Handled via dense emb.       |
| Training data                            | 400M image-text pairs   | Text corpora (Wikipedia etc) |
| Fine-tuning needed                       | Often no (zero-shot)    | Often yes (domain-specific)  |

## Image Retrieval Pipeline

### Text-to-image retrieval

The most common use case: find images matching a text query.

```
Offline (index time):
  for each image in corpus:
    image_embedding = clip.encode_image(image)   → 512-dim vector
    store in vector index (FAISS, Qdrant, etc.)

Online (query time):
  text_embedding = clip.encode_text(query)       → 512-dim vector
  top_k_images   = ANN_search(text_embedding, image_index, k)
  return top_k_images
```

### Image-to-image retrieval

Find visually similar images given an image query:

```
query_embedding = clip.encode_image(query_image)
top_k_similar   = ANN_search(query_embedding, image_index, k)
```

Since both the query and corpus use the image encoder, this is visual similarity
search - images that look similar produce similar embeddings.

### Image-to-text retrieval

Find relevant documents given an image query:

```
Offline: encode all documents with text encoder
Online:  encode query image with image encoder
         search text embeddings with image query vector
```

Works because image and text encoders share the same embedding space - an image
of a scientific diagram can retrieve text documents describing the same concept.

## CLIP Limitations

### Short text context

Standard CLIP processes only 77 tokens. Longer captions or descriptions are
truncated. For retrieval over long documents, CLIP text embeddings are too
constrained - use a separate dense text retrieval model and combine with CLIP
for the visual component.

### Object composition

CLIP is strong at recognizing individual objects and styles but struggles with
spatial relationships and compositional queries:

```
Strong:  "a red car"                       → finds red cars
Weak:    "a red car to the left of a bus"  → spatial reasoning difficult
Strong:  "a happy dog"                     → finds happy-looking dogs
Weak:    "a dog chasing a cat"             → action/relationship difficult
```

### Fine-grained discrimination

CLIP embeddings encode high-level semantics but may not capture subtle differences:

```
"a golden retriever" and "a labrador retriever" may have similar embeddings
"Leonardo da Vinci painting" vs "Monet painting" may not be well-separated
```

Domain-specific CLIP variants (medical imaging, satellite imagery) often produce
better fine-grained discrimination for specialized domains.

### No spatial information

Image embeddings are global - they represent the whole image as one vector. A
query for "text in the upper right corner" cannot be answered by global CLIP
embeddings. Region-level search requires different approaches (patch-level
embeddings, object detection + retrieval).

## CLIP Variants and Extensions

### OpenCLIP

Open-source CLIP training and model zoo by LAION. Trained on LAION-400M and
LAION-5B (5 billion image-text pairs). Produces stronger representations than
original OpenAI CLIP for many benchmarks.

```
Key models:
  ViT-H-14:     1.8B params, highest quality LAION model
  ViT-G-14:     1.0B params, good quality/speed tradeoff
  ViT-B-32:     87M params, fast, good baseline
```

### SigLIP (Google, 2023)

CLIP variant using sigmoid loss instead of softmax. Better performance with smaller
batch sizes - more practical for fine-tuning:

```
SigLIP loss:
  L = -mean(
    y × log(sigmoid(sim)) + (1-y) × log(1 - sigmoid(sim))
  )

Where y=1 for matching pairs, y=0 for non-matching
```

### ALIGN (Google, 2021)

Similar to CLIP but trained on a larger, noisier dataset (1.8B image-text pairs).
Shows that scale can compensate for data quality in visual-language pretraining.

### BLIP-2 (Salesforce, 2023)

Extends CLIP-style vision-language alignment to support image captioning and
visual question answering - not just retrieval:

```
Image → Q-Former (adapter) → LLM → text output
```

Enables multimodal RAG where retrieved images are described in text and fed to
an LLM for generation.

### Long-CLIP (2024)

Extends CLIP text encoder to 248 tokens (vs 77 in standard CLIP) while preserving
image encoder quality. Critical for retrieval over long captions or product
descriptions where 77 tokens is insufficient.

### Domain-specific CLIP models

```
BiomedCLIP:   medical images + biomedical text
RemoteCLIP:   satellite imagery + geographic descriptions
ArtCLIP:      artwork + art-historical descriptions
FashionCLIP:  fashion images + product descriptions
```

## Multimodal Hybrid Search

Combining CLIP-based image retrieval with BM25/dense text retrieval for richer
multimodal search:

```
Query: "red sports car high performance"

Retrieval:
  BM25/dense text retrieval   → finds text documents about red sports cars
  CLIP image retrieval        → finds images of red sports cars

Fusion (RRF):
  text_results + image_results → unified ranked list of text + image results

Application:
  E-commerce: query returns both product descriptions and product images
  News:       query returns both articles and photojournalism
  Medical:    query returns both case reports and diagnostic images
```

## CLIP in a Multimodal RAG Pipeline

CLIP-based image retrieval connects to the broader RAG architecture:

```
User query: "show me papers about attention mechanisms with diagrams"

Step 1 - Multimodal retrieval:
  Text search:   retrieve papers about attention mechanisms (BM25/dense)
  Image search:  retrieve diagrams about attention using CLIP
  Fusion:        RRF over text results + image results

Step 2 - Context assembly:
  Text passages: relevant paper excerpts
  Images:        attention diagrams (described by captioning model)
  Combined prompt: text + image descriptions

Step 3 - LLM generation:
  Generates answer grounded in both text papers and visual diagrams
```

The key bridge is a captioning model (BLIP-2, LLaVA) that converts retrieved
images into text descriptions the LLM can process. This is multimodal RAG -
retrieval over multiple modalities feeding a language model.

CLIP is the enabling technology for multimodal IR. Once you have shared
image-text embeddings, the rest of the retrieval pipeline - ANN indexing,
hybrid search, reranking - applies with minimal modification. The key new
concern is handling the different modalities gracefully throughout the pipeline.

## My Summary

CLIP learns a shared embedding space for images and text by training dual encoders
on 400 million image-text pairs with a contrastive objective. In this space, an
image and its caption produce nearby vectors enabling text-to-image, image-to-image,
and image-to-text retrieval without task-specific fine-tuning. The image encoder
(ViT-B/32 to ViT-H/14) and text encoder (GPT-2-style) are trained simultaneously
to maximize similarity for matching pairs and minimize it for non-matching pairs
using InfoNCE loss. Core limitations are the 77-token text context limit, weak
compositional reasoning, and poor fine-grained discrimination - addressed by
Long-CLIP, domain-specific variants, and patch-level embeddings respectively. For
IR practitioners, CLIP is the image-side component of multimodal hybrid search:
text queries retrieve both text documents (BM25/dense) and images (CLIP) with RRF
fusion, enabling richer retrieval over corpora containing both modalities.
