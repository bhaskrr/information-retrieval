# Multimodal Embeddings

Multimodal embeddings are dense vector representations that encode information
from multiple modalities - text, images, audio, video, and structured data -
into a single unified vector space. Unlike CLIP, which maintains separate image
and text encoders that happen to share an output space, true multimodal embeddings
aim for deeper fusion where information from different modalities interacts during
encoding rather than only at the similarity comparison stage. The goal is a single
embedding model that can represent any combination of modalities in one coherent
geometric space where semantic similarity across modalities, within modalities, and
between mixed-modality inputs is captured by vector proximity.

## Intuition

CLIP was the proof of concept: shared embedding spaces work. A text query and a
matching image can be brought close together in a high-dimensional space through
training. But CLIP has fundamental architectural limitations - its text and image
encoders never interact during encoding, only at the output layer via cosine
similarity. This late fusion means the text encoder never sees the image it is being
compared to, and the image encoder never sees the query text.

Consider the difference between these two retrieval scenarios:

```
Scenario A - CLIP-style late fusion:
  Encode query text → text_vector
  Encode image      → image_vector
  Compare: cosine_similarity(text_vector, image_vector)

Scenario B - Deep multimodal fusion:
  Encode query text + image jointly
  → the text and image tokens attend to each other
  → representations are mutually conditioned
  → relevance is assessed with full cross-modal context
```

Scenario B is more powerful but much more expensive. It requires running a joint
encoder for every (query, image) pair - the same tradeoff as cross-encoder vs
bi-encoder in text retrieval. The field is exploring architectures that sit between
these extremes: deeper than CLIP's late fusion, cheaper than full joint encoding.

Multimodal embeddings also extend beyond pairs. Modern multimodal embedding models
handle interleaved sequences of text and images - a document that contains a diagram
followed by a caption followed by another diagram - as a single input, producing
one embedding that captures the full mixed-modality document.

## The Modality Alignment Problem

The fundamental challenge in multimodal embeddings is aligning representations
across modalities that have very different statistical properties:

```
Text:  discrete tokens, sequential, arbitrary length, ~50K vocabulary
Image: continuous pixels, spatial, fixed grid, infinite variation
Audio: continuous waveform, temporal, variable length
Video: spatial + temporal, very high-dimensional
```

Three strategies for alignment:

### Strategy 1 - Projection to shared space (CLIP-style)

Each modality has its own encoder. A linear projection maps all encoder outputs
to a common dimension. Contrastive training aligns the projected spaces.

```
image → image_encoder → linear_proj → shared_space
text  → text_encoder  → linear_proj → shared_space
```

Simple, scalable, used in CLIP, ALIGN, SigLIP.
Limitation: no cross-modal interaction during encoding.

### Strategy 2 - Cross-attention fusion

Process each modality with its own encoder, then use cross-attention to let
modalities attend to each other:

```
image_tokens = image_encoder(image)
text_tokens  = text_encoder(text)

# Text tokens attend to image tokens (and vice versa)
fused = cross_attention(text_tokens, image_tokens)
```

Deeper interaction than CLIP but still modality-specific first stage.
Used in FLAVA, BridgeTower, VinVL.

### Strategy 3 - Unified sequence processing

Convert all modalities to token sequences and process with a single transformer:

```
image → patch_tokens → [concat with] → text_tokens → unified_transformer
audio → frame_tokens → [concat with]
video → frame_tokens → [concat with]
```

Maximum cross-modal interaction but computationally expensive.
Used in Flamingo, GPT-4V, LLaVA, Gemini.

## Key Multimodal Embedding Models

### ALIGN (Google, 2021)

Similar architecture to CLIP but trained on 1.8 billion noisy image-text pairs.
Demonstrates that scale can compensate for data quality in vision-language alignment.

```
Architecture: EfficientNet (image) + BERT (text) → shared space
Training:     1.8B image-text pairs from web (noisy, no curation)
Key finding:  scale > data quality for visual-language alignment
Embedding:    256-dim
```

### Florence (Microsoft, 2021)

Universal visual representation - single model covering multiple visual tasks:

```
Architecture: CoSwin (hierarchical image transformer) + text encoder
Trained on:   900M image-text pairs (LAION subset)
Outputs:      Image embeddings for retrieval, detection, segmentation, captioning
Key design:   Adapter heads for different tasks, shared trunk
```

### FLAVA (Meta, 2022)

Explicitly trains on three objectives - unimodal text, unimodal image, multimodal

- producing representations that work for both single-modality and cross-modal tasks:

```
Architecture: Image ViT + Text BERT + Multimodal encoder
Training objectives:
  Unimodal:    MLM on text, MIM on image (self-supervised)
  Multimodal:  Image-text matching, masked multimodal modeling
Advantage:    One model that excels at both unimodal and cross-modal tasks
```

### ImageBind (Meta, 2023)

Extends beyond image-text to six modalities in a single shared space:

```
Modalities: images, text, audio, video, depth, thermal, IMU (motion)
Key insight: bind everything to images
  image-text:    learn from internet image-caption pairs (like CLIP)
  image-audio:   learn from videos (image and audio co-occur)
  image-depth:   learn from RGBD data
  image-thermal: learn from paired RGB-thermal cameras
  ...
Result: all six modalities in one shared space
        → text can retrieve audio, audio can retrieve video, etc.
```

ImageBind's binding strategy is elegant: since images co-occur with every other
modality in natural data, training image-X pairs for each modality X creates a
hub-and-spoke alignment structure where all modalities are aligned through images.

### E5-Mistral and Gecko (2024)

Large language model-based text embeddings that serve as the text component
in modern multimodal systems:

```
E5-Mistral-7B:  7B parameter LLM fine-tuned for text embedding
                Strong zero-shot performance across MTEB benchmarks

Gecko (Google): Distilled from large language models
                Efficient, strong multilingual performance
```

These LLM-based text encoders are increasingly paired with vision encoders in
multimodal systems, replacing BERT-based text encoders.

### Nomic Embed Vision (2024)

Multimodal embedding model that produces compatible text and image embeddings
in the same space as Nomic Embed Text - enabling unified text and image search
with the same index:

```
Design:    CLIP-style but uses Nomic Embed Text as text encoder
           Same text embedding space as the text-only model
Benefit:   Single FAISS index for both text and image documents
           Queries can be either text or image, results can be either
```

### Voyage Multimodal (2024)

Commercial multimodal embedding API designed for document retrieval where
documents contain interleaved text and images:

```
Input:  interleaved sequences of text and images
Output: single embedding per document (not separate per modality)
Use:    PDF retrieval where pages contain both text and figures
```

## Interleaved Multimodal Documents

A critical real-world use case that CLIP cannot handle: documents containing
text and images mixed together in a single semantic unit.

```
Research paper page:
  [text: "Figure 3 shows the architecture of our proposed model"]
  [image: architecture diagram]
  [text: "The encoder consists of three transformer layers..."]

Product page:
  [image: product photo front view]
  [text: "Professional-grade noise cancelling headphones"]
  [image: product photo side view]
  [text: "40-hour battery life, USB-C charging"]
```

CLIP would encode the text and images separately - losing the semantic connection
between the caption and the diagram, between the product text and the product image.

Interleaved multimodal embeddings process the full sequence as a single input,
producing one embedding that captures the combined meaning.

## Late Interaction for Multimodal Retrieval

The bi-encoder vs cross-encoder tradeoff from text IR applies directly to multimodal
retrieval, with additional complexity:

```
Multimodal bi-encoder (CLIP-style):
  index_time:  encode each image/document independently → store vectors
  query_time:  encode query → ANN search
  latency:     fast (precomputed document vectors)
  quality:     limited (no query-document interaction)

Multimodal cross-encoder (BLIP-2-style):
  index_time:  none (cannot precompute)
  query_time:  jointly encode query + document → relevance score
  latency:     slow (full model per candidate)
  quality:     high (full cross-modal attention)

Multimodal late interaction (ColBERT-style):
  index_time:  store multiple vectors per document (one per patch/token)
  query_time:  MaxSim over query tokens × document patch vectors
  latency:     medium (precomputed but larger index)
  quality:     medium-high (token-level matching)
```

Late interaction is an active research area for multimodal retrieval - models
like ColPali use document patch embeddings with MaxSim scoring to achieve better
quality than CLIP with manageable index size.

## ColPali - Late Interaction for Document Images

ColPali (2024) applies the ColBERT late interaction idea to visual document
retrieval. Instead of encoding a document page as one vector, it encodes each
visual patch independently:

```
Document page → PaliGemma (vision-language model) → patch embeddings
               (one 128-dim embedding per 16×16 patch)
               → 1024 embeddings for a 1024×1024 page

Query → text encoder → token embeddings

Score = Σᵢ max_j (query_token_i · document_patch_j)
      ← MaxSim over query tokens and document patches
```

This captures fine-grained spatial-semantic matching - a query about "the graph
in the upper right" can match against the specific patch region containing that
graph, not the global page embedding.

## Evaluation Benchmarks for Multimodal Embeddings

### MSCOCO Retrieval

Standard image-text retrieval benchmark from Microsoft COCO dataset:

```
Task:    text-to-image and image-to-text retrieval
Corpus:  5,000 images with 5 captions each
Metric:  Recall@1, Recall@5, Recall@10
```

### Flickr30K

Smaller, cleaner image-text retrieval benchmark:

```
Corpus:  31,000 images with 5 captions each
Metric:  Recall@1, Recall@5, Recall@10
```

### MTEB Retrieval (multimodal extension)

MTEB is extending to cover multimodal retrieval tasks - the emerging standard
for comprehensive multimodal embedding evaluation.

### DocVQA

Visual question answering over document images - tests whether embeddings
capture document layout, tables, and figures:

```
Task:    answer questions about document page images
Format:  document image + question → answer
Tests:   understanding of charts, tables, forms, figures
```

## Fusion Strategy Selection Guide

| Scenario                             | Fusion strategy                      |
| ------------------------------------ | ------------------------------------ |
| Text is primary, image supplementary | weighted (image_weight=0.3)          |
| Image is primary, text is caption    | weighted (image_weight=0.7)          |
| Text and image equally important     | late_fusion (equal weight)           |
| Text and image semantically linked   | clip_text (use CLIP text encoder)    |
| No images in corpus                  | text_only (skip image encoding)      |
| No text in corpus                    | image_only (skip text encoding)      |
| Query is always text-only            | late_fusion or text_only for queries |
| Need maximum retrieval recall        | late_fusion (covers both modalities) |

## From Multimodal Embeddings to Multimodal RAG

Multimodal embeddings are the retrieval component of multimodal RAG:

```
Multimodal corpus (text + images)
    ↓
Multimodal embedding model
    → encode each document with appropriate fusion
    → store in unified vector index
    ↓
Retrieval (text or image query)
    → encode query with same model
    → ANN search over unified index
    → retrieve text docs + image docs
    ↓
Context assembly
    → text docs: include directly
    → image docs: describe with captioning model (BLIP-2, LLaVA)
    ↓
LLM generation
    → answer grounded in text + image content
```

## My Summary

Multimodal embeddings extend single-modality representation learning to unified
vector spaces that capture meaning across text, images, audio, video, and other
modalities. Three fusion strategies exist with different quality-efficiency tradeoffs:
late fusion (separate encoders, combined outputs) is scalable but limits cross-modal
interaction; cross-attention fusion allows deeper interaction at higher cost; unified
sequence processing (like LLaVA or Gemini) provides maximum fusion but requires
running a large model for every document. ImageBind generalizes this by aligning six
modalities through a shared image hub. In practice, most production multimodal
retrieval systems use CLIP-style late fusion because it enables precomputing document
embeddings offline - the bi-encoder advantage from text IR applies here too. The
choice of fusion strategy depends on whether text and image are equally important
(late fusion) or one modality dominates (weighted fusion), and on whether the corpus
is text-heavy, image-heavy, or truly mixed. ColPali applies the ColBERT late
interaction idea to visual documents, storing per-patch embeddings for finer-grained
matching - an emerging direction that bridges CLIP-style retrieval and cross-encoder
quality.
