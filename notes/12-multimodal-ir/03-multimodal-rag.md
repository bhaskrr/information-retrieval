# Multimodal RAG

Multimodal RAG (Retrieval-Augmented Generation) extends the standard text-only RAG
pipeline to handle corpora that contain images, figures, charts, tables, and other
non-textual content alongside or instead of text. Instead of retrieving only text
passages and feeding them to a language model, a multimodal RAG system retrieves
content across modalities - finding relevant images, diagrams, and text together -
and assembles them into a context that a multimodal language model can reason over
to produce grounded, accurate answers. It is the integration point for everything
covered in the multimodal IR module: CLIP-based image retrieval, multimodal
embeddings, and generation from a vision-language model.

## Intuition

Standard RAG fails silently on multimodal content. Consider a corpus of technical
documentation that includes:

```
- Text explanations of algorithms
- Architecture diagrams as PNG files
- Performance comparison tables as images
- Mathematical notation rendered as figures
- Code screenshots
- Charts and graphs
```

A text-only RAG system indexes and retrieves only the text portions. Queries about
"the architecture diagram for BERT" or "the performance table comparing models"
return nothing or irrelevant results because the relevant content exists as images
that were never indexed. The system appears to work - it returns results - but
silently misses a large fraction of the corpus.

Multimodal RAG solves this by treating images as first-class retrievable objects.
The relevant diagram is retrieved by its visual content and semantic similarity to
the query. It is then described by a vision-language model so the generation LLM
can reason about what the diagram shows. The final answer is grounded in both the
textual and visual content of the corpus.

## The Gap Between Standard and Multimodal RAG

```
Standard RAG pipeline:
  Corpus:    text documents
  Indexing:  text → embeddings → vector index
  Retrieval: text query → retrieve text passages
  Context:   text passages
  Generation: LLM reads text context → generates answer

Multimodal RAG pipeline:
  Corpus:    text + images + mixed documents
  Indexing:  text → text embeddings
             images → CLIP/multimodal embeddings
             mixed pages → multimodal embeddings
  Retrieval: query → retrieve text + images + mixed
  Context:   text passages + image descriptions + table extractions
  Generation: VLM reads multimodal context → generates answer
```

The key additions are: multimodal indexing, cross-modal retrieval, visual content
processing (captioning, OCR, table extraction), and a vision-language model capable
of reasoning over images in the context.

## Multimodal RAG Architectures

Three main architectural patterns for multimodal RAG, ordered by sophistication:

### Architecture 1 - Caption-then-Index

Convert all images to text before indexing. Process images offline with a captioning
model, store the captions, index only text (including captions).

```
Offline:
  for each image in corpus:
    caption = captioning_model(image)   ← BLIP-2, LLaVA, etc.
    store (image_id, image_path, caption)
    index caption as text document

Online:
  standard text retrieval → text passages + image captions
  assemble context: text + captions
  standard LLM generation (no vision model needed)
```

**Advantages:** simple, no vision model at query time, any text LLM works
**Disadvantages:** captions lose information (table values, code content, fine
details), captions are generated offline and cannot adapt to the query

### Architecture 2 - Retrieve-then-Describe

Index images using visual embeddings (CLIP). Retrieve images by visual similarity.
At query time, describe retrieved images using a vision model.

```
Offline:
  encode all images → CLIP embeddings → FAISS index

Online:
  step 1: retrieve images by CLIP similarity to query
  step 2: describe retrieved images with vision model
          description = vlm(image, query)   ← query-conditioned!
  step 3: assemble: text passages + image descriptions
  step 4: LLM generation over assembled context
```

**Advantages:** query-conditioned descriptions are more relevant, images retrieved
visually (not just by caption text), no offline captioning required
**Disadvantages:** vision model runs at query time (adds latency), more complex

### Architecture 3 - Native Multimodal Context

Feed images directly to a vision-language model alongside text - no intermediate
captioning step.

```
Offline:
  encode images → CLIP/multimodal embeddings → index

Online:
  step 1: retrieve relevant images + text passages
  step 2: assemble multimodal prompt:
          {system prompt}
          {retrieved text passage 1}
          [retrieved image 1]
          {retrieved text passage 2}
          [retrieved image 2]
          {query}
  step 3: VLM processes full multimodal prompt → generates answer
```

**Advantages:** no information loss from captioning, images and text can interact
during generation, highest quality answers
**Disadvantages:** requires a capable VLM (GPT-4V, Claude 3.5 Sonnet, Gemini),
more expensive, image tokens in context are costly

## Document Processing for Multimodal RAG

Before indexing, documents must be parsed into their multimodal components:

### PDF processing

PDFs are the most common document format containing mixed text and images:

```
PDF page → extraction:
  text blocks:  positions, font sizes, content
  images:       embedded PNG/JPEG extracted from PDF
  tables:       detected via layout analysis
  figures:      detected via caption matching ("Figure 3...")
```

Tools:

```
pymupdf (fitz):    fast, good text + image extraction
pdfplumber:        excellent for table extraction
unstructured:      ML-based document parsing, handles complex layouts
marker:            PDF → markdown with figure descriptions
docling (IBM):     high-quality multimodal document parsing
```

### Image-heavy document handling

Some documents are primarily visual - scanned PDFs, presentation slides,
diagrams. Two approaches:

**OCR + visual embedding**
Run OCR (Tesseract, AWS Textract, Google Document AI) to extract text, then
embed both the extracted text and the original image:

```
Scanned page → OCR → extracted text → text embedding
             → CLIP → visual embedding
Store both embeddings for the same document chunk
```

**Page-as-image**
Treat entire document pages as images. Embed with CLIP or a document-aware
vision model (ColPali, nomic-embed-vision):

```
PDF page → render as PNG → CLIP/ColPali embedding → index
No text extraction needed - visual embedding captures all content
```

The page-as-image approach is simpler and handles cases where OCR fails
(handwriting, complex layouts, mathematical notation).

### Table extraction

Tables in documents deserve special handling because their structure carries
information that a visual description often loses:

```
Table image → table extraction model → structured data:
  | Model   | NDCG@10 | Recall@100 |
  |---------|---------|------------|
  | BM25    | 0.428   | 0.821      |
  | Dense   | 0.451   | 0.849      |
  | Hybrid  | 0.478   | 0.871      |
```

This structured representation can be directly embedded as text or processed
by the LLM as a table rather than describing it visually.

## Query Processing in Multimodal RAG

Queries in multimodal RAG may themselves be multimodal:

```
Text query:         "how does the attention mechanism work?"
                    → retrieve text explanations + architecture diagrams

Image query:        [user uploads a diagram]
                    "what does this diagram represent?"
                    → find similar diagrams + related text

Multimodal query:   "in this chart [image], which model performs best?"
                    → find similar charts + text discussing the results
```

For mixed queries, the query embedding must also be multimodal:

```
text_query_emb  = text_encoder(query_text)
image_query_emb = image_encoder(query_image)
query_emb       = fuse(text_query_emb, image_query_emb)
→ search multimodal index with fused query
```

## Context Assembly for Multimodal Generation

Assembling a multimodal context for the VLM requires careful ordering and
representation of different content types:

### Ordering retrieved content

```
Recommended ordering:
  1. System instructions (how to use retrieved context)
  2. Relevant text passages (most semantically relevant first)
  3. Retrieved images with captions (most visually similar first)
  4. Tables as structured text (preserve structure)
  5. User query
```

### Token budget management

Images consume many tokens in VLM context windows:

```
GPT-4V:   low resolution image ≈ 85 tokens
          high resolution image ≈ 170-1105 tokens (tile-based)
Claude:   image ≈ ~1500-2000 tokens typically
Gemini:   image ≈ 258 tokens (fixed)
```

With a 100K token budget and 5 retrieved images at 500 tokens each, images
consume 2,500 tokens - about 2.5% of the budget. For longer contexts, image
count becomes the binding constraint.

Strategies for token budget management:

- Reduce image resolution before including in context
- Use captioning for images below a relevance threshold
- Only include full images for the top-1 or top-2 most relevant visual results
- Use text descriptions for tables, code screenshots (OCR first)

## Evaluation of Multimodal RAG

Multimodal RAG evaluation is more complex than text RAG because both retrieval
and generation have visual components:

### Retrieval evaluation

```
Text retrieval:   standard NDCG@K, Recall@K
Image retrieval:  Recall@K for relevant images
Combined:         weighted combination of text and image recall
```

### Generation evaluation

```
Faithfulness:     does the answer accurately reflect the retrieved content?
                  (including visual content - does it describe the image correctly?)
Completeness:     does the answer address aspects requiring visual understanding?
Hallucination:    does the model invent visual details not in retrieved images?
```

### Benchmarks

```
DocVQA:     visual question answering over document images
SlideVQA:   QA over presentation slides with mixed text and charts
ChartQA:    QA specifically requiring chart understanding
MP-DocVQA:  multi-page document QA with interleaved text and figures
MMTE:       multimodal text extraction benchmark
```

## Architecture Comparison

| Architecture              | Latency | Quality | Complexity | VLM at query?      |
| ------------------------- | ------- | ------- | ---------- | ------------------ |
| Caption-then-index        | Fast    | Lower   | Low        | No                 |
| Retrieve-then-describe    | Medium  | Medium  | Medium     | Yes (per image)    |
| Native multimodal context | Slow    | Highest | High       | Yes (full context) |

For most applications, retrieve-then-describe with query-conditioned captions
provides the best balance - images are retrieved by visual similarity, described
with query relevance in mind, and the standard text LLM can process the descriptions.

## Production Considerations

### Offline preprocessing cost

Generating captions or descriptions for large image corpora is expensive. At scale:

```
1M images × 500ms per caption × $0.001 per API call = ~$1000 + 6 days
```

Strategies:

- Caption only images above a size/quality threshold
- Use a smaller local captioning model (BLIP-2-opt-2.7B) for bulk captioning,
  reserve GPT-4V for high-value images
- Cache captions and regenerate only when documents are updated

### Image deduplication

Corpora often contain duplicate or near-duplicate images. Index only unique images:

```
For each new image:
  hash = perceptual_hash(image)   ← pHash, dHash, or average hash
  if hash not in existing_hashes:
    index image
  else:
    link to existing indexed image
```

### Multi-resolution indexing

Store images at multiple resolutions for different use cases:

```
Thumbnail (64px):  for fast visual search
Medium (224px):    for CLIP embedding
Full resolution:   for VLM context when retrieved
```

### Freshness and updates

When source documents are updated, image indexes must be refreshed:

```
Document update → extract new images → compute new embeddings
               → delete old embeddings → insert new embeddings
               → update caption store
```

Multimodal RAG is the culmination of the multimodal IR module. It combines
CLIP-based image retrieval, multimodal embedding strategies, and vision-language
generation into a system that can answer questions over any combination of text
and visual content - the practical application of everything covered in this module.

## My Summary

Multimodal RAG extends standard text RAG to corpora containing images, diagrams,
charts, and mixed-modality documents. Three architectures exist: caption-then-index
(convert images to captions offline, use standard text retrieval), retrieve-then-
describe (retrieve images by CLIP visual similarity, generate query-conditioned
descriptions at query time), and native multimodal context (feed images directly to
a VLM, highest quality but most expensive). Document processing is a critical
upstream concern - PDFs must be parsed to extract images, tables, and figures
as separate indexable chunks. Context assembly requires managing token budgets since
images consume many tokens in VLM context windows. The key engineering insight is
that the same bi-encoder/cross-encoder tradeoff from text IR applies here: use fast
CLIP retrieval as the first stage and VLM description/reasoning as the second stage.
Production deployments must handle offline captioning costs, image deduplication,
multi-resolution storage, and index freshness when source documents change.
