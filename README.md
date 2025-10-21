# Enhance-SERP-based-Search-with-NER-driven-Intent-and-Entity-Aware-Query-Optimization

An advanced intelligent search system that combines **Named Entity Recognition (NER)**, **semantic intent detection**, and **LLM-powered query rephrasing** to perform **context-aware searches** using the **SerpAPI**.  

This application dynamically determines whether to perform **local map searches (Google Maps)** or **news/general searches (Google Search)** based on the user’s query, location, and temporal expressions — producing **concise, entity-preserving queries** and **summarized digests**.

---

## 🚀 Features

✅ **Intent-aware search**
- Classifies user intent (`GetNews`, `FindLocation`, `GetWeather`, `GetProductPrice`) via sentence-transformer embeddings.  

✅ **Entity-driven reasoning (NER)**
- Detects **location**, **date**, **organization**, and **product** entities to refine query context.

✅ **Temporal inference**
- Extracts and normalizes time expressions (e.g., “today”, “last month”, “2024”) for SerpAPI’s `tbs` filters.

✅ **LLM-based query rewriting**
- Uses **LLaMA (via Ollama)** to generate single-line, concise, keyword-rich search queries.

✅ **Automatic geolocation**
- Uses browser GPS or IP geolocation; falls back to NER-based location geocoding if unavailable.

✅ **Article summarization**
- Extracts and summarizes article content using LLMs with relevance weighting and entropy-based length control.

✅ **Multi-layer text extraction**
- **Trafilatura → BeautifulSoup → Playwright** pipeline ensures extraction even from Cloudflare-protected pages.

✅ **Performance-optimized backend**
- Uses model caching (`@lru_cache`), parallel text extraction (`ThreadPoolExecutor`), and lazy model loading.

---

## 🧩 Architecture Overview

```mermaid
graph TD
A[User Query] --> B[NER + Entity Extraction]
B --> C[Intent Detection by MiniLM Embeddings]
C --> D[Temporal Context Inference]
D --> E[LLM Query Optimization]
E --> F[SerpAPI Search Engine]
F --> G{Search Type?}
G -->|Maps| H[geo-aware with Google Maps API]
G -->|News/General| I[Google Search API]
H --> J[Digest Summarization]
I --> J
J --> K[Frontend UI Display]
