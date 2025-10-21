import requests, re, json, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from serpapi.google_search import GoogleSearch
import trafilatura
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import spacy

# =====================================================
# 🔧 CONFIG
# =====================================================
SERPAPI_API_KEY = "<Your SerpAPI API key>"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# =====================================================
# ⚙️ Lazy Model Loading + Caching
# =====================================================
from functools import lru_cache

@lru_cache(maxsize=1)
def get_nlp():
    print("🧠 Loading spaCy model (light mode)...")
    return spacy.load("en_core_web_sm", disable=["parser", "textcat"])

@lru_cache(maxsize=1)
def get_embedder():
    print("⚙️ Loading SentenceTransformer (MiniLM-L3)...")
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

nlp = get_nlp()
embedder = get_embedder()
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})

# =====================================================
# 🧩 Intent Detection (Precomputed Embeddings)
# =====================================================
INTENT_PATTERNS = {
    "GetWeather": ["weather", "temperature", "rain", "forecast", "climate", "humidity"],
    "FindLocation": ["near me", "nearby", "around me", "in my area", "location", "map", "place", "area"],
    "GetProductPrice": ["price", "cost", "buy", "purchase", "product", "rate", "deal"],
    "GetNews": ["news", "headline", "update", "current", "trending", "latest"],
}
SIMILARITY_THRESHOLD = 0.55
INTENT_EMBEDDINGS = {intent: embedder.encode(words, convert_to_tensor=True) for intent, words in INTENT_PATTERNS.items()}

def detect_intent_lemmatized(text):
    """
    Detects the user's query intent semantically (e.g., News, Location, Product, Weather).
    Uses spaCy + sentence-transformer embeddings for similarity scoring.
    """
    doc = nlp(text.lower())
    lemmas = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
    lemma_vecs = embedder.encode(" ".join(lemmas), convert_to_tensor=True)
    best_intent, best_score = "Unknown", 0.0

    for intent, kw_vecs in INTENT_EMBEDDINGS.items():
        score = float(util.cos_sim(lemma_vecs, kw_vecs).max())
        if score > best_score:
            best_score, best_intent = score, intent

    # Heuristic for geographic content
    if any(ent.label_ in ["GPE", "LOC"] for ent in doc.ents) and any(
        w in text.lower() for w in ["restaurant", "hotel", "atm", "bank", "shop", "cafe", "mall"]
    ):
        best_intent, best_score = "FindLocation", max(best_score, 0.6)

    return (best_intent if best_score >= SIMILARITY_THRESHOLD else "Unknown"), round(best_score, 3)

def analyze_query(text):
    doc = nlp(text)
    intent, score = detect_intent_lemmatized(text)
    entities = {ent.label_: ent.text for ent in doc.ents}

    # Extract useful entity types
    geo_entities = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    date_entities = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    product_entities = [ent.text for ent in doc.ents if ent.label_ in ("PRODUCT", "ORG")]

    return {
        "query": text,
        "intent": intent,
        "similarity_score": score,
        "entities": entities,
        "geo_entities": geo_entities,
        "date_entities": date_entities,
        "product_entities": product_entities
    }


# =====================================================
# 🕒 Temporal Context Detection (Precomputed)
# =====================================================
TIME_CATEGORIES = {
    "today": ["today", "now", "currently", "as of now", "right now", "this moment"],
    "recent": ["latest", "recent", "newest", "fresh", "upcoming", "current"],
    "yesterday": ["yesterday", "previous day", "day before"],
    "week": ["this week", "past week", "last week", "recent days"],
    "month": ["this month", "past month", "last month"],
    "year": ["this year", "annual", "yearly", "fiscal year"],
    "historical": ["history", "ancient", "long ago", "decade ago", "in the past"],
}
TIME_EMBEDDINGS = {k: embedder.encode(v, convert_to_tensor=True) for k, v in TIME_CATEGORIES.items()}

def infer_temporal_context(query_text: str):
    now = datetime.now()
    current_year = now.year
    doc = nlp(query_text.lower())
    lemmas = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
    lemma_vecs = embedder.encode(lemmas, convert_to_tensor=True)

    best_label, best_score = None, 0.0
    for label, kw_vecs in TIME_EMBEDDINGS.items():
        score = float(util.cos_sim(lemma_vecs, kw_vecs).max())
        if score > best_score:
            best_score, best_label = score, label

    # Numeric year detection
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query_text)
    if year_match:
        year_val = int(year_match.group(1))
        best_label = "historical" if year_val < current_year - 1 else "year"
        best_score = max(best_score, 0.8)

    # ✅ Relaxed threshold for modern keywords
    TIME_THRESH = 0.55

    if best_score < TIME_THRESH:
        # Check for soft matches manually
        if any(k in query_text.lower() for k in ["latest", "recent", "new", "upcoming", "current"]):
            best_label = "recent"
        else:
            best_label = "week"  # default fallback

    # 🎯 Improved mapping
    if best_label in ["today", "recent"]:
        return ("qdr:d,sbd:1", now.strftime("%B %d, %Y"))
    elif best_label == "yesterday":
        return ("qdr:d,sbd:1", (now - timedelta(days=1)).strftime("%B %d, %Y"))
    elif best_label == "week":
        return ("qdr:w,sbd:1", f"week of {(now - timedelta(days=7)).strftime('%B %d, %Y')}")
    elif best_label == "month":
        return ("qdr:m,sbd:1", now.strftime("%B %Y"))
    elif best_label == "year":
        return ("qdr:y", str(current_year))
    elif best_label == "historical":
        return ("", f"before {current_year - 1}")
    return ("qdr:w,sbd:1", now.strftime("%B %Y"))

# Precomputed embeddings (once per session)
NEARBY_PHRASES = [
    "near me", "around me", "close to me", "in my area", "nearby", 
    "near this location", "around here", "local", "my place", "vicinity"
]
NEARBY_EMBEDDINGS = embedder.encode(NEARBY_PHRASES, convert_to_tensor=True)

@lru_cache(maxsize=512)
def get_query_embedding(text):
    doc = nlp(text)
    lemmas = " ".join([t.lemma_ for t in doc if not t.is_stop])
    return embedder.encode(lemmas, convert_to_tensor=True)

def detect_geo_intent(query):
    """
    Semantic + literal hybrid detection for 'near me' context.
    Returns True if query is location-aware.
    """
    q_lower = query.lower()
    # quick literal check (fast path)
    if any(term in q_lower for term in NEARBY_PHRASES):
        return True
    
    # fallback: semantic similarity
    q_emb = get_query_embedding(q_lower)
    sim = float(util.cos_sim(q_emb, NEARBY_EMBEDDINGS).max())
    return sim > 0.65

# =====================================================
# 🧠 Query Optimization with Llama
# =====================================================
def rephrase_query_with_llama(user_query: str):
    today = datetime.now().strftime("%B %d, %Y")
    _, normalized_time = infer_temporal_context(user_query)
    normalized_query = f"{user_query} ({normalized_time})"
    print(f"🔍 Normalized Query: {normalized_query}")

    prompt = (
        f"Today's date is {today}.\n"
        "You are a professional search query optimizer.\n"
        "Rewrite the user's request into a single, concise, keyword-rich query for Google search.\n"
        "❌ Do NOT include explanations, notes, examples, or extra text.\n"
        "✅ Output only the rewritten query, on a single line, nothing else.\n\n"
        f"User query: {normalized_query}\n\n"
        "Output:\n"
    )

    try:
        with session.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            stream=True,
            timeout=30,
        ) as resp:
            resp.raise_for_status()
            text = "".join(json.loads(line.decode())["response"] for line in resp.iter_lines() if line)
        # Post-cleanup in case the model still sneaks in extra text
        cleaned = text.strip().split("\n")[0]
        cleaned = re.sub(r'^(["“”]+|[A-Za-z ]*[:：]\s*)', '', cleaned).strip()
        return cleaned
    except Exception as e:
        print(f"❌ Llama failed: {e}")
        return user_query.strip()

# =====================================================
# 🌐 Article Extraction (Multi-layered)
# =====================================================
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

def fetch_with_playwright(url):
    if not PLAYWRIGHT_AVAILABLE:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=20000)
            text = page.inner_text("body")
            browser.close()
            return text[:20000]
    except Exception as e:
        print(f"⚠️ Playwright failed: {e}")
        return None

def extract_article_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.split()) > 50:
                return text.strip()
    except Exception:
        pass

    try:
        html = session.get(url, timeout=10)
        if "cloudflare" in html.text.lower():
            return fetch_with_playwright(url)
        soup = BeautifulSoup(html.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 60]
        return "\n".join(paragraphs[:10]) if paragraphs else None
    except Exception:
        return fetch_with_playwright(url) if PLAYWRIGHT_AVAILABLE else None

# =====================================================
# 📰 Summarization + Serp Search
# =====================================================
def summarize_with_llama(text):
    try:
        with session.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text}, stream=True, timeout=30) as r:
            r.raise_for_status()
            return "".join(json.loads(line.decode())["response"] for line in r.iter_lines() if line).strip()
    except Exception:
        return None

import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util

# make sure nltk resources are downloaded (run once in setup)
# nltk.download('punkt')
# nltk.download('stopwords')

def content_entropy(texts):
    """
    Measures lexical diversity (entropy) across a list of summaries.
    High entropy → diverse topics → longer digest.
    """
    tokens = [w.lower() for t in texts for w in t.split() if w.isalpha()]
    if not tokens:
        return 0
    counts = Counter(tokens)
    probs = np.array(list(counts.values())) / sum(counts.values())
    return -np.sum(probs * np.log2(probs))


def make_news_digest(articles, user_query):
    """
    Generates a dynamic, content-aware news digest.
    Digest length is determined by:
      - number of articles,
      - average semantic similarity to the query, and
      - lexical entropy (diversity) of content.
    """
    summaries, scores = [], []
    print(f"📰 Fetched {len(articles)} articles.")

    with ThreadPoolExecutor(max_workers=4) as pool:
        texts = list(pool.map(lambda a: extract_article_text(a["link"]), articles))

    # Compute similarity between query and each article
    query_emb = embedder.encode(user_query, convert_to_tensor=True)

    for art, content in zip(articles, texts):
        if not content:
            continue

        # Compute similarity
        art_emb = embedder.encode(content[:1000], convert_to_tensor=True)
        sim = float(util.cos_sim(query_emb, art_emb))
        scores.append(sim)

        # Individual summarization
        prompt = (
            f"Summarize this article in 3–5 sentences, focusing on: '{user_query}'.\n\n"
            f"Title: {art['title']}\n\n{content[:2000]}"
        )
        summary = summarize_with_llama(prompt)
        summaries.append(f"• {art['title']}\n{summary}\n")

    # --- DYNAMIC LENGTH LOGIC ---
    avg_sim = sum(scores) / len(scores) if scores else 0.5
    entropy = content_entropy(summaries)
    print(f"📊 Avg similarity: {avg_sim:.2f}, Entropy: {entropy:.2f}")

    # baseline length based on relevance
    if avg_sim > 0.75:
        digest_len = 90
    elif avg_sim > 0.6:
        digest_len = 120
    else:
        digest_len = 160

    # adjust by number of articles
    digest_len += int(len(articles) * 5)

    # adjust by content diversity (entropy)
    if entropy > 7.5:
        digest_len += 40  # more diverse = longer
    elif entropy < 5.0:
        digest_len -= 20  # repetitive = shorter

    digest_len = max(80, min(digest_len, 250))  # clamp to 80–250 range
    print(f"🧮 Final dynamic digest length: ~{digest_len} words")

    # --- FINAL DIGEST GENERATION ---
    final_prompt = (
        f"Combine the following {len(summaries)} summaries into one unified, concise digest "
        f"of roughly {digest_len} words. "
        "Do not mention the word count or describe the summary itself. "
        f"Focus only on factual information related to '{user_query}'. "
        "Write directly, without introductions like 'Here is a summary' or 'In this digest'.\n\n"
        f"{''.join(summaries)}"
    )

    final = summarize_with_llama(final_prompt) or "⚠️ Digest unavailable."
    return final


def serp_search(user_query: str, lat: float | None, lon: float | None, num_results=5):
    """
    Unified search logic combining:
    - Semantic intent detection
    - Temporal context inference
    - NER-based location detection
    - Browser geolocation (lat/lon)
    """
    # ----------------------------------------------------
    # 🔍 STEP 1 — Analyze Query (Intent + Entities + NER)
    # ----------------------------------------------------
    analysis = analyze_query(user_query)
    intent = analysis["intent"]
    entities = analysis["entities"]
    geo_entities = analysis.get("geo_entities", [])
    date_entities = analysis.get("date_entities", [])

    print(f"🧭 Intent: {intent} (score={analysis['similarity_score']})")
    print(f"📘 Entities: {entities}")

    # Default setup
    engine = "google"
    refined_query = user_query.strip()
    print(f"Refined query: {refined_query}")

    # ----------------------------------------------------
    # 🌍 STEP 2 — GEO INTENT DETECTION (NER + Semantic)
    # ----------------------------------------------------
    geo_intent = detect_geo_intent(user_query)
    valid_coords = lat is not None and lon is not None
    if app.debug:
        print(f"[DEBUG] geo_intent={geo_intent}, valid_coords={valid_coords}")


    # ✅ Try to geocode NER-detected location if no browser coords
    if not valid_coords and geo_entities:
        location_name = geo_entities[0]
        coords = geocode_location(location_name)
        if coords:
            lat, lon = coords
            valid_coords = True
            print(f"📍 Using NER-detected location: {location_name} → {lat}, {lon}")
        else:
            print(f"⚠️ Could not geocode NER location: {location_name}")

    # ----------------------------------------------------
    # 🧠 STEP 3 — TEMPORAL CONTEXT + QUERY REPHRASING
    # ----------------------------------------------------
    if intent == "FindLocation" or geo_intent:
        # Skip rephrasing for location-based queries to preserve terms like "near me"
        engine = "google_maps"
        print("⚠️ Skipped rephrase for geo-intent query (preserving 'near me').")
    else:
        # Prefer NER-detected DATE/TIME entities if available
        if date_entities:
            entity_text = " ".join(date_entities)
            tbs_value, normalized_time = infer_temporal_context(entity_text)
            print(f"📅 NER-based temporal context detected: {entity_text}")
        else:
            # Fall back to free-form temporal inference from query
            tbs_value, normalized_time = infer_temporal_context(user_query)
            print(f"📅 Context inferred from query: {normalized_time}")

        # Rephrase the query while preserving entities and normalized time reference
        rephrase_input = f"{user_query} ({normalized_time})" if normalized_time else user_query
        refined_query = rephrase_query_with_llama(rephrase_input)

        print(f"🕒 Normalized time: {normalized_time}")
        print(f"✨ Rephrased Query: {refined_query}")


    # ----------------------------------------------------
    # 🔧 STEP 4 — SERPAPI PARAMETERS
    # ----------------------------------------------------
    params = {
        "engine": engine,
        "q": refined_query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "in",
        "num": num_results,
    }

    # ✅ Correct map handling for coordinate-based search
    if engine == "google_maps":
        if valid_coords:
            params["ll"] = f"@{round(lat,6)},{round(lon,6)},15z"
            params["type"] = "search"
            print(f"📍 Performing location-aware search at: ({lat}, {lon})")
        else:
            params["location"] = "India"  # Fallback country
            print("🌐 No lat/lon — using fallback location: India")
    elif engine == "google" and intent == "GetNews":
        params["tbm"] = "nws"

    # ----------------------------------------------------
    # 🔍 STEP 5 — EXECUTE SERPAPI SEARCH
    # ----------------------------------------------------
    try:
        res = GoogleSearch(params).get_dict()
    except Exception as e:
        print(f"❌ SerpAPI request failed: {e}")
        return {"mode": "error", "items": [], "debug": params}

    # ----------------------------------------------------
    # 📍 STEP 6 — MAP RESULTS
    # ----------------------------------------------------
    if engine == "google_maps":
        results = res.get("local_results") or res.get("place_results") or []
        if isinstance(results, dict):
            results = [results]
        print(f"📍 Found {len(results)} local places.")
        return {"mode": "maps", "items": results[:num_results], "debug": params}

    # ----------------------------------------------------
    # 📰 STEP 7 — NEWS / GENERAL RESULTS
    # ----------------------------------------------------
    results = res.get("news_results") or res.get("organic_results") or []
    articles = [{"title": a.get("title"), "link": a.get("link")} for a in results[:num_results]]
    print(f"📰 Fetched {len(articles)} articles.")
    return {"mode": "news", "items": articles, "debug": params}


def geocode_location(place_name):
    """Convert named location to lat/lon using Google Maps API."""
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": place_name, "key": "YOUR_GOOGLE_API_KEY"}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            print(f"📍 Geocoded '{place_name}' → ({loc['lat']}, {loc['lng']})")
            return loc["lat"], loc["lng"]
        else:
            print(f"⚠️ Geocoding failed: {data.get('status')}")
            return None
    except Exception as e:
        print(f"❌ Geocoding exception for '{place_name}': {e}")
        return None

# =====================================================
# 🌍 Flask Routes
# =====================================================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    if filename.endswith((".js", ".css", ".png", ".jpg", ".ico")):
        return send_from_directory(".", filename)
    return send_from_directory(".", "index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "")
    lat, lon, acc = data.get("latitude"), data.get("longitude"), data.get("accuracy")

    if acc and isinstance(acc, (int, float)) and acc > 10000:
        lat, lon = None, None

    try:
        result = serp_search(query, lat, lon)
        if result["mode"] == "news":
            digest = make_news_digest(result["items"], query)
            return jsonify({"mode": "news", "digest": digest, "debug": result["debug"]})
        return jsonify({"mode": "maps", "places": result["items"], "debug": result["debug"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

