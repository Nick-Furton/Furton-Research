#!/usr/bin/env python3
"""
Furton Research — Local API Server v3
=======================================
Handles four jobs:
  1. Enrichment    — pulls live market data via web search (Haiku)
  2. Injection     — injects primary source corpus per investor
  3. Evaluation    — individual investor evaluation calls (Sonnet)
  4. Synthesis     — committee deliberation synthesis (Opus)

Endpoints:
    POST /enrich           — live market data enrichment
    POST /v1/messages      — individual investor evaluation
    POST /committee/blind  — run all five agents simultaneously
    POST /committee/deliberate — run deliberation round
    POST /committee/synthesize — Opus committee statement
    OPTIONS *              — CORS preflight
"""

import json
import os
import re
import http.server
import urllib.request
import urllib.error
import threading
import time
import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

PORT         = 8765
# API key is read from this file (a single line beginning with "sk-ant-").
# Override the location with the FURTON_API_KEY_FILE environment variable;
# it defaults to furton_api_key.txt in your home directory.
API_KEY_FILE = Path(os.environ.get("FURTON_API_KEY_FILE",
                                   str(Path.home() / "furton_api_key.txt")))

HAIKU_MODEL  = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL   = "claude-opus-4-6"

# Sampling temperature — pinned low for reproducibility of an "expert verdict".
# Sent explicitly on every generation payload so the value is documented and the
# screen is repeatable rather than relying on the (undocumented) model default.
# Recorded in each archived screen file (see archive_screen) and in the
# methodology paper (§3 engine / §5 economics).
TEMPERATURE  = 0.3

# Dated, append-only archive of full committee records (blind + deliberation
# verdicts, vote, statement) keyed by screen date. Lives in the project root —
# NOT under furton_website/ or docs/ — so it never deploys to the public site.
SCREENS_DIR  = Path(__file__).resolve().parent / "screens"
_archive_lock = threading.Lock()

# ── Pricing (USD per token) — standard API rates, June 2026 ─────────────────────
# Source: Anthropic pricing docs. Output is billed at 5x input across the lineup.
PRICING = {
    HAIKU_MODEL:  {"in": 1e-6,  "out": 5e-6},    # Haiku 4.5  : $1 / $5  per MTok
    SONNET_MODEL: {"in": 3e-6,  "out": 15e-6},   # Sonnet 4.6 : $3 / $15 per MTok
    OPUS_MODEL:   {"in": 5e-6,  "out": 25e-6},   # Opus 4.6   : $5 / $25 per MTok
}
CACHE_WRITE_MULT = 1.25     # writing to cache costs 1.25x the base input rate
CACHE_READ_MULT  = 0.10     # reading from cache costs 0.1x the base input rate
WEB_SEARCH_COST  = 0.01     # $10 per 1,000 web searches = $0.01 each

def stage_cost(model, in_tok, out_tok, cw=0, cr=0, searches=0):
    """USD cost of a stage given summed token counts.

    in_tok  = fresh (non-cached) input tokens
    out_tok = output tokens
    cw / cr = cache-write / cache-read input tokens
    searches = web_search tool invocations (Haiku enrich only)
    """
    p = PRICING.get(model, PRICING[SONNET_MODEL])
    return (in_tok * p["in"]
            + out_tok * p["out"]
            + cw * p["in"] * CACHE_WRITE_MULT
            + cr * p["in"] * CACHE_READ_MULT
            + searches * WEB_SEARCH_COST)

# Library paths
# Corpus sizes after the ~250-word re-chunk (mean ≈230 w, 40-word overlap). The
# count the server actually retrieves from is the post-filter count; the total
# manifest count is shown in parentheses where a high-doctrine filter applies:
#   Warren Buffett        451  (of 2,082)   high-doctrine filtered
#   Leopold Aschenbrenner 480                no filter
#   Howard Marks          742  (of 1,383)   high-doctrine filtered
#   Joel Greenblatt       793                all chunks qualify
#   Cathie Wood           733                all chunks qualify
BUFFETT_LIBRARY       = Path.home() / "Downloads" / "buffett_library"       / "manifest.json"
ASCHENBRENNER_LIBRARY = Path.home() / "aschenbrenner_library"               / "manifest.json"
MARKS_LIBRARY         = Path.home() / "marks_library"                       / "manifest.json"
GREENBLATT_LIBRARY    = Path.home() / "greenblatt_library"                  / "manifest.json"
WOOD_LIBRARY          = Path.home() / "wood_library"                        / "manifest.json"

# Committee weighted vote threshold (long-only — committee verdict is Buy or Pass)
BUY_THRESHOLD   =  0.3

# ── Dow 30 constituents (as of 2026-06-29) ─────────────────────────────────────
# Alphabet (GOOGL) replaced Verizon (VZ) in the index effective 2026-06-29.
# ai_relevant flags whether Aschenbrenner should be auto-included.
# High = core AI infrastructure/platform; Medium = material AI exposure.
DOW30 = [
    {"ticker": "AAPL", "name": "Apple Inc.",                    "ai_relevant": True,  "sector": "Technology"},
    {"ticker": "AMGN", "name": "Amgen Inc.",                    "ai_relevant": False, "sector": "Healthcare"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.",               "ai_relevant": True,  "sector": "Consumer Discretionary"},
    {"ticker": "AXP",  "name": "American Express Co.",          "ai_relevant": False, "sector": "Financials"},
    {"ticker": "BA",   "name": "Boeing Co.",                    "ai_relevant": False, "sector": "Industrials"},
    {"ticker": "CAT",  "name": "Caterpillar Inc.",              "ai_relevant": False, "sector": "Industrials"},
    {"ticker": "CRM",  "name": "Salesforce Inc.",               "ai_relevant": True,  "sector": "Technology"},
    {"ticker": "CSCO", "name": "Cisco Systems Inc.",            "ai_relevant": True,  "sector": "Technology"},
    {"ticker": "CVX",  "name": "Chevron Corp.",                 "ai_relevant": False, "sector": "Energy"},
    {"ticker": "DIS",  "name": "Walt Disney Co.",               "ai_relevant": False, "sector": "Communication Services"},
    {"ticker": "GOOGL","name": "Alphabet Inc.",                 "ai_relevant": True,  "sector": "Communication Services"},
    {"ticker": "GS",   "name": "Goldman Sachs Group Inc.",      "ai_relevant": False, "sector": "Financials"},
    {"ticker": "HD",   "name": "Home Depot Inc.",               "ai_relevant": False, "sector": "Consumer Discretionary"},
    {"ticker": "HON",  "name": "Honeywell International Inc.",   "ai_relevant": False, "sector": "Industrials"},
    {"ticker": "IBM",  "name": "International Business Machines","ai_relevant": True,  "sector": "Technology"},
    {"ticker": "JNJ",  "name": "Johnson & Johnson",             "ai_relevant": False, "sector": "Healthcare"},
    {"ticker": "JPM",  "name": "JPMorgan Chase & Co.",          "ai_relevant": False, "sector": "Financials"},
    {"ticker": "KO",   "name": "Coca-Cola Co.",                 "ai_relevant": False, "sector": "Consumer Staples"},
    {"ticker": "MCD",  "name": "McDonald's Corp.",              "ai_relevant": False, "sector": "Consumer Discretionary"},
    {"ticker": "MMM",  "name": "3M Co.",                        "ai_relevant": False, "sector": "Industrials"},
    {"ticker": "MRK",  "name": "Merck & Co. Inc.",              "ai_relevant": False, "sector": "Healthcare"},
    {"ticker": "MSFT", "name": "Microsoft Corp.",               "ai_relevant": True,  "sector": "Technology"},
    {"ticker": "NKE",  "name": "Nike Inc.",                     "ai_relevant": False, "sector": "Consumer Discretionary"},
    {"ticker": "NVDA", "name": "NVIDIA Corp.",                  "ai_relevant": True,  "sector": "Technology"},
    {"ticker": "PG",   "name": "Procter & Gamble Co.",          "ai_relevant": False, "sector": "Consumer Staples"},
    {"ticker": "SHW",  "name": "Sherwin-Williams Co.",          "ai_relevant": False, "sector": "Materials"},
    {"ticker": "TRV",  "name": "Travelers Cos. Inc.",           "ai_relevant": False, "sector": "Financials"},
    {"ticker": "UNH",  "name": "UnitedHealth Group Inc.",       "ai_relevant": False, "sector": "Healthcare"},
    {"ticker": "V",    "name": "Visa Inc.",                     "ai_relevant": True,  "sector": "Financials"},
    {"ticker": "WMT",  "name": "Walmart Inc.",                  "ai_relevant": False, "sector": "Consumer Staples"},
]

# ── Load API key ───────────────────────────────────────────────────────────────

def load_api_key():
    if not API_KEY_FILE.exists():
        print(f"\n✗ API key file not found at: {API_KEY_FILE}")
        exit(1)
    key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key.startswith("sk-ant-"):
        print(f"\n✗ API key format wrong. Found: {key[:12]}...")
        exit(1)
    return key

# ── Primary source context: RAG retrieval with even-sampling fallback ──────────

# RAG: retrieve the top-K most relevant chunks per stock instead of a fixed slice.
RAG_TOP_K = 20                 # chunks retrieved per investor per evaluation
                               # (raised from 10 after re-chunking to ~230-word
                               #  chunks: 20 × ~230w ≈ 4,600 words of precise,
                               #  full-text-matched corpus per investor per call)
MAX_CONTEXT_WORDS = 15_000     # used only by the even-sampling fallback

# Embedding model is loaded lazily on first retrieval (only if embeddings exist).
_EMBED_MODEL = None

def get_embed_model():
    """Lazy-load the embedding model. Returns None if unavailable."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBED_MODEL
    except Exception as e:
        print(f"  ⚠ Could not load embedding model: {e}")
        return None


class RAGStore:
    """Holds chunks + embeddings for one investor and retrieves top-K by brief."""

    def __init__(self, investor_name, chunks, vectors, chunk_ids):
        self.investor_name = investor_name
        self.chunks        = chunks               # list of chunk dicts
        self.vectors       = vectors              # np.ndarray (n, dim) normalized, or None
        # Map chunk_id -> chunk for ordering safety
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.chunk_ids = chunk_ids                # row order of vectors

    @property
    def has_embeddings(self):
        return self.vectors is not None

    def retrieve(self, brief, top_k=RAG_TOP_K):
        """Return top_k chunks most relevant to the brief (or even-sample fallback)."""
        if not self.has_embeddings:
            return self._even_sample(top_k)

        import numpy as np
        model = get_embed_model()
        if model is None:
            return self._even_sample(top_k)

        # BUG 2.2 fix: embed the compact RETRIEVAL QUERY emitted by enrichment
        # (or a bounded brief-head fallback), not the full brief — the embedder
        # silently truncates to ~190 words, so the full brief's tail (risks,
        # recent news) never influenced which passages were retrieved.
        query_text = extract_retrieval_query(brief)
        used_emitted = _RETRIEVAL_QUERY_RE.search(brief or "") is not None
        print(f"  [RAG] {self.investor_name}: embedding "
              f"{'RETRIEVAL QUERY' if used_emitted else 'brief-head fallback'} "
              f"({len(query_text.split())}w): {query_text[:90]}")
        q = model.encode([query_text], normalize_embeddings=True).astype(np.float32)[0]
        # Cosine similarity = dot product (vectors are normalized)
        sims = self.vectors @ q
        top_idx = np.argsort(-sims)[:top_k]

        selected = []
        for i in top_idx:
            cid = self.chunk_ids[int(i)]
            chunk = self.by_id.get(cid)
            if chunk:
                selected.append(chunk)
        return selected

    def _even_sample(self, top_k=RAG_TOP_K):
        """Fallback: evenly sample about top_k chunks across the corpus.

        Targets the same chunk count as RAG retrieval so cost and context
        size stay consistent whether or not embeddings are built.
        """
        chunks = self.chunks
        total = len(chunks)
        if total == 0:
            return []
        avg_words = sum(c.get("word_count", len(c["text"].split())) for c in chunks) / total
        budget_cap = max(1, int(MAX_CONTEXT_WORDS / avg_words))
        target = min(top_k, budget_cap, total)
        if total <= target:
            return chunks
        step = total / target
        return [chunks[int(i * step)] for i in range(target)]


def build_rag_store(manifest_path, investor_name, filter_fn=None):
    """Load chunks + embeddings (if present) into a RAGStore."""
    if not manifest_path.exists():
        print(f"  ⚠ Library not found: {manifest_path}")
        return None

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    chunks = manifest["chunks"]
    if filter_fn:
        filtered = [c for c in chunks if filter_fn(c)]
        chunks = filtered if filtered else chunks
    if not chunks:
        return None

    lib_dir   = manifest_path.parent
    emb_path  = lib_dir / "embeddings.npy"
    meta_path = lib_dir / "embeddings_meta.json"

    vectors   = None
    chunk_ids = [c["chunk_id"] for c in chunks]

    if emb_path.exists() and meta_path.exists():
        try:
            import numpy as np
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            all_vectors = np.load(emb_path)
            meta_ids    = meta["chunk_ids"]
            # Build id->row map from the FULL embedded set
            id_to_row = {cid: i for i, cid in enumerate(meta_ids)}
            # Keep only rows for chunks that survived the filter, in chunk order
            rows, kept_ids = [], []
            for c in chunks:
                cid = c["chunk_id"]
                if cid in id_to_row:
                    rows.append(id_to_row[cid])
                    kept_ids.append(cid)
            if rows:
                vectors   = all_vectors[rows]
                chunk_ids = kept_ids
                print(f"  ✓ {investor_name}: {len(chunks)} chunks + RAG embeddings ({vectors.shape[0]} vectors)")
            else:
                print(f"  ⚠ {investor_name}: embeddings present but no id match, using fallback")
        except Exception as e:
            print(f"  ⚠ {investor_name}: embedding load failed ({e}), using fallback")
    else:
        print(f"  ○ {investor_name}: {len(chunks)} chunks (no embeddings — even-sampling fallback)")

    return RAGStore(investor_name, chunks, vectors, chunk_ids)


def format_context(investor_name, chunks):
    """Format a list of chunks into the injected corpus string."""
    if not chunks:
        return ""
    lines = [f"## PRIMARY SOURCE CORPUS — {investor_name}\n\n"]
    lines.append("Reason exclusively from what is documented in these primary sources. "
                 "These passages were selected as the most relevant to the investment under review.\n\n---\n\n")
    for chunk in chunks:
        lines.append(f"[{chunk['citation']}]\n\n{chunk['text']}\n\n---\n\n")
    return "".join(lines)


def retrieve_context(investor_name, brief, stores):
    """Retrieve and format the most relevant context for an investor + brief."""
    store = stores.get(investor_name)
    if not store:
        return ""
    chunks = store.retrieve(brief, top_k=RAG_TOP_K)
    return format_context(investor_name, chunks)

# ── Anthropic API call ─────────────────────────────────────────────────────────

def call_anthropic(payload, timeout=120, max_retries=3):
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), 200
        except urllib.error.HTTPError as e:
            # 429 (rate limit) and 5xx (server errors) are worth retrying; 4xx is not
            if e.code == 429 or e.code >= 500:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"    HTTP {e.code} — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
            # Print the exact error message so failures are visible, not silent
            err_body = e.read()
            try:
                err_msg = json.loads(err_body).get("error", {}).get("message", "")
            except Exception:
                err_msg = err_body.decode("utf-8", "replace")[:400]
            print(f"    >>> API ERROR {e.code}: {err_msg}")
            return err_body, e.code
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as e:
            # Transient network errors (dropped connection, reset, timeout) — retry
            reason = getattr(e, "reason", e)
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"    Network error ({reason}) — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            return json.dumps({"error": {"message": str(reason)}}).encode(), 503

    return json.dumps({"error": {"message": "Max retries exceeded"}}).encode(), 503

# ── Enrichment ─────────────────────────────────────────────────────────────────

ENRICH_SYSTEM = """You are a financial data researcher for Furton Research. Given a stock ticker or company name, use web search to gather current data and produce a structured investment brief. Be factual and specific — use exact numbers.

Cover: company overview, current valuation (price, market cap, P/E forward/trailing, EV/EBITDA, EV/Revenue), recent financials (last 2 quarters, YoY growth, gross margin, operating margin, FCF), competitive position, management, key risks, and material news in the last 90 days.

Format as clean prose, not bullet points. Write for a sophisticated investor committee. Label each section clearly.

After the prose brief, append exactly three machine-readable lines, each on its own line and in this exact order, with nothing after them:
REFERENCE_PRICE: <the most recent quoted share price you found, in USD, as a bare number with no currency symbol or thousands separators (e.g. 187.42); write Unknown if no price was found>
REFERENCE_PRICE_ASOF: <the as-of date or date-time of that quoted price from your search, ISO format preferred (e.g. 2026-06-24); write Unknown if not determinable>
RETRIEVAL QUERY: <a single dense line of at most 60 words — business description, sector, unit economics, and the key thesis and risk keywords — written to maximize semantic similarity against an investing-philosophy text corpus, not as prose>

CRITICAL OUTPUT RULE: Begin your response immediately with the brief itself, starting with the company name as a heading. Do NOT write any preamble, acknowledgment, or conversational introduction. Never begin with phrases like "Sure," "I'll gather," "Here is," "Let me," or "Certainly." The first words of your output must be the company name. End with exactly the three machine-readable lines described above (REFERENCE_PRICE, REFERENCE_PRICE_ASOF, RETRIEVAL QUERY) and nothing after them — no closing remarks, offers to help, or summary statements."""

def enrich(query):
    print(f"  [Enrich] {query}")
    payload = {
        "model": HAIKU_MODEL,
        # 3500 (was 2000): the brief's prose alone runs ~1,400 output tokens and,
        # with web-search tool blocks counted against the same budget, a 2000-token
        # cap stopped Haiku mid-news (stop_reason=max_tokens) — so the brief was
        # truncated AND the machine-readable trailer (REFERENCE_PRICE / RETRIEVAL
        # QUERY, emitted last) never appeared. Extra Haiku output tokens are cheap.
        "max_tokens": 3500,
        "temperature": TEMPERATURE,
        "system": ENRICH_SYSTEM,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": f"Research and produce a full investment brief for: {query}. Search for current price, recent earnings, valuation multiples, competitive position, and material recent news. Today is June 2026. Begin immediately with the company name — no preamble. End with the REFERENCE_PRICE, REFERENCE_PRICE_ASOF, and RETRIEVAL QUERY lines exactly as instructed."}]
    }
    resp_bytes, status = call_anthropic(payload, timeout=60)
    if status != 200:
        print(f"  [Enrich] failed — API returned status {status} (see error above)")
        return None, {}
    try:
        data = json.loads(resp_bytes)
        usage = data.get("usage", {})
        text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        brief = "\n\n".join(text_parts).strip()
        brief = strip_preamble(brief)
        if not brief:
            print("  [Enrich] failed — got a response but no text content to use")
            return None, usage
        return brief, usage
    except Exception as e:
        print(f"  [Enrich] failed — could not read the response: {e}")
        return None, {}

def strip_preamble(text):
    """Remove any conversational preamble that slips past the system prompt."""
    import re
    # Common preamble openers — if the first line/sentence is one of these, drop it
    preamble_patterns = [
        r"^(sure|certainly|of course|absolutely)[,!.\s].*?\n",
        r"^(here'?s?|here is|here are)\b.*?(brief|analysis|overview|report)[:.]?\s*\n",
        r"^i'?ll?\s+(gather|research|compile|put together|provide|create|prepare).*?\n",
        r"^let me\s+(gather|research|compile|provide|create|prepare|pull).*?\n",
        r"^based on (my|the) (research|search|web search).*?[,:]\s*\n",
    ]
    cleaned = text
    for _ in range(3):  # strip up to 3 stacked preamble lines
        original = cleaned
        for pat in preamble_patterns:
            cleaned = re.sub(pat, "", cleaned, count=1, flags=re.IGNORECASE).lstrip()
        if cleaned == original:
            break
    return cleaned.strip()

# ── Enrichment trailer parsing (RETRIEVAL QUERY / REFERENCE_PRICE) ───────────────
# Every brief ends with three machine-readable lines (see ENRICH_SYSTEM). We parse
# them here. The RETRIEVAL QUERY drives RAG retrieval (BUG 2.2 — embed a focused
# query, not the full brief which the embedder silently truncates to its ~190-word
# window so retrieval only ever saw the brief's opening). REFERENCE_PRICE / _ASOF
# give a decision-time advisory price captured at screen time (GAP 1.6).
_re = re  # module-level `re` (imported at top); alias kept for this section's names

# Tolerant of optional markdown emphasis (**LABEL:**) around the label/colon.
_RETRIEVAL_QUERY_RE = _re.compile(
    r"RETRIEVAL\s+QUERY\s*:\s*\**\s*(.+?)\s*\**\s*$", _re.IGNORECASE | _re.MULTILINE)
_REFERENCE_PRICE_RE = _re.compile(
    r"REFERENCE_PRICE\s*:\s*\**\s*\$?\s*([0-9][0-9,]*\.?[0-9]*)", _re.IGNORECASE)
_REFERENCE_ASOF_RE = _re.compile(
    r"REFERENCE_PRICE_ASOF\s*:\s*\**\s*(.+?)\s*\**\s*$", _re.IGNORECASE | _re.MULTILINE)

_UNKNOWN_VALUES = {"unknown", "n/a", "na", "none", "null", "-", "tbd", ""}

def extract_retrieval_query(brief, fallback_words=180):
    """Return the compact RETRIEVAL QUERY emitted by enrichment, or fall back to
    the first ~fallback_words of the brief if the line is missing/blank. This is
    what RAG embeds, so retrieval keys off a dense thesis/risk summary rather than
    only the brief's truncated opening (BUG 2.2). Never raises."""
    if not brief:
        return ""
    m = _RETRIEVAL_QUERY_RE.search(brief)
    if m:
        q = m.group(1).strip().strip("<>").strip()
        if q and q.lower() not in _UNKNOWN_VALUES:
            return q
    return " ".join(brief.split()[:fallback_words])

def parse_reference_price(brief):
    """Extract the advisory decision-time price + as-of timestamp from the brief
    trailer (GAP 1.6). Returns (price_float_or_None, asof_str_or_None). The price
    is a web-search snapshot — possibly stale or end-of-day — so it is advisory
    only. Never raises and never blocks a screen: missing/garbled → None."""
    if not brief:
        return None, None
    price = None
    m = _REFERENCE_PRICE_RE.search(brief)
    if m:
        try:
            price = float(m.group(1).replace(",", ""))
        except ValueError:
            price = None
        if price is not None and price <= 0:
            price = None
    asof = None
    m2 = _REFERENCE_ASOF_RE.search(brief)
    if m2:
        val = m2.group(1).strip().strip("<>").strip()
        if val.lower() not in _UNKNOWN_VALUES:
            asof = val
    return price, asof

# ── Investor system prompts ────────────────────────────────────────────────────

INVESTOR_SYSTEMS = {
    "Warren Buffett": """You are Warren Buffett, simulated exclusively from Berkshire Hathaway Shareholder Letters (1977–2023) and documented public statements. Additional primary source text will be injected by the Furton Research server.

CORE PRINCIPLES: (1) Demand a durable competitive advantage — moat. Four types: pricing power, switching costs, network effects, low-cost producer. (2) Business buyer not stock trader — "If the market closed for ten years, would I still want this?" (3) Margin of safety via DCF — not EBITDA or EV/revenue. (4) Management must be able AND honest. (5) Circle of competence — cannot picture economics in 10 years? Pass. (6) Concentrate. (7) Permanent skeptic of leverage.

WILL NOT DO: Invest in negative owner earnings, chase fashion, invest in gold/crypto, short stocks, trust narrative over DCF.

VOICE: Plain American English, everyday commerce analogies, reference letter years explicitly. Never use "alpha," "beta," or "risk-adjusted returns."

OUTPUT FORMAT — use exactly:
POSITION: [BUY / PASS]
CONVICTION: [1-10]
THESIS: [100-200 words in Buffett's voice]""",

    "Leopold Aschenbrenner": """You are Leopold Aschenbrenner, simulated exclusively from "Situational Awareness: The Decade Ahead" (June 2024) and your Dwarkesh Patel podcast transcript. Additional primary source text will be injected.

CORE FRAMEWORKS: (1) Regime shift thinking — discontinuous transitions, not incremental change. (2) Physical constraint analysis — compute scaling, power availability, fab capacity, interconnect bandwidth. (3) Critical path framework — progress determined by longest chain of dependent constraints. (4) Infrastructure over application — AI compute/power underpriced vs application layer. (5) National security as central scenario — not tail risk.

EXPANDED DOMAIN: You evaluate any company where AI creates a material investment thesis — either as infrastructure (compute, power, chips), as an AI-native platform (software eating AI workloads), as a major AI adopter whose economics will be transformed by AI productivity gains, or as a company facing serious disruption risk from AI. This covers technology platforms, software companies, financial services automation, healthcare AI, autonomous systems, robotics, and any industrial company whose labor or knowledge-work cost structure will be materially disrupted. You still ABSTAIN on companies with no meaningful AI exposure — pure consumer staples, traditional retail, commodity businesses with no AI angle.

VOICE: Intellectually intense, specific, numerical. Cite "Situational Awareness" chapters. Distinguish high-conviction claims from extrapolations. Flag uncertainty explicitly. When engaging on AI-adjacent companies, assess specifically: how much of their moat depends on AI leadership, and how exposed are they to being disrupted by the next capability jump.

OUTPUT FORMAT — use exactly:
POSITION: [BUY / PASS / ABSTAIN]
CONVICTION: [1-10, or 0 if ABSTAIN]
AI EXPOSURE: [High / Medium / Low / None] — [one sentence on AI relevance]
THESIS: [100-200 words in Aschenbrenner's voice, with domain caveat if ABSTAIN]""",

    "Howard Marks": """You are Howard Marks, simulated exclusively from Oaktree Capital Management memos (1990–2025). Additional primary source text will be injected.

CORE FRAMEWORKS: (1) Second-level thinking — not "is this good?" but "is this good relative to what the market expects?" (2) Cycle assessment — assess where we are on 1-10 scale (1=max fear/opportunity, 10=max greed/danger) from: valuations vs history, credit availability, investor psychology, quality of capital at the margin. (3) Risk is permanent loss, not volatility. (4) Knowable vs unknowable — invest in the former, prepare for the latter. (5) Asymmetric positioning — acceptable upside, limited downside.

WILL NOT DO: Confident macro predictions, invest heavily in late-cycle regardless of quality, treat low volatility as safety.

VOICE: Essayistic, teacherly. Use rhetorical questions. Reference memo titles explicitly. Use "the question isn't whether X is true, but whether X is in the price."

OUTPUT FORMAT — use exactly:
POSITION: [BUY / PASS]
CONVICTION: [1-10]
CYCLE ASSESSMENT: [1-10] — [one sentence]
THESIS: [100-200 words in Marks's voice with second-level thinking and cycle context]""",

    "Joel Greenblatt": """You are Joel Greenblatt, simulated exclusively from "You Can Be a Stock Market Genius" (1997), "The Little Book That Beats the Market" (2005), and "The Little Book That Still Beats the Market" (2010). Additional primary source text will be injected.

CORE FRAMEWORKS: (1) Magic Formula — rank by Return on Capital (EBIT/Net Working Capital + Net Fixed Assets) combined with Earnings Yield (EBIT/Enterprise Value). Best combination of both wins. (2) Special situations — spinoffs, merger securities, restructurings, bankruptcies create systematic mispricings. (3) Mr. Market — buy when irrationally depressed, sell when irrationally euphoric. (4) Margin of safety always required. (5) Formula requires 3-5 year minimum horizon.

VOICE: Plain, self-deprecating, often funny. Use gum shop and Jason analogies. Reference books explicitly. Explain from first principles, avoid jargon.

OUTPUT FORMAT — use exactly:
POSITION: [BUY / PASS]
CONVICTION: [1-10]
RETURN ON CAPITAL: [High / Average / Low / Unknown]
EARNINGS YIELD: [Attractive / Fair / Expensive / Unknown]
SPECIAL SITUATION: [Yes / No] — [one sentence if yes]
THESIS: [100-200 words in Greenblatt's voice]""",

    "Cathie Wood": """You are Cathie Wood, simulated exclusively from ARK Invest Big Ideas reports (2017–2026) and public interview transcripts. Additional primary source text will be injected.

CORE FRAMEWORKS: (1) Wright's Law — costs fall a fixed % per cumulative doubling of units. Battery tech: 28%. DNA sequencing: 40%. Industrial robots: 50%. AI training: 70%/year. (2) Five platform convergence — AI, robotics, energy storage, genomics, blockchain. Most powerful opportunities are at intersections. (3) Five-year scenario-weighted expected value — bear/base/bull with explicit probabilities. (4) Consensus is wrong on innovation — DCF cannot capture exponential change. (5) Leaders, enablers, and beneficiaries for each platform.

WILL NOT DO: Invest in market defenders not market creators, use P/E for growth-phase companies, sell on price weakness if thesis intact.

VOICE: Genuine optimism and conviction. Reference Wright's Law with specific cost decline numbers. Cite Big Ideas reports. Speak in five-year horizons. Engage critics by noting their timeframe is shorter than yours.

OUTPUT FORMAT — use exactly:
POSITION: [BUY / PASS]
CONVICTION: [1-10]
PLATFORMS: [which of five platforms are relevant]
SCENARIOS: Bear [outcome] ([prob]%) / Base [outcome] ([prob]%) / Bull [outcome] ([prob]%)
THESIS: [100-200 words in Wood's voice]"""
}

INVESTORS = list(INVESTOR_SYSTEMS.keys())

# ── Shared blind-evaluation prompt ──────────────────────────────────────────────
# Used by every blind-vote path (real-time committee, screener, and batch) so the
# instruction stays identical across them. The committee is long-only: each member
# votes BUY or PASS (Aschenbrenner may also ABSTAIN on names with no AI angle). A
# security a member finds unattractive is a low-conviction PASS — there is no short
# verdict, consistent with the long-only philosophies the committee comprises.
EVAL_USER_PROMPT = (
    "Please evaluate the following investment opportunity and respond "
    "in exactly the format specified in your instructions.\n\n"
    "INVESTMENT BRIEF:\n{brief}"
)

# ── Structured-verdict parsing (fail loud, not silent) ──────────────────────────
# Members are asked to answer with `POSITION:` / `CONVICTION:` lines, but the model
# routinely wraps those in markdown — `**POSITION:** BUY`, `Position - Buy`,
# `POSITION: **BUY** (conviction 9/10)`. The original `POSITION:\s*(BUY|PASS|ABSTAIN)`
# missed every one of those and the caller then defaulted to PASS / conviction 5 —
# silently recording a real Buy as a middling Pass and poisoning the vote
# denominator. These tolerant patterns absorb markdown (* _ ` ~), an optional colon
# or dash separator, and surrounding whitespace between the label and its value.
_VERDICT_NOISE = r"[\*_`~ \t]*"          # markdown / spacing around label & value
_VERDICT_SEP   = r"[:\-–—]?"   # optional ':' or '-'/en-dash/em-dash

POSITION_RE   = re.compile(
    r"POSITION" + _VERDICT_NOISE + _VERDICT_SEP + _VERDICT_NOISE +
    r"(BUY|PASS|ABSTAIN)", re.I)
CONVICTION_RE = re.compile(
    r"CONVICTION" + _VERDICT_NOISE + _VERDICT_SEP + _VERDICT_NOISE +
    r"(\d{1,2})", re.I)


def parse_verdict(text):
    """Parse a member's POSITION / CONVICTION out of free-form model output.

    Returns (position, conviction, error):
      - position   : 'BUY' | 'PASS' | 'ABSTAIN', or None if unparseable
      - conviction : int 0–10, or None if unparseable/out-of-range
      - error      : None on full success, else a short human-readable reason

    An ABSTAIN with no conviction line is treated as conviction 0 (the documented
    convention), not a failure. Callers MUST check `error` and fail loud — never
    fabricate a PASS/5 on failure.
    """
    text = text or ""
    pos_m  = POSITION_RE.search(text)
    conv_m = CONVICTION_RE.search(text)

    position = pos_m.group(1).upper() if pos_m else None

    conviction = None
    if conv_m:
        n = int(conv_m.group(1))
        if 0 <= n <= 10:
            conviction = n

    # ABSTAIN legitimately carries no conviction; normalize to 0.
    if position == "ABSTAIN" and conviction is None:
        conviction = 0

    problems = []
    if position is None:
        problems.append("position unparseable")
    if conviction is None:
        problems.append("conviction unparseable")

    return position, conviction, ("; ".join(problems) if problems else None)


# ── Committee quorum ────────────────────────────────────────────────────────────
# A committee verdict is only authoritative if enough members actually returned a
# parseable vote. Members that error, time out, or fail to parse are NOT silently
# dropped to make a clean-looking panel: we record who is missing, mark the panel
# incomplete, and refuse to present a BUY off a half panel (the portfolio panel
# would otherwise size a real position on 3 of 5 voices).
#
# Rule (MIN_QUORUM): a full panel (>=4 active) may be missing at most one member —
# need active_count - 1, floored at 3. Smaller panels require everyone.
def committee_quorum(active_count):
    """Minimum number of parseable verdicts for an authoritative panel."""
    if active_count >= 4:
        return max(3, active_count - 1)
    return active_count


def apply_quorum(vote, verdicts, active_investors, missing_members):
    """Annotate `vote` with panel-completeness info and gate the BUY.

    Adds responded / active_count / missing_members / panel_complete / quorum_met.
    If quorum is not met, a would-be BUY is downgraded to 'INCOMPLETE' so no
    downstream consumer sizes a position on a partial committee. A partial PASS is
    left as PASS (nothing is bought either way).
    """
    active_count = len(active_investors)
    responded    = len(verdicts)
    required     = committee_quorum(active_count)
    quorum_met   = responded >= required

    vote["responded"]       = responded
    vote["active_count"]    = active_count
    vote["missing_members"] = missing_members
    vote["panel_complete"]  = (len(missing_members) == 0)
    vote["quorum_met"]      = quorum_met

    if not quorum_met and vote.get("position") == "BUY":
        vote["position"] = "INCOMPLETE"
    return vote


# ── Weighted vote calculation ──────────────────────────────────────────────────

def calculate_committee_vote(verdicts):
    """
    verdicts: list of {investor, position, conviction, ...}
    Returns: {score, position, participating, breakdown}
    """
    score_sum   = 0.0
    weight_sum  = 0.0
    breakdown   = []

    for v in verdicts:
        pos  = v.get("position", "PASS").upper()
        conv = float(v.get("conviction", 0))

        if pos == "ABSTAIN" or conv == 0:
            breakdown.append({**v, "weighted": 0, "included": False})
            continue

        direction = 1 if pos == "BUY" else 0
        weighted  = direction * conv
        score_sum  += weighted
        weight_sum += conv
        breakdown.append({**v, "weighted": weighted, "included": True})

    if weight_sum == 0:
        normalized = 0.0
    else:
        normalized = score_sum / weight_sum

    if normalized >= BUY_THRESHOLD:
        committee_position = "BUY"
    else:
        committee_position = "PASS"

    return {
        "score": round(normalized, 3),
        "position": committee_position,
        "participating": int(weight_sum > 0),
        "breakdown": breakdown
    }

# ── Two-stage filter: should this stock advance to deliberation? ────────────────

# Stage 2 (deliberation + Opus) only runs on stocks where the blind vote
# shows a meaningful signal. This roughly halves cost by skipping deliberation
# on the many stocks that get a unanimous lukewarm PASS.
ADVANCE_SCORE_THRESHOLD = 0.2   # committee score at/above this = clear enough
                                # signal to deliberate. Set below the +0.3 committee
                                # Buy line (BUY_THRESHOLD) so near-miss names — the
                                # ones a deliberation round could actually tip across
                                # the line — advance, while genuinely indifferent
                                # all-PASS names (score near 0) still skip.
ADVANCE_CONVICTION_HIGH = 8     # any single investor this conviction = advance

def should_deliberate(verdicts, vote):
    """
    Returns (advances: bool, reason: str).
    A stock advances to deliberation if the committee shows a real signal:
      - committee score at or above ADVANCE_SCORE_THRESHOLD, OR
      - at least one high-conviction individual voice.
    Otherwise deliberation is skipped as all-around low conviction.
    """
    active = [v for v in verdicts if v.get("position") != "ABSTAIN"]
    if not active:
        return False, "all members abstained"

    score = vote.get("score", 0)
    # Conviction only advances the stock when it sits behind a BUY. A strong PASS
    # (conv 9 against the name) used to trigger an expensive deliberation round on
    # a stock that can never be bought — gauge conviction over BUY voices only.
    max_buy_conv = max((v.get("conviction", 0) or 0
                        for v in active if v.get("position") == "BUY"), default=0)

    if score >= ADVANCE_SCORE_THRESHOLD:
        return True, f"committee score {vote.get('score')} at/above {ADVANCE_SCORE_THRESHOLD}"
    if max_buy_conv >= ADVANCE_CONVICTION_HIGH:
        return True, f"high Buy conviction present ({max_buy_conv}/10)"

    return False, "all-around low conviction"



def build_cached_system(persona_prompt, corpus_context, cache=True):
    """
    Returns the system prompt for an investor call.

    When cache=True, marks the corpus block with cache_control so that
    identical corpus blocks across calls within 5 minutes read from cache
    at 0.1x instead of paying full price. This helps ONLY when the corpus
    block is byte-identical between calls (the even-sampling fallback).

    When cache=False (RAG active), the retrieved chunks differ for every
    stock, so caching would never get a read hit — it would only add the
    1.25x write premium. In that case we send a plain string and pay the
    standard input rate with no premium.

    If there is no corpus, returns a plain string.
    """
    if not corpus_context:
        return persona_prompt

    if not cache:
        # RAG path: context differs per stock, so caching only adds cost.
        return persona_prompt + "\n\n" + corpus_context

    return [
        {
            "type": "text",
            "text": persona_prompt
        },
        {
            "type": "text",
            "text": corpus_context,
            "cache_control": {"type": "ephemeral"}
        }
    ]


def rag_active_for(investor_name, stores):
    """True if this investor is using RAG retrieval (unstable context → no cache)."""
    store = stores.get(investor_name)
    return bool(store and store.has_embeddings)

# ── Batch API helpers ──────────────────────────────────────────────────────────

def submit_batch(requests_list):
    """
    Submit a Message Batch. requests_list is a list of
    {custom_id, params} dicts. Returns (batch_id, error).
    """
    body = json.dumps({"requests": requests_list}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages/batches",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("id"), None
    except urllib.error.HTTPError as e:
        return None, e.read().decode("utf-8", "ignore")
    except urllib.error.URLError as e:
        return None, str(e.reason)

def poll_batch(batch_id):
    """Check batch status. Returns (status_dict, error)."""
    req = urllib.request.Request(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
        },
        method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        return None, e.read().decode("utf-8", "ignore")
    except urllib.error.URLError as e:
        return None, str(e.reason)

def retrieve_batch_results(results_url):
    """
    Fetch JSONL results from results_url. Returns dict mapping
    custom_id -> result text (or None on error).
    """
    req = urllib.request.Request(
        results_url,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
        },
        method="GET"
    )
    out = {}
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry     = json.loads(line)
                custom_id = entry.get("custom_id")
                result    = entry.get("result", {})
                if result.get("type") == "succeeded":
                    msg   = result.get("message", {})
                    text  = "".join(b.get("text","") for b in msg.get("content",[]) if b.get("type")=="text")
                    usage = msg.get("usage", {})
                    out[custom_id] = {"text": text, "usage": usage}
                else:
                    out[custom_id] = {"text": "", "error": result.get("type", "failed")}
            except Exception:
                continue
        return out, None
    except urllib.error.HTTPError as e:
        return None, e.read().decode("utf-8", "ignore")
    except urllib.error.URLError as e:
        return None, str(e.reason)

# Short codes for custom_ids (must be alphanumeric/underscore/hyphen, <64 chars)
INVESTOR_CODE = {
    "Warren Buffett": "WB", "Leopold Aschenbrenner": "LA",
    "Howard Marks": "HM", "Joel Greenblatt": "JG", "Cathie Wood": "CW",
}
CODE_INVESTOR = {v: k for k, v in INVESTOR_CODE.items()}

# ── Weekly committee-record archive (data-capture obligation §7.3) ──────────────

def archive_screen(ticker, entry, new_run=True):
    """Merge one stock's committee record into screens/screen_YYYY-MM-DD.json.

    Append-only by screen date: the first stock screened on a given day creates
    the file; later stocks add their own keyed entry. Persists the blind verdicts
    + deliberation verdicts + vote + statement that the API otherwise discards, so
    deliberation flip-rate and blind-vs-final analysis can be reconstructed later.

    Same-day re-runs are PRESERVED, not clobbered (needed for the §7.2 repeated-run
    stability check). The ticker's top-level fields always reflect the latest run;
    every earlier completed run is pushed onto a per-ticker `runs` list.

    `new_run` controls merge vs. history:
      - new_run=True  (a fresh blind evaluation — screener_stock, batch_blind):
        if a record already exists for this ticker today, the existing record is
        archived into `runs` before the new run replaces the top-level view.
      - new_run=False (a continuation that fills in the *same* run — e.g. the
        batch deliberation half merging onto its blind half): the entry is merged
        into the current top-level record without starting a new run.

    Writes are serialized and atomic (temp file + replace) so concurrent stock
    threads can't corrupt the file. Failures are logged but never break a screen.
    """
    if not ticker:
        return False
    try:
        SCREENS_DIR.mkdir(exist_ok=True)
        date_str = datetime.date.today().isoformat()
        path = SCREENS_DIR / f"screen_{date_str}.json"
        with _archive_lock:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"date": date_str, "temperature": TEMPERATURE, "stocks": {}}

            existing = data["stocks"].get(ticker)
            if existing and new_run and (existing.keys() - {"runs", "archived_at"}):
                # A populated record already exists for today and this is a brand
                # new run — preserve the old run before overwriting the top-level
                # view. `runs` holds every earlier completed run (newest last).
                history = existing.pop("runs", [])
                history.append(existing)
                current = {"runs": history}
            elif existing:
                current = existing          # merge / continuation
            else:
                current = {"runs": []}       # first run for this ticker today

            current.update(entry)
            current["archived_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            current.setdefault("runs", [])
            data["stocks"][ticker] = current

            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.replace(path)
        run_count = len(data["stocks"][ticker].get("runs", [])) + 1
        suffix = f" (run {run_count})" if run_count > 1 else ""
        print(f"  📁 Archived {ticker}{suffix} → screens/screen_{date_str}.json")
        return True
    except Exception as e:
        print(f"  ⚠ Could not archive screen for {ticker}: {e}")
        return False

# ── Request handler ────────────────────────────────────────────────────────────

class FurtonHandler(http.server.BaseHTTPRequestHandler):

    stores = {}

    # Running cost totals for a full screener session (reset on server restart)
    session_cost     = 0.0
    session_in       = 0
    session_out      = 0
    session_searches = 0
    session_stocks   = 0

    def log_message(self, format, *args):
        pass  # suppress default logging

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/screener/dow30list":
            self.handle_dow30_list()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        routes = {
            "/enrich":               self.handle_enrich,
            "/v1/messages":          self.handle_messages,
            "/committee/blind":      self.handle_committee_blind,
            "/committee/deliberate": self.handle_committee_deliberate,
            "/committee/synthesize": self.handle_committee_synthesize,
            "/screener/stock":       self.handle_screener_stock,
            "/screener/batch/submit_blind":      self.handle_batch_submit_blind,
            "/screener/batch/retrieve_blind":    self.handle_batch_retrieve_blind,
            "/screener/batch/submit_deliberate": self.handle_batch_submit_deliberate,
            "/screener/batch/retrieve_deliberate": self.handle_batch_retrieve_deliberate,
            "/screener/batch/poll":              self.handle_batch_poll,
        }
        handler = routes.get(self.path)
        if handler:
            handler()
        elif self.path == "/screener/dow30list":
            self.handle_dow30_list()
        else:
            self.send_response(404)
            self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    # ── /enrich ────────────────────────────────────────────────────────────────

    def handle_enrich(self):
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        query = body.get("ticker") or body.get("description", "").strip()
        if not query:
            self.send_json(400, {"error": "Provide ticker or description"})
            return
        brief, _usage = enrich(query)
        if brief:
            ref_price, ref_asof = parse_reference_price(brief)
            self.send_json(200, {
                "brief": brief, "query": query,
                "reference_price": ref_price,
                "reference_price_asof": ref_asof,
            })
        else:
            self.send_json(503, {"error": "Enrichment failed", "query": query})

    # ── /v1/messages ───────────────────────────────────────────────────────────

    def handle_messages(self):
        try:
            payload = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        system = payload.get("system", "")
        # Identify which investor this is and retrieve relevant context for the brief
        for investor_name, store in self.stores.items():
            if investor_name in system and store:
                # Extract the user's brief text to retrieve against
                brief = ""
                for msg in payload.get("messages", []):
                    if msg.get("role") == "user":
                        c = msg.get("content", "")
                        brief = c if isinstance(c, str) else " ".join(
                            b.get("text", "") for b in c if isinstance(b, dict))
                        break
                context = retrieve_context(investor_name, brief, self.stores)
                payload["system"] = build_cached_system(
                    system, context, cache=not rag_active_for(investor_name, self.stores))
                print(f"  Injected {investor_name} corpus (RAG, {RAG_TOP_K} chunks).")
                break
        payload["model"] = SONNET_MODEL
        payload["temperature"] = TEMPERATURE   # pin for reproducibility
        resp_bytes, status = call_anthropic(payload, timeout=120)
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_bytes)
        if status == 200:
            try:
                data  = json.loads(resp_bytes)
                usage = data.get("usage", {})
                inp   = usage.get("input_tokens", 0)
                out   = usage.get("output_tokens", 0)
                print(f"  Tokens: {inp:,}/{out:,} (≈${stage_cost(payload.get('model', SONNET_MODEL), inp, out):.4f})")
            except Exception:
                pass

    # ── /committee/blind ───────────────────────────────────────────────────────

    def handle_committee_blind(self):
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return

        brief    = body.get("brief", "")
        ticker   = body.get("ticker", "")
        exclude  = body.get("exclude", [])  # list of investor names to skip
        if not brief:
            self.send_json(400, {"error": "Provide brief"})
            return

        active_investors = [n for n in INVESTORS if n not in exclude]
        print(f"\n[Committee blind] {ticker or 'unnamed'} ({len(active_investors)} investors)")

        results   = {}
        errors    = {}
        lock      = threading.Lock()

        def evaluate_investor(investor_name):
            persona  = INVESTOR_SYSTEMS[investor_name]
            context  = retrieve_context(investor_name, brief, self.stores)
            system   = build_cached_system(persona, context,
                                           cache=not rag_active_for(investor_name, self.stores))

            payload = {
                "model":      SONNET_MODEL,
                "max_tokens": 600,
                "temperature": TEMPERATURE,
                "system":     system,
                "messages":   [{
                    "role":    "user",
                    "content": EVAL_USER_PROMPT.format(brief=brief)
                }]
            }

            resp_bytes, status = call_anthropic(payload, timeout=90)

            with lock:
                if status == 200:
                    try:
                        data  = json.loads(resp_bytes)
                        text  = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
                        usage = data.get("usage", {})

                        # Parse structured output — fail loud, never invent a PASS-5
                        position, conviction, perr = parse_verdict(text)

                        results[investor_name] = {
                            "investor":   investor_name,
                            "position":   position,
                            "conviction": conviction,
                            "parse_error": perr,
                            "raw":        text,
                            "tokens":     (usage.get("input_tokens",0), usage.get("output_tokens",0)),
                            "cache_read": usage.get("cache_read_input_tokens", 0),
                            "cache_write": usage.get("cache_creation_input_tokens", 0),
                        }
                        cache_str = ""
                        if usage.get("cache_read_input_tokens", 0) > 0:
                            cache_str = f" [cache HIT: {usage['cache_read_input_tokens']:,} tokens saved]"
                        elif usage.get("cache_creation_input_tokens", 0) > 0:
                            cache_str = f" [cache WRITE: {usage['cache_creation_input_tokens']:,} tokens written]"
                        if perr:
                            # Surface the failure and exclude the member from the
                            # vote (treated like an abstain) — do NOT fabricate a PASS.
                            errors[investor_name] = f"parse-failure: {perr}"
                            print(f"  ✗ {investor_name}: parse-failure ({perr}){cache_str}")
                        else:
                            print(f"  ✓ {investor_name}: {position} ({conviction}){cache_str}")
                    except Exception as e:
                        errors[investor_name] = str(e)
                        print(f"  ✗ {investor_name}: parse error {e}")
                else:
                    try:
                        err = json.loads(resp_bytes)
                        errors[investor_name] = err.get("error", {}).get("message", f"HTTP {status}")
                    except Exception:
                        errors[investor_name] = f"HTTP {status}"
                    print(f"  ✗ {investor_name}: API error {status}")

        # Run active investors in parallel (excluded ones skipped)
        threads = [threading.Thread(target=evaluate_investor, args=(name,)) for name in active_investors]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=150)

        # Only cleanly-parsed members count toward the verdict; parse-failures and
        # API errors are excluded (like abstains) but tracked below for quorum.
        verdicts = [results[name] for name in active_investors
                    if name in results and not results[name].get("parse_error")]
        missing  = [name for name in active_investors
                    if name not in results or results[name].get("parse_error")]
        vote     = calculate_committee_vote(verdicts)
        apply_quorum(vote, verdicts, active_investors, missing)
        if not vote["quorum_met"]:
            print(f"  ⚠ Panel incomplete: {len(verdicts)}/{len(active_investors)} "
                  f"voted; missing {missing} — BUY suppressed")

        total_in    = sum(r["tokens"][0] for r in results.values())
        total_out   = sum(r["tokens"][1] for r in results.values())
        total_read  = sum(r.get("cache_read", 0) for r in results.values())
        total_write = sum(r.get("cache_write", 0) for r in results.values())

        # Actual cost: fresh input at $3/MTok, cache write at $3.75/MTok, cache read at $0.30/MTok, output at $15/MTok
        actual_cost = (total_in * 0.000003) + (total_write * 0.000003750) + (total_read * 0.0000003) + (total_out * 0.000015)
        no_cache_cost = ((total_in + total_read) * 0.000003) + (total_out * 0.000015)
        savings = no_cache_cost - actual_cost

        print(f"  Committee: {vote['position']} (score={vote['score']})")
        print(f"  Tokens: {total_in:,} fresh / {total_read:,} cache_read / {total_write:,} cache_write / {total_out:,} output")
        print(f"  Cost: ${actual_cost:.4f} (saved ${savings:.4f} vs no cache)")

        self.send_json(200, {
            "verdicts": verdicts,
            "vote":     vote,
            "errors":   errors,
            "ticker":   ticker
        })

    # ── /committee/deliberate ──────────────────────────────────────────────────

    def handle_committee_deliberate(self):
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return

        brief    = body.get("brief", "")
        verdicts = body.get("verdicts", [])
        ticker   = body.get("ticker", "")
        exclude  = body.get("exclude", [])

        if not brief or not verdicts:
            self.send_json(400, {"error": "Provide brief and verdicts"})
            return

        active_investors = [n for n in INVESTORS if n not in exclude]
        print(f"\n[Committee deliberate] {ticker or 'unnamed'} ({len(active_investors)} investors)")

        # Build the other-verdicts summary each investor will see
        def build_peer_summary(exclude_name):
            lines = ["The other committee members have submitted their initial verdicts:\n"]
            for v in verdicts:
                if v["investor"] == exclude_name:
                    continue
                lines.append(f"**{v['investor']}**: {v['position']} (conviction {v['conviction']}/10)")
                # Include first 300 chars of their thesis
                raw = v.get("raw", "")
                thesis_match = __import__("re").search(r"THESIS:\s*([\s\S]+)", raw, __import__("re").I)
                if thesis_match:
                    snippet = thesis_match.group(1).strip()[:300]
                    lines.append(f"  \"{snippet}...\"")
                lines.append("")
            return "\n".join(lines)

        results = {}
        lock    = threading.Lock()

        def deliberate_investor(investor_name):
            # Find this investor's blind verdict
            my_verdict = next((v for v in verdicts if v["investor"] == investor_name), None)
            if not my_verdict:
                return

            persona  = INVESTOR_SYSTEMS[investor_name]
            context  = retrieve_context(investor_name, brief, self.stores)
            system   = build_cached_system(persona, context,
                                           cache=not rag_active_for(investor_name, self.stores))

            peer_summary = build_peer_summary(investor_name)

            deliberation_prompt = f"""You previously evaluated this investment and gave:
POSITION: {my_verdict['position']}
CONVICTION: {my_verdict['conviction']}/10

{peer_summary}

Having seen your colleagues' positions, you may now respond to their arguments and revise your conviction if warranted. You do not have to change your position — but you must engage with the most significant disagreement on the committee.

INVESTMENT BRIEF (for reference):
{brief[:1500]}

Respond in exactly this format:
POSITION: [BUY / PASS / ABSTAIN — can be same as before]
CONVICTION: [1-10 — revised if appropriate]
RESPONSE TO COMMITTEE: [100-150 words engaging with the most significant disagreement, in your documented voice]"""

            payload = {
                "model":      SONNET_MODEL,
                "max_tokens": 500,
                "temperature": TEMPERATURE,
                "system":     system,
                "messages":   [{"role": "user", "content": deliberation_prompt}]
            }

            resp_bytes, status = call_anthropic(payload, timeout=90)

            with lock:
                if status == 200:
                    try:
                        data  = json.loads(resp_bytes)
                        text  = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
                        usage = data.get("usage", {})

                        # Tolerant parse; on failure carry forward this member's own
                        # blind verdict (their real prior vote — never a phantom PASS).
                        d_pos, d_conv, d_perr = parse_verdict(text)
                        resp_match  = re.search(r"RESPONSE TO COMMITTEE:\s*([\s\S]+)", text, re.I)

                        results[investor_name] = {
                            "investor":   investor_name,
                            "position":   d_pos  if d_pos  is not None else my_verdict["position"],
                            "conviction": d_conv if d_conv is not None else my_verdict["conviction"],
                            "parse_error": d_perr,
                            "response":   resp_match.group(1).strip()    if resp_match else text,
                            "prev_position":   my_verdict["position"],
                            "prev_conviction": my_verdict["conviction"],
                            "tokens": (usage.get("input_tokens",0), usage.get("output_tokens",0))
                        }
                        if d_perr:
                            print(f"  ⚠ {investor_name}: deliberation parse-failure ({d_perr}) — blind verdict carried forward")
                        changed = results[investor_name]["conviction"] != my_verdict["conviction"]
                        print(f"  ✓ {investor_name}: {results[investor_name]['position']} ({results[investor_name]['conviction']}) {'↕' if changed else ''}")
                    except Exception as e:
                        print(f"  ✗ {investor_name}: {e}")
                else:
                    print(f"  ✗ {investor_name}: HTTP {status}")

        threads = [threading.Thread(target=deliberate_investor, args=(name,)) for name in active_investors]
        for t in threads:
            t.start()
        # Give each thread up to 150s — deliberation calls include corpus + peer summaries
        for t in threads:
            t.join(timeout=150)

        # Flag any investors who timed out without a result
        for name in active_investors:
            if name not in results:
                # Carry forward their blind verdict so they appear in deliberation output
                blind = next((v for v in verdicts if v["investor"] == name), None)
                if blind:
                    results[name] = {
                        "investor":        name,
                        "position":        blind["position"],
                        "conviction":      blind["conviction"],
                        "response":        "[Deliberation timed out — blind verdict carried forward]",
                        "prev_position":   blind["position"],
                        "prev_conviction": blind["conviction"],
                        "tokens":          (0, 0)
                    }
                    print(f"  ⚠ {name}: timed out, blind verdict carried forward")

        deliberation_verdicts = [results[name] for name in active_investors if name in results]
        vote = calculate_committee_vote(deliberation_verdicts)

        total_in  = sum(r["tokens"][0] for r in results.values())
        total_out = sum(r["tokens"][1] for r in results.values())
        print(f"  Final committee: {vote['position']} (score={vote['score']})")
        print(f"  Deliberation tokens: {total_in:,}/{total_out:,} (≈${stage_cost(SONNET_MODEL, total_in, total_out):.4f})")

        self.send_json(200, {
            "deliberation": deliberation_verdicts,
            "vote":         vote,
            "ticker":       ticker
        })

    # ── /committee/synthesize ──────────────────────────────────────────────────

    def handle_committee_synthesize(self):
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return

        blind_verdicts       = body.get("blind_verdicts", [])
        deliberation_verdicts = body.get("deliberation_verdicts", [])
        vote                 = body.get("vote", {})
        brief                = body.get("brief", "")
        ticker               = body.get("ticker", "")

        print(f"\n[Committee synthesize — Opus] {ticker or 'unnamed'}")

        # Build full deliberation record for Opus
        record_lines = [f"INVESTMENT: {ticker}\n", f"COMMITTEE POSITION: {vote.get('position','?')} (weighted score: {vote.get('score','?')})\n\n"]

        record_lines.append("=== BLIND EVALUATION ===\n")
        for v in blind_verdicts:
            record_lines.append(f"\n{v['investor']}: {v['position']} ({v['conviction']}/10)\n")
            record_lines.append(v.get("raw", "")[:500] + "\n")

        record_lines.append("\n=== DELIBERATION ===\n")
        for v in deliberation_verdicts:
            changed = v.get("conviction") != v.get("prev_conviction")
            record_lines.append(f"\n{v['investor']}: {v['position']} ({v['conviction']}/10){' [revised]' if changed else ''}\n")
            record_lines.append(v.get("response", "")[:400] + "\n")

        full_record = "".join(record_lines)

        synthesis_prompt = f"""You are the secretary of the Furton Research Investment Committee. You have observed a full committee deliberation session. Your job is to write the official committee statement — a single authoritative paragraph that:

1. States the committee's final position and weighted conviction score
2. Identifies the dominant argument that carried the majority
3. Names the most significant dissent and why it did not prevail
4. Notes any interesting convergence or surprise that emerged in deliberation
5. Ends with the committee's recommended action

Write in a formal but readable voice — this will go in the public decision log. Do not invent facts. Draw only from the deliberation record provided.

DELIBERATION RECORD:
{full_record}

ORIGINAL INVESTMENT BRIEF (summary):
{brief[:800]}

Write the committee statement now. 150-250 words."""

        payload = {
            "model":      OPUS_MODEL,
            "max_tokens": 400,
            "temperature": TEMPERATURE,
            "messages":   [{"role": "user", "content": synthesis_prompt}]
        }

        resp_bytes, status = call_anthropic(payload, timeout=120)

        if status == 200:
            try:
                data   = json.loads(resp_bytes)
                text   = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
                usage  = data.get("usage", {})
                inp    = usage.get("input_tokens", 0)
                out    = usage.get("output_tokens", 0)
                print(f"  ✓ Opus synthesis complete ({inp:,}/{out:,} tokens, ≈${stage_cost(OPUS_MODEL, inp, out):.4f})")
                self.send_json(200, {"statement": text, "tokens": {"input": inp, "output": out}})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
        else:
            try:
                err = json.loads(resp_bytes)
                self.send_json(status, {"error": err.get("error", {}).get("message", f"HTTP {status}")})
            except Exception:
                self.send_json(status, {"error": f"HTTP {status}"})

    # ── /screener/dow30list ────────────────────────────────────────────────────

    def handle_dow30_list(self):
        self.send_json(200, {"stocks": DOW30, "count": len(DOW30)})

    # ── Internal committee helpers (shared by screener) ────────────────────────

    def _run_blind(self, brief, active_investors):
        """Run blind evaluation for given investors. Returns (verdicts, results_dict)."""
        results = {}
        lock    = threading.Lock()

        def evaluate(investor_name):
            persona = INVESTOR_SYSTEMS[investor_name]
            context = retrieve_context(investor_name, brief, self.stores)
            system  = build_cached_system(persona, context,
                                          cache=not rag_active_for(investor_name, self.stores))
            payload = {
                "model": SONNET_MODEL, "max_tokens": 600, "temperature": TEMPERATURE, "system": system,
                "messages": [{"role": "user", "content": EVAL_USER_PROMPT.format(brief=brief)}]
            }
            resp_bytes, status = call_anthropic(payload, timeout=90)
            with lock:
                if status == 200:
                    try:
                        data  = json.loads(resp_bytes)
                        text  = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
                        usage = data.get("usage", {})
                        # Fail loud — a parse failure is recorded as such (kept for
                        # cost accounting) but excluded from the vote, never PASS-5.
                        position, conviction, perr = parse_verdict(text)
                        if perr:
                            print(f"    ✗ parse-failure {investor_name}: {perr}")
                        results[investor_name] = {
                            "investor": investor_name,
                            "position": position,
                            "conviction": conviction,
                            "parse_error": perr,
                            "raw": text,
                            "tokens": (usage.get("input_tokens",0), usage.get("output_tokens",0)),
                            "cache_read": usage.get("cache_read_input_tokens", 0),
                            "cache_write": usage.get("cache_creation_input_tokens", 0),
                        }
                    except Exception as e:
                        print(f"    parse error {investor_name}: {e}")

        threads = [threading.Thread(target=evaluate, args=(n,)) for n in active_investors]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=150)
        # Exclude parse-failures/errors from the verdict (kept in results for cost).
        verdicts = [results[n] for n in active_investors
                    if n in results and not results[n].get("parse_error")]
        return verdicts, results

    def _run_deliberation(self, brief, verdicts, active_investors):
        """Run deliberation round. Returns deliberation verdicts."""
        import re
        results = {}
        lock    = threading.Lock()

        def peer_summary(exclude_name):
            lines = ["The other committee members submitted these verdicts:\n"]
            for v in verdicts:
                if v["investor"] == exclude_name:
                    continue
                lines.append(f"**{v['investor']}**: {v['position']} (conviction {v['conviction']}/10)")
                tm = re.search(r"THESIS:\s*([\s\S]+)", v.get("raw",""), re.I)
                if tm:
                    lines.append(f'  "{tm.group(1).strip()[:300]}..."')
                lines.append("")
            return "\n".join(lines)

        def deliberate(investor_name):
            my = next((v for v in verdicts if v["investor"] == investor_name), None)
            if not my:
                return
            persona = INVESTOR_SYSTEMS[investor_name]
            context = retrieve_context(investor_name, brief, self.stores)
            system  = build_cached_system(persona, context,
                                          cache=not rag_active_for(investor_name, self.stores))
            prompt = f"""You previously evaluated this investment:
POSITION: {my['position']}
CONVICTION: {my['conviction']}/10

{peer_summary(investor_name)}

Having seen your colleagues' positions, respond to the most significant disagreement and revise your conviction if warranted. You need not change your position.

INVESTMENT BRIEF (reference):
{brief[:1500]}

Respond exactly:
POSITION: [BUY / PASS / ABSTAIN]
CONVICTION: [1-10]
RESPONSE TO COMMITTEE: [100-150 words engaging the key disagreement, in your voice]"""
            payload = {
                "model": SONNET_MODEL, "max_tokens": 500, "temperature": TEMPERATURE, "system": system,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp_bytes, status = call_anthropic(payload, timeout=90)
            with lock:
                if status == 200:
                    try:
                        data = json.loads(resp_bytes)
                        text = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
                        usage = data.get("usage", {})
                        # Tolerant parse; carry forward this member's blind verdict
                        # on failure (their real prior vote, never a phantom PASS).
                        d_pos, d_conv, d_perr = parse_verdict(text)
                        rsp  = re.search(r"RESPONSE TO COMMITTEE:\s*([\s\S]+)", text, re.I)
                        if d_perr:
                            print(f"    ⚠ delib parse-failure {investor_name}: {d_perr} — blind carried forward")
                        results[investor_name] = {
                            "investor": investor_name,
                            "position": d_pos  if d_pos  is not None else my["position"],
                            "conviction": d_conv if d_conv is not None else my["conviction"],
                            "parse_error": d_perr,
                            "response": rsp.group(1).strip() if rsp else text,
                            "prev_position": my["position"],
                            "prev_conviction": my["conviction"],
                            "tokens": (usage.get("input_tokens",0), usage.get("output_tokens",0)),
                            "cache_read": usage.get("cache_read_input_tokens", 0),
                            "cache_write": usage.get("cache_creation_input_tokens", 0),
                        }
                    except Exception as e:
                        print(f"    delib parse error {investor_name}: {e}")

        threads = [threading.Thread(target=deliberate, args=(n,)) for n in active_investors]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=150)
        # Carry forward any that timed out
        for name in active_investors:
            if name not in results:
                blind = next((v for v in verdicts if v["investor"] == name), None)
                if blind:
                    results[name] = {
                        "investor": name, "position": blind["position"],
                        "conviction": blind["conviction"],
                        "response": "[Deliberation timed out — blind verdict carried forward]",
                        "prev_position": blind["position"], "prev_conviction": blind["conviction"],
                        "tokens": (0, 0), "cache_read": 0, "cache_write": 0,
                    }
        return [results[n] for n in active_investors if n in results]

    def _run_synthesis(self, ticker, brief, blind_verdicts, delib_verdicts, vote):
        """Run Opus synthesis. Returns (statement string, usage dict)."""
        lines = [f"INVESTMENT: {ticker}\n", f"COMMITTEE POSITION: {vote.get('position')} (score {vote.get('score')})\n\n"]
        lines.append("=== BLIND ===\n")
        for v in blind_verdicts:
            lines.append(f"\n{v['investor']}: {v['position']} ({v['conviction']}/10)\n{v.get('raw','')[:400]}\n")
        lines.append("\n=== DELIBERATION ===\n")
        for v in delib_verdicts:
            lines.append(f"\n{v['investor']}: {v['position']} ({v['conviction']}/10)\n{v.get('response','')[:350]}\n")
        record = "".join(lines)

        prompt = f"""You are the secretary of the Furton Research Investment Committee. Write the official committee statement: one authoritative paragraph that states the final position and score, the dominant argument that carried the majority, the most significant dissent and why it didn't prevail, and the recommended action. Formal but readable. Draw only from the record.

RECORD:
{record}

BRIEF SUMMARY:
{brief[:600]}

Write the statement now. 120-200 words."""
        payload = {"model": OPUS_MODEL, "max_tokens": 350, "temperature": TEMPERATURE,
                   "messages": [{"role": "user", "content": prompt}]}
        resp_bytes, status = call_anthropic(payload, timeout=120)
        if status == 200:
            try:
                data = json.loads(resp_bytes)
                usage = data.get("usage", {})
                text = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
                return text, usage
            except Exception:
                return "[Synthesis parse error]", {}
        return "[Synthesis failed]", {}

    # ── /screener/stock — full pipeline for one stock ──────────────────────────

    def handle_screener_stock(self):
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return

        ticker        = body.get("ticker", "")
        name          = body.get("name", ticker)
        ai_relevant   = body.get("ai_relevant", False)
        run_delib     = body.get("deliberate", True)
        run_synth     = body.get("synthesize", True)

        if not ticker:
            self.send_json(400, {"error": "Provide ticker"})
            return

        print(f"\n[Screener] {ticker} ({name})")

        # 1. Enrich
        brief, enrich_usage = enrich(f"{name} ({ticker})")
        if not brief:
            self.send_json(200, {"ticker": ticker, "name": name, "error": "Enrichment failed"})
            return

        # Decision-time advisory reference price from the enrichment web-search
        # snapshot (GAP 1.6) — possibly stale/EOD, advisory only, never blocks.
        reference_price, reference_price_asof = parse_reference_price(brief)
        if reference_price is not None:
            print(f"  Ref price: ${reference_price} as of {reference_price_asof or 'unknown'} (advisory)")
        else:
            print("  Ref price: none captured")

        # 2. Determine active investors — auto-include Aschenbrenner only if AI-relevant
        active = [n for n in INVESTORS if n != "Leopold Aschenbrenner"]
        if ai_relevant:
            active.append("Leopold Aschenbrenner")
        # Keep canonical order
        active = [n for n in INVESTORS if n in active]
        print(f"  Active: {len(active)} investors{' (+Aschenbrenner)' if ai_relevant else ''}")

        # 3. Blind
        blind_verdicts, blind_results = self._run_blind(brief, active)
        vote = calculate_committee_vote(blind_verdicts)
        blind_missing = [n for n in active
                         if n not in blind_results or blind_results[n].get("parse_error")]
        apply_quorum(vote, blind_verdicts, active, blind_missing)
        print(f"  Blind: {vote['position']} ({vote['score']})"
              + (f"  ⚠ incomplete — missing {blind_missing}" if not vote["quorum_met"] else ""))

        # 3b. Two-stage filter — does this stock earn a deliberation round?
        advances, reason = should_deliberate(blind_verdicts, vote)

        # 4. Deliberation (only if it advances)
        delib_verdicts = []
        deliberation_skipped = False
        skip_reason = ""
        if run_delib and blind_verdicts and advances:
            delib_verdicts = self._run_deliberation(brief, blind_verdicts, active)
            vote = calculate_committee_vote(delib_verdicts)
            delib_missing = [n for n in active
                             if n not in {v["investor"] for v in delib_verdicts}]
            apply_quorum(vote, delib_verdicts, active, delib_missing)
            print(f"  Final: {vote['position']} ({vote['score']}) — advanced: {reason}")
        elif run_delib and not advances:
            deliberation_skipped = True
            skip_reason = reason
            print(f"  Deliberation SKIPPED — {reason}")

        # 5. Synthesis
        statement = ""
        synth_usage = {}
        if deliberation_skipped:
            # No Opus call — write a clear, deterministic skip statement
            statement = (
                f"Deliberation round skipped due to all-around low conviction. "
                f"The committee's blind vote was {vote['position']} "
                f"(weighted score {vote['score']:+.2f}), below the "
                f"{ADVANCE_SCORE_THRESHOLD} deliberation threshold and with no "
                f"individual conviction at or above {ADVANCE_CONVICTION_HIGH}/10. "
                f"No deliberation or committee statement was generated for this "
                f"stock to conserve cost."
            )
        elif run_synth:
            statement, synth_usage = self._run_synthesis(ticker, brief, blind_verdicts,
                                                         delib_verdicts or blind_verdicts, vote)

        # ── Cost accounting ────────────────────────────────────────────────────
        # Enrich (Haiku + web search)
        e_in   = enrich_usage.get("input_tokens", 0)
        e_out  = enrich_usage.get("output_tokens", 0)
        e_srch = enrich_usage.get("server_tool_use", {}).get("web_search_requests", 0)
        e_cost = stage_cost(HAIKU_MODEL, e_in, e_out, searches=e_srch)

        # Blind (Sonnet) — count every returned response, including parse-failures
        # (excluded from the vote but they still cost tokens).
        b_in  = sum(r.get("tokens", (0, 0))[0] for r in blind_results.values())
        b_out = sum(r.get("tokens", (0, 0))[1] for r in blind_results.values())
        b_cr  = sum(r.get("cache_read", 0)  for r in blind_results.values())
        b_cw  = sum(r.get("cache_write", 0) for r in blind_results.values())
        b_cost = stage_cost(SONNET_MODEL, b_in, b_out, cw=b_cw, cr=b_cr)

        # Deliberation (Sonnet)
        d_in  = sum(r.get("tokens", (0, 0))[0] for r in delib_verdicts)
        d_out = sum(r.get("tokens", (0, 0))[1] for r in delib_verdicts)
        d_cr  = sum(r.get("cache_read", 0)  for r in delib_verdicts)
        d_cw  = sum(r.get("cache_write", 0) for r in delib_verdicts)
        d_cost = stage_cost(SONNET_MODEL, d_in, d_out, cw=d_cw, cr=d_cr)

        # Synthesis (Opus)
        s_in   = synth_usage.get("input_tokens", 0)
        s_out  = synth_usage.get("output_tokens", 0)
        s_cost = stage_cost(OPUS_MODEL, s_in, s_out)

        stock_in   = e_in + b_in + d_in + s_in
        stock_out  = e_out + b_out + d_out + s_out
        stock_cost = e_cost + b_cost + d_cost + s_cost

        # Accumulate across the whole run (resets when the server restarts)
        FurtonHandler.session_cost     += stock_cost
        FurtonHandler.session_in       += stock_in
        FurtonHandler.session_out      += stock_out
        FurtonHandler.session_searches += e_srch
        FurtonHandler.session_stocks   += 1

        print(f"  Enrich (Haiku):  {e_in:,} in / {e_out:,} out / {e_srch} searches  → ${e_cost:.4f}")
        print(f"  Blind  (Sonnet): {b_in:,} in / {b_out:,} out  → ${b_cost:.4f}")
        if delib_verdicts:
            print(f"  Delib  (Sonnet): {d_in:,} in / {d_out:,} out  → ${d_cost:.4f}")
        if s_in or s_out:
            print(f"  Synth  (Opus):   {s_in:,} in / {s_out:,} out  → ${s_cost:.4f}")
        print(f"  STOCK {ticker}: {stock_in:,} in / {stock_out:,} out  → ${stock_cost:.4f}")
        print(f"  SESSION TOTAL: ${FurtonHandler.session_cost:.4f} "
              f"over {FurtonHandler.session_stocks} stock(s), "
              f"{FurtonHandler.session_searches} searches")

        # ── Archive the full dated committee record (data-capture §7.3) ─────────
        # Persist the complete blind + deliberation verdicts, vote, and statement
        # before they're discarded. Stores the untruncated brief for analysis.
        archive_screen(ticker, {
            "ticker":        ticker,
            "name":          name,
            "ai_relevant":   ai_relevant,
            "temperature":   TEMPERATURE,
            "brief":         brief,
            "reference_price":      reference_price,
            "reference_price_asof": reference_price_asof,
            "blind":         blind_verdicts,
            "deliberation":  delib_verdicts,
            "vote":          vote,
            "statement":     statement,
            "advanced":      advances,
            "advance_reason": reason,
            "deliberation_skipped": deliberation_skipped,
            "skip_reason":   skip_reason,
            "active_count":  len(active),
            "panel_complete": vote.get("panel_complete"),
            "quorum_met":     vote.get("quorum_met"),
            "missing_members": vote.get("missing_members", []),
            "cost":          round(stock_cost, 4),
            "source":        "screener_stock",
        })

        self.send_json(200, {
            "ticker":        ticker,
            "name":          name,
            "ai_relevant":   ai_relevant,
            "brief":         brief[:1200],
            "reference_price":      reference_price,
            "reference_price_asof": reference_price_asof,
            "blind":         blind_verdicts,
            "deliberation":  delib_verdicts,
            "vote":          vote,
            "statement":     statement,
            "active_count":  len(active),
            "panel_complete": vote.get("panel_complete"),
            "quorum_met":     vote.get("quorum_met"),
            "missing_members": vote.get("missing_members", []),
            "deliberation_skipped": deliberation_skipped,
            "skip_reason":   skip_reason,
            "advanced":      advances,
            "cost": {
                "enrich":  round(e_cost, 4),
                "blind":   round(b_cost, 4),
                "delib":   round(d_cost, 4),
                "synth":   round(s_cost, 4),
                "stock":   round(stock_cost, 4),
                "input_tokens":  stock_in,
                "output_tokens": stock_out,
                "web_searches":  e_srch,
                "session_total": round(FurtonHandler.session_cost, 4),
                "session_stocks": FurtonHandler.session_stocks,
            },
        })

    # ── Batch mode endpoints ───────────────────────────────────────────────────

    def handle_batch_poll(self):
        """Poll a batch by id. Body: {batch_id}. Returns status + counts."""
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        batch_id = body.get("batch_id")
        status, err = poll_batch(batch_id)
        if err:
            self.send_json(503, {"error": err})
            return
        self.send_json(200, {
            "status":        status.get("processing_status"),
            "request_counts": status.get("request_counts", {}),
            "results_url":   status.get("results_url"),
            "ended":         status.get("processing_status") == "ended",
        })

    def handle_batch_submit_blind(self):
        """
        Body: {stocks: [{ticker, name, ai_relevant, brief}]}
        Builds one request per (stock × active investor), submits as a batch.
        Returns {batch_id, request_count}.
        """
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        stocks = body.get("stocks", [])
        if not stocks:
            self.send_json(400, {"error": "No stocks provided"})
            return

        requests_list = []
        for stock in stocks:
            ticker = stock["ticker"]
            brief  = stock.get("brief", "")
            active = [n for n in INVESTORS if n != "Leopold Aschenbrenner"]
            if stock.get("ai_relevant"):
                active.append("Leopold Aschenbrenner")
            active = [n for n in INVESTORS if n in active]
            for investor in active:
                persona = INVESTOR_SYSTEMS[investor]
                context = retrieve_context(investor, brief, self.stores)
                system  = build_cached_system(persona, context,
                                              cache=not rag_active_for(investor, self.stores))
                requests_list.append({
                    "custom_id": f"blind_{ticker}_{INVESTOR_CODE[investor]}",
                    "params": {
                        "model": SONNET_MODEL, "max_tokens": 600, "temperature": TEMPERATURE, "system": system,
                        "messages": [{"role": "user", "content": EVAL_USER_PROMPT.format(brief=brief)}]
                    }
                })

        print(f"\n[Batch blind] Submitting {len(requests_list)} requests for {len(stocks)} stocks")
        batch_id, err = submit_batch(requests_list)
        if err:
            self.send_json(503, {"error": err})
            return
        print(f"  Batch submitted: {batch_id}")
        self.send_json(200, {"batch_id": batch_id, "request_count": len(requests_list)})

    def handle_batch_retrieve_blind(self):
        """
        Body: {results_url, stocks: [{ticker, name, ai_relevant}]}
        Retrieves blind results, computes committee vote + advancement per stock.
        Returns {results: {ticker: {blind, vote, advances, reason}}}.
        """
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        results_url = body.get("results_url")
        stocks      = body.get("stocks", [])

        raw, err = retrieve_batch_results(results_url)
        if err:
            self.send_json(503, {"error": err})
            return

        out = {}
        for stock in stocks:
            ticker = stock["ticker"]
            active = [n for n in INVESTORS if n != "Leopold Aschenbrenner"]
            if stock.get("ai_relevant"):
                active.append("Leopold Aschenbrenner")
            active = [n for n in INVESTORS if n in active]

            verdicts = []
            missing  = []
            for investor in active:
                key = f"blind_{ticker}_{INVESTOR_CODE[investor]}"
                entry = raw.get(key)
                if not entry or not entry.get("text"):
                    missing.append(investor)         # no result returned for this member
                    continue
                text = entry["text"]
                # Fail loud — a parse failure excludes the member (never PASS-5).
                position, conviction, perr = parse_verdict(text)
                if perr:
                    missing.append(investor)
                    print(f"    ✗ parse-failure {ticker}/{investor}: {perr}")
                    continue
                verdicts.append({
                    "investor": investor,
                    "position": position,
                    "conviction": conviction,
                    "raw": text,
                })

            vote = calculate_committee_vote(verdicts)
            apply_quorum(vote, verdicts, active, missing)
            advances, reason = should_deliberate(verdicts, vote)
            out[ticker] = {
                "blind": verdicts, "vote": vote,
                "advances": advances, "reason": reason,
            }

            # Archive the blind half of the record; deliberate-retrieve merges
            # the deliberation half into the same dated entry by ticker.
            archive_screen(ticker, {
                "ticker":        ticker,
                "name":          stock.get("name", ticker),
                "ai_relevant":   stock.get("ai_relevant", False),
                "temperature":   TEMPERATURE,
                "blind":         verdicts,
                "blind_vote":    vote,
                "advanced":      advances,
                "advance_reason": reason,
                "panel_complete": vote.get("panel_complete"),
                "quorum_met":     vote.get("quorum_met"),
                "missing_members": missing,
                "source":        "batch_blind",
            }, new_run=True)

        advancing = [t for t, r in out.items() if r["advances"]]
        print(f"  Blind retrieved: {len(out)} stocks, {len(advancing)} advance to deliberation")
        self.send_json(200, {"results": out, "advancing": advancing})

    def handle_batch_submit_deliberate(self):
        """
        Body: {advancers: [{ticker, ai_relevant, brief, blind_verdicts}]}
        Builds deliberation requests for advancing stocks, submits batch.
        """
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        advancers = body.get("advancers", [])
        if not advancers:
            self.send_json(200, {"batch_id": None, "request_count": 0})
            return

        import re
        requests_list = []
        for adv in advancers:
            ticker   = adv["ticker"]
            brief    = adv.get("brief", "")
            verdicts = adv.get("blind_verdicts", [])
            active   = [v["investor"] for v in verdicts]

            for investor in active:
                my = next((v for v in verdicts if v["investor"] == investor), None)
                if not my:
                    continue
                # Build peer summary
                peer_lines = ["The other committee members submitted these verdicts:\n"]
                for v in verdicts:
                    if v["investor"] == investor:
                        continue
                    peer_lines.append(f"**{v['investor']}**: {v['position']} (conviction {v['conviction']}/10)")
                    tm = re.search(r"THESIS:\s*([\s\S]+)", v.get("raw",""), re.I)
                    if tm:
                        peer_lines.append(f'  "{tm.group(1).strip()[:300]}..."')
                    peer_lines.append("")
                peer_summary = "\n".join(peer_lines)

                persona = INVESTOR_SYSTEMS[investor]
                context = retrieve_context(investor, brief, self.stores)
                system  = build_cached_system(persona, context,
                                              cache=not rag_active_for(investor, self.stores))
                prompt = f"""You previously evaluated this investment:
POSITION: {my['position']}
CONVICTION: {my['conviction']}/10

{peer_summary}

Having seen your colleagues' positions, respond to the most significant disagreement and revise your conviction if warranted. You need not change your position.

INVESTMENT BRIEF (reference):
{brief[:1500]}

Respond exactly:
POSITION: [BUY / PASS / ABSTAIN]
CONVICTION: [1-10]
RESPONSE TO COMMITTEE: [100-150 words engaging the key disagreement, in your voice]"""
                requests_list.append({
                    "custom_id": f"delib_{ticker}_{INVESTOR_CODE[investor]}",
                    "params": {
                        "model": SONNET_MODEL, "max_tokens": 500, "temperature": TEMPERATURE, "system": system,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                })

        print(f"\n[Batch deliberate] Submitting {len(requests_list)} requests for {len(advancers)} stocks")
        batch_id, err = submit_batch(requests_list)
        if err:
            self.send_json(503, {"error": err})
            return
        print(f"  Batch submitted: {batch_id}")
        self.send_json(200, {"batch_id": batch_id, "request_count": len(requests_list)})

    def handle_batch_retrieve_deliberate(self):
        """
        Body: {results_url, advancers: [{ticker, ai_relevant, blind_verdicts}]}
        Retrieves deliberation results, recomputes vote per stock.
        """
        try:
            body = json.loads(self.read_body())
        except Exception:
            self.send_json(400, {"error": "Invalid JSON"})
            return
        results_url = body.get("results_url")
        advancers   = body.get("advancers", [])

        raw, err = retrieve_batch_results(results_url)
        if err:
            self.send_json(503, {"error": err})
            return

        import re
        out = {}
        for adv in advancers:
            ticker   = adv["ticker"]
            blind    = adv.get("blind_verdicts", [])
            active   = [v["investor"] for v in blind]

            delib = []
            missing = []
            for investor in active:
                key   = f"delib_{ticker}_{INVESTOR_CODE[investor]}"
                entry = raw.get(key)
                my    = next((v for v in blind if v["investor"] == investor), None)
                if not entry or not entry.get("text"):
                    # Carry forward blind verdict on failure (real prior vote).
                    if my:
                        delib.append({
                            "investor": investor, "position": my["position"],
                            "conviction": my["conviction"],
                            "response": "[Deliberation unavailable — blind verdict carried forward]",
                            "prev_position": my["position"], "prev_conviction": my["conviction"],
                        })
                    else:
                        missing.append(investor)
                    continue
                text = entry["text"]
                # Tolerant parse; carry forward blind on failure (never PASS-5).
                d_pos, d_conv, d_perr = parse_verdict(text)
                rsp  = re.search(r"RESPONSE TO COMMITTEE:\s*([\s\S]+)", text, re.I)
                if d_perr and not my:
                    missing.append(investor)
                    print(f"    ✗ delib parse-failure {ticker}/{investor}: {d_perr} (no blind to carry)")
                    continue
                delib.append({
                    "investor": investor,
                    "position": d_pos  if d_pos  is not None else (my["position"]   if my else "PASS"),
                    "conviction": d_conv if d_conv is not None else (my["conviction"] if my else 5),
                    "response": rsp.group(1).strip() if rsp else text,
                    "prev_position": my["position"] if my else "PASS",
                    "prev_conviction": my["conviction"] if my else 5,
                })

            vote = calculate_committee_vote(delib)
            apply_quorum(vote, delib, active, missing)
            out[ticker] = {"deliberation": delib, "vote": vote}

            # Merge the deliberation half into the dated record started at
            # blind-retrieve (same run). "vote" here is the final post-deliberation
            # vote; new_run=False keeps it merged onto its own blind half.
            archive_screen(ticker, {
                "deliberation": delib,
                "vote":         vote,
                "source":       "batch_deliberate",
            }, new_run=False)

        print(f"  Deliberation retrieved: {len(out)} stocks")
        self.send_json(200, {"results": out})

# ── Self-test (no network) — run with: python furton_server.py --selftest ───────

def run_selftests():
    """Offline checks for the vote-integrity fixes. Returns process exit code."""
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"  ✗ {name}\n      got:  {got!r}\n      want: {want!r}")
        else:
            print(f"  ✓ {name}")

    print("\n── parse_verdict (markdown/punctuation tolerance) ──")
    check("plain",            parse_verdict("POSITION: BUY\nCONVICTION: 9"),            ("BUY", 9, None))
    check("bold markdown",    parse_verdict("**POSITION:** BUY\n**CONVICTION:** 9/10"), ("BUY", 9, None))
    check("dash separator",   parse_verdict("Position - Buy\nConviction - 7"),          ("BUY", 7, None))
    check("inline conviction",parse_verdict("POSITION: **BUY** (conviction 9/10)"),     ("BUY", 9, None))
    check("pass low",         parse_verdict("POSITION: PASS\nCONVICTION: 3"),           ("PASS", 3, None))
    check("conviction ten",   parse_verdict("POSITION: BUY\nCONVICTION: 10"),           ("BUY", 10, None))
    check("abstain no conv",  parse_verdict("POSITION: ABSTAIN"),                       ("ABSTAIN", 0, None))
    # Fail-loud cases — must NOT silently become PASS/5
    check("garbage",          parse_verdict("I think this is a great company."),
          (None, None, "position unparseable; conviction unparseable"))
    check("pos only",         parse_verdict("POSITION: BUY"),                ("BUY", None, "conviction unparseable"))
    check("conv out of range",parse_verdict("POSITION: BUY\nCONVICTION: 15"),("BUY", None, "conviction unparseable"))

    print("\n── committee_quorum ──")
    check("quorum of 5", committee_quorum(5), 4)
    check("quorum of 4", committee_quorum(4), 3)
    check("quorum of 3", committee_quorum(3), 3)
    check("quorum of 2", committee_quorum(2), 2)

    print("\n── apply_quorum (BUY gated on half panel) ──")
    five = ["A", "B", "C", "D", "E"]
    v_half = apply_quorum({"position": "BUY", "score": 0.9}, [1, 2, 3], five, ["D", "E"])
    check("3/5 BUY → INCOMPLETE", (v_half["position"], v_half["quorum_met"], v_half["panel_complete"]),
          ("INCOMPLETE", False, False))
    v_one = apply_quorum({"position": "BUY", "score": 0.9}, [1, 2, 3, 4], five, ["E"])
    check("4/5 BUY → stays BUY",  (v_one["position"], v_one["quorum_met"], v_one["panel_complete"]),
          ("BUY", True, False))
    v_pass = apply_quorum({"position": "PASS", "score": 0.0}, [1, 2, 3], five, ["D", "E"])
    check("3/5 PASS → stays PASS",(v_pass["position"], v_pass["quorum_met"]), ("PASS", False))

    print("\n── should_deliberate (high-conviction PASS must not advance) ──")
    pass_conv9 = [{"position": "PASS", "conviction": 9}, {"position": "PASS", "conviction": 1}]
    vote_p = calculate_committee_vote(pass_conv9)
    adv_p, _ = should_deliberate(pass_conv9, vote_p)
    check("unanimous PASS w/ conv9 → skip", adv_p, False)

    buy_conv8 = [{"position": "BUY", "conviction": 8}]
    vote_b = calculate_committee_vote(buy_conv8)
    adv_b, _ = should_deliberate(buy_conv8, vote_b)
    check("BUY conv8 → advance", adv_b, True)

    # Isolate the conviction path: BUY conv8 diluted below the score threshold
    diluted = ([{"position": "BUY", "conviction": 8}]
               + [{"position": "PASS", "conviction": 10}] * 3
               + [{"position": "PASS", "conviction": 5}])
    vote_d = calculate_committee_vote(diluted)
    adv_d, reason_d = should_deliberate(diluted, vote_d)
    check("low-score BUY conv8 → advance via conviction", adv_d, True)

    print()
    if failures:
        print(f"SELFTEST FAILED — {len(failures)} case(s):")
        for f in failures:
            print(f)
        return 1
    print("SELFTEST PASSED — all vote-integrity checks green.")
    return 0


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global API_KEY

    print("=" * 60)
    print("FURTON RESEARCH — Local API Server v3")
    print("=" * 60)

    print("\nLoading API key...")
    API_KEY = load_api_key()
    print(f"  ✓ Key loaded ({API_KEY[:12]}...)")
    print(f"  ✓ Sampling temperature pinned at {TEMPERATURE} (logged in each screen archive)")
    print(f"  ✓ Screen archive dir: {SCREENS_DIR}")

    print("\nLoading primary source libraries...")
    FurtonHandler.stores = {
        "Warren Buffett":        build_rag_store(BUFFETT_LIBRARY,       "Warren Buffett",        filter_fn=lambda c: c.get("high_doctrine", False)),
        "Leopold Aschenbrenner": build_rag_store(ASCHENBRENNER_LIBRARY, "Leopold Aschenbrenner", filter_fn=None),
        "Howard Marks":          build_rag_store(MARKS_LIBRARY,         "Howard Marks",          filter_fn=lambda c: c.get("high_doctrine", False)),
        "Joel Greenblatt":       build_rag_store(GREENBLATT_LIBRARY,    "Joel Greenblatt",       filter_fn=lambda c: c.get("high_doctrine", False)),
        "Cathie Wood":           build_rag_store(WOOD_LIBRARY,          "Cathie Wood",           filter_fn=lambda c: c.get("high_doctrine", False)),
    }

    loaded   = [k for k, v in FurtonHandler.stores.items() if v]
    with_rag = [k for k, v in FurtonHandler.stores.items() if v and v.has_embeddings]
    print(f"\n  {len(loaded)}/5 investor libraries loaded")
    if len(with_rag) == len(loaded) and with_rag:
        print(f"  {len(with_rag)}/5 using RAG retrieval")
    elif with_rag:
        print(f"  {len(with_rag)}/5 using RAG retrieval; the rest use even-sampling fallback")
    else:
        print("  No embeddings found — all using even-sampling fallback.")
        print("  Run furton_embed.py to enable RAG retrieval.")

    server = http.server.HTTPServer(("localhost", PORT), FurtonHandler)
    print(f"\n✓ Server running at http://localhost:{PORT}")
    print("  Endpoints:")
    print("    POST /enrich                — Haiku market data enrichment")
    print("    POST /v1/messages           — individual investor evaluation")
    print("    POST /committee/blind       — all 5 agents simultaneously (Sonnet)")
    print("    POST /committee/deliberate  — deliberation round (Sonnet)")
    print("    POST /committee/synthesize  — official statement (Opus)")
    print("    POST /screener/stock        — full pipeline for one stock (real-time)")
    print("    GET  /screener/dow30list    — current Dow 30 constituents")
    print("    POST /screener/batch/*      — batch mode (50% cost, async)")
    print("\n  Leave this window open. Press Ctrl+C to stop.\n")
    print("-" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(run_selftests())
    main()
