# Furton Research

Furton Research simulates a five-member committee of expert investors — Warren
Buffett, Howard Marks, Joel Greenblatt, Cathie Wood, and Leopold Aschenbrenner —
each a Claude persona grounded in that investor's own primary sources via
retrieval-augmented generation (RAG). The committee deliberates over the Dow 30
and manages a real $10,000 portfolio, benchmarked against the index.

This repository holds the **code** behind the project. It does **not** include
the investor libraries (they are built from copyrighted primary sources — see
[Primary-source libraries](#primary-source-libraries)) or any API key.

The full design is described in the methodology paper published on the project
website; this README covers only how the code is organized and run.

## The four-phase engine

Each weekly screen runs every stock through up to four phases:

1. **Enrichment** — a live market-data brief is assembled (Claude Haiku + web search).
2. **Blind evaluation** — all five agents independently issue a verdict
   (Buy / Pass / Short / Abstain) and a 1–10 conviction, each grounded in its own
   retrieved primary-source context, without seeing the others (Claude Sonnet).
3. **Deliberation** — agents see each other's verdicts and may revise, engaging
   the most significant disagreement (Claude Sonnet). A two-stage filter skips
   this expensive phase for names with no meaningful signal.
4. **Synthesis** — a secretary model writes the official committee statement
   (Claude Opus).

A conviction-weighted vote turns the verdicts into a committee position; Buys are
sized in proportion to conviction (subject to a concentration cap) to form the
long-only portfolio.

## Files

| File | Role |
|---|---|
| `furton_server.py` | Local API server — the four-phase engine, weighted vote, two-stage filter, batch mode, cost accounting, and the dated screen archive. |
| `furton_embed.py` | Builds sentence-embedding vectors for each investor library (for RAG retrieval). |
| `furton_rechunk.py` | Re-splits library chunks into ~250-word windows that fit the embedding model's context, so each chunk is represented in full rather than truncated. |
| `furton_committee.html` | Committee control room — runs a screen, shows verdicts and the deliberation, sizes positions. |
| `furton_tradelog.html` | Weekly trade log and rebalancing ticket. |
| `furton_performance.html` | Equity-curve tracker with risk metrics (Sharpe, volatility, max drawdown, information ratio). |

The `.html` control panels are static pages that talk to the local server.

## Setup

Requires Python 3.10+ and an Anthropic API key.

```bash
pip install -r requirements.txt
```

Put your API key in a single-line file (it should begin with `sk-ant-`):

```
# default location: <your home directory>/furton_api_key.txt
```

To use a different location, set the `FURTON_API_KEY_FILE` environment variable to
the file's path.

## Running

```bash
python furton_server.py
```

The server listens on `http://localhost:8765`. Open any of the `.html` control
panels in a browser to drive it. To enable RAG retrieval, build the libraries
(below) and run `python furton_embed.py` once to generate embeddings; without
embeddings the server falls back to evenly-sampled context.

## Primary-source libraries

The committee's grounding corpora are **not** included in this repository. Each
is assembled from primary sources the investor or their fund produced —
shareholder letters, books, fund memos, reports, and interview transcripts — most
of which are copyrighted and cannot be redistributed.

To reproduce the apparatus you build your own libraries: collect the source
documents, segment them into ~250-word chunks with a citation and metadata per
chunk in a JSON manifest, then run `furton_embed.py` to vectorize them. The server
loads each library from a `manifest.json` (paths configured near the top of
`furton_server.py`).

## Disclaimer

The agents are language-model personas constrained to primary-source text. They
are **not** the real investors, do not represent those investors' current views,
and endorse no security. Nothing here is investment advice.

## Author

Nick Furton — Ross School of Business, University of Michigan —
[nicholas@furton.com](mailto:nicholas@furton.com)
