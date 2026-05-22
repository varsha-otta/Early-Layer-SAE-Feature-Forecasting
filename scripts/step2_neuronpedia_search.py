"""Step 2: search Neuronpedia for safety-flavored SAE features at Gemma-2-2B layer 20.

Pipeline:
  1. POST /api/explanation/search for each safety-related keyword
  2. Deduplicate results by feature index
  3. Fetch /api/feature/... for the curated top 10 candidates
  4. Write data/neuronpedia_search_raw.json and data/shortlist_v1.json

Idempotent. Network only.

Decision log (which features were picked and why): see docs/02_feature_selection.md.
"""
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make stdout utf-8 so Gemma sentencepiece tokens (e.g. "▁") print on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NEURONPEDIA_API = "https://www.neuronpedia.org/api"
MODEL_ID = "gemma-2-2b"
SAE_LAYER_ID = "20-gemmascope-res-16k"

QUERIES = [
    "refusal", "refuse to answer", "sycophancy", "agreement and flattery",
    "deception", "lying and dishonesty", "hedging", "softening language",
    "harm", "dangerous content", "evasion", "deflection", "apology",
    "warning", "ethics", "moral judgment", "uncertainty", "qualification",
    "manipulation", "persuasion",
]

# The 10 candidates we fetched per-feature detail for (chosen for diversity across
# safety dimensions, all in the target firing rate band). See docs/02_feature_selection.md.
SHORTLIST_CANDIDATES = [
    (9989,  "refusal and resistance",                          "refusal-direct"),
    (2128,  "Refusal to comment or provide information",       "refusal-behavioral"),
    (817,   "lying and falsehoods",                            "deception-direct"),
    (6382,  "situations involving deception or trickery",      "deception-situational"),
    (1959,  "uncertainty or hedging",                          "hedging"),
    (892,   "insincere or exaggerated language",               "sycophancy-adjacent"),
    (1031,  "risk and harm",                                   "harm-direct"),
    (8544,  "warnings about graphic/sensitive content",        "safety-classifier"),
    (1607,  "caution or warning",                              "warning"),
    (12730, "proper behavior and ethics",                      "ethics"),
]


def search_explanations(query: str) -> tuple[str, list]:
    """POST a single keyword search against Neuronpedia explanation-search API."""
    body = json.dumps({
        "modelId": MODEL_ID,
        "layers": [SAE_LAYER_ID],
        "query": query,
    }).encode()
    req = urllib.request.Request(
        f"{NEURONPEDIA_API}/explanation/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return query, json.loads(r.read()).get("results", [])


def fetch_feature(idx: int) -> tuple[int, dict]:
    """GET full record for one feature (includes top-activating example contexts)."""
    url = f"{NEURONPEDIA_API}/feature/{MODEL_ID}/{SAE_LAYER_ID}/{idx}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return idx, json.loads(r.read())


def run_search() -> list[dict]:
    """Run all 20 keyword searches in parallel, dedupe by feature index."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        per_query = list(ex.map(search_explanations, QUERIES))

    by_idx: dict[str, dict] = {}
    for query, results in per_query:
        for r in results[:5]:
            idx = r.get("index")
            if not idx:
                continue
            n = r.get("neuron", {})
            entry = {
                "idx": int(idx),
                "queries": [query],
                "desc": (r.get("description") or "").strip(),
                "frac_nonzero": n.get("frac_nonzero", 0),
                "maxActApprox": n.get("maxActApprox", 0),
                "pos_str": n.get("pos_str", [])[:10],
                "pos_values": [round(v, 2) for v in n.get("pos_values", [])[:10]],
                "similarity": r.get("cosine_similarity", 0),
            }
            if idx in by_idx:
                by_idx[idx]["queries"].append(query)
                if entry["similarity"] > by_idx[idx]["similarity"]:
                    by_idx[idx].update({"desc": entry["desc"], "similarity": entry["similarity"]})
            else:
                by_idx[idx] = entry
    return list(by_idx.values())


def verify_shortlist() -> list[dict]:
    """Fetch per-feature details (incl. top activation contexts) for the curated 10."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = dict(ex.map(fetch_feature, [c[0] for c in SHORTLIST_CANDIDATES]))

    shortlist = []
    for idx, label, theme in SHORTLIST_CANDIDATES:
        d = fetched[idx]
        acts = sorted(d.get("activations", []), key=lambda a: a.get("maxValue", 0), reverse=True)[:5]
        shortlist.append({
            "idx": idx,
            "auto_interp": label,
            "theme": theme,
            "frac_nonzero": d["frac_nonzero"],
            "maxAct": d["maxActApprox"],
            "pos_str": d["pos_str"][:10],
            "top_contexts": [
                {"max_value": a["maxValue"], "tokens": a["tokens"], "values": a["values"],
                 "max_idx": a["maxValueTokenIndex"]}
                for a in acts
            ],
        })
    return shortlist


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"[1/2] Searching {len(QUERIES)} keywords against Neuronpedia...")
    raw = run_search()
    in_band = sum(1 for e in raw if 0.0005 <= e["frac_nonzero"] <= 0.02)
    print(f"      {len(raw)} unique features ({in_band} in firing-rate band [0.0005, 0.02])")
    (data_dir / "neuronpedia_search_raw.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      -> data/neuronpedia_search_raw.json")

    print(f"\n[2/2] Fetching detail + top-activating contexts for {len(SHORTLIST_CANDIDATES)} candidates...")
    shortlist = verify_shortlist()
    (data_dir / "shortlist_v1.json").write_text(
        json.dumps(shortlist, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      -> data/shortlist_v1.json")
    print()
    print("Decision and verdicts: see docs/02_feature_selection.md")
    print("Committed picks: see data/target_features.json")


if __name__ == "__main__":
    main()
