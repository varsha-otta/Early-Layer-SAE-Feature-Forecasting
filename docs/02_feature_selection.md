# Step 2: Target feature selection

**Decision**: option A: pick features `[9989, 817, 12730, 892, 1031]` as the precursor-probe targets.

**Date**: 2026-05-22.

## Goal

Choose 3-5 SAE features at Gemma-2-2B layer 20 (residual stream, width 16k) to use as targets for the precursor probes in Steps 3-6. Criteria:

- Safety-flavored auto-interp label (refusal, deception, sycophancy, harm, ethics, hedging, etc.)
- Firing rate in the ~0.05-1% band; rare enough to be specific, common enough to learn
- Auto-interp label verified to actually match the top-activating contexts (because auto-interp labels are LLM-generated and can be wrong)
- Diversity across safety dimensions

## Data source

GemmaScope canonical residual-stream SAEs (Lieberum et al. 2024) on Gemma-2-2B (Gemma Team 2024), layer 20, width 16k. Feature browsing via the **Neuronpedia public API** (Lin et al. 2023; no auth required).

Full bibliographic entries for these references are in `docs/07_writeup.md#references`.

| Resource | URL |
|---|---|
| Browse a single feature | `https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/<idx>` |
| Search by keyword | `POST https://www.neuronpedia.org/api/explanation/search` |
| Get single-feature details | `GET https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/<idx>` |

### Example: keyword search

```bash
curl -X POST https://www.neuronpedia.org/api/explanation/search \
  -H "Content-Type: application/json" \
  -d '{"modelId": "gemma-2-2b", "layers": ["20-gemmascope-res-16k"], "query": "refusal"}'
```

The response ranks features by cosine similarity between the auto-interp embedding and the query embedding. For each match, the response includes the full feature record: `frac_nonzero` (firing rate), `maxActApprox` (max observed activation), `pos_str` / `pos_values` (top tokens the feature direction is aligned with in the unembed), activation histograms, etc.

### Example: feature detail (with top-activating contexts)

```bash
curl https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/9989
```

The `activations` field of the response contains up to 45 example contexts; each is a list of tokens with per-token activation values, plus the index of the max-activating token. This is what lets us verify whether the auto-interp label actually matches what the feature does on real text.

## Search

We queried 20 safety-relevant concepts in parallel:

```
refusal, refuse to answer, sycophancy, agreement and flattery,
deception, lying and dishonesty, hedging, softening language,
harm, dangerous content, evasion, deflection, apology,
warning, ethics, moral judgment, uncertainty, qualification,
manipulation, persuasion
```

Each query returns the top ~5 matches. After deduplicating by feature index:

- **79 unique candidate features** surfaced across the 20 queries
- **73 of these** are in the target firing rate band [0.0005, 0.02]

Raw results: `data/neuronpedia_search_raw.json`.

## Verification of the top 10 candidates

Auto-interp labels are LLM-generated (`gemini-2.5-flash-lite` per the response metadata) and are not always reliable. We selected the 10 most-promising candidates spanning safety dimensions, then fetched their top-activating example contexts to check whether activations actually match the label.

This step was load-bearing: **3 of the 10 candidates turned out to be mislabeled or too narrow**, and we'd have wasted Step 3-4 cycles training probes on them if we hadn't checked.

| ★/▲/✗ | # | Auto-interp | Fire % | Theme | Verdict |
|---|---|---|---|---|---|
| ★ | **9989** | refusal and resistance | 0.75 | refusal | **Strong.** Top tokens: refusal, refused, resistance. Fires anticipatorily on tokens just before refusal verbs (e.g. on "she" in "she refused"). Exactly the precursor pattern we want. |
| ★ | **817**  | lying and falsehoods | 0.76 | deception | **Strong.** Top tokens: false, lie, lies, lied, falsehood. Activations: fires on "testified" before "falsely", on "faced" in "bold-faced lie". |
| ★ | **12730** | proper behavior and ethics | 0.49 | ethics | **Strong.** Top tokens: ethical, ethics, moral, morals, decency. Activations cleanly fire on ethics-relevant phrasing ("rules of decency and good behaviour"). |
| ▲ | **892**  | insincere or exaggerated language | 0.28 | sycophancy-adj. | **Decent.** Top tokens: exaggeration, cliché, exaggerating, sounding. Activations: "is a serious understatement". Closest sycophancy-adjacent feature in this base model. |
| ▲ | **1031** | risk and harm | 0.48 | harm | **Mostly clean.** Top English tokens: risk, harm, jeopardize. Mild polysemy flag: top-1 cosine-aligned token in `pos_str` is an Arabic word (يتيمه = "her orphan"), suggesting a spurious unembed direction, but English activations look genuinely harm-related. |
| ▲ | 1607 | caution or warning | 0.49 | warning | Interesting: anticipatory firing on "But" before caveats. Some non-safety contexts (programming caveats). **Not picked** for option A; would swap in for 12730 if we wanted a second anticipatory feature. |
| ▲ | 1959 | uncertainty or hedging | 0.37 | hedging | Mixed: auto-interp tokens (perhaps, maybe) match but top *activation* contexts fire on parenthetical qualifications ("centuries even") rather than epistemic hedging. **Not picked.** |
| ✗ | 8544 | "warnings about graphic/sensitive content" | 0.12 | (mislabeled) | **Rejected.** Top tokens are fanfiction, fanfic, chapter, fic - it's a fanfiction-structure detector, not a safety classifier. Auto-interp got it wrong. |
| ✗ | 2128 | "refusal to comment" | 0.14 | (too narrow) | **Rejected.** All top-activating examples are the *same* news template ("officials could not be reached for comment"). Overfit to one context type. |
| ✗ | 6382 | "deception/trickery" | 0.27 | (polysemantic) | **Rejected.** Primarily a literal "trap" detector (top tokens: trap, traps, Trap, TRAP) with polysemantic overlap into "subterfuge". |

Verified candidates with their top activation contexts: `data/shortlist_v1.json`.

## Decision

**Committed picks**: `[9989, 817, 12730, 892, 1031]` (option A from the conversation shortlist).

| # | Theme | Firing rate | Why this one |
|---|---|---|---|
| **9989** | refusal | 0.75% | Strongest safety-flavored feature; anticipatory firing pattern is precursor-relevant. |
| **817** | deception | 0.76% | Clean concept, well-verified contexts. |
| **12730** | ethics | 0.49% | Clean concept, broad coverage of moral language. |
| **892** | sycophancy-adj. | 0.28% | Best sycophancy-adjacent feature available in this base model. |
| **1031** | harm | 0.48% | Diversifies across safety dimensions to harm-recognition. |

Rationale: maximum diversity across safety dimensions, all in the 0.28-0.76% firing band, all verified against actual activation contexts.

## Things we couldn't find

- **A clean sycophancy feature.** Gemma-2-2B *base* wasn't instruction-tuned, so sycophantic-response behavior hasn't crystallized into a single feature at layer 20. Feature 892 is the closest proxy. If sycophancy turns out to be the target we care most about, the natural next step is to move to Gemma-2-2B-it (instruction-tuned) with one of the community SAEs (e.g., from Apollo / Goodfire).
- **A clean "controversial topic" / "sensitive subject" feature**: none surfaced from our 20 queries.
- **A clean "harmful-content recognition" feature distinct from the general "risk and harm" feature 1031.**

If Step 4 probe results suggest one of our picks is too noisy, the natural swap candidates are:
- Replace 892 (sycophancy-adj) with 1607 (warning), as it gives a second anticipatory feature
- Replace 1031 (harm) with 1959 (hedging), as it gives a softer epistemic feature

## Reproduction

The full search + verification pipeline is in `scripts/step2_neuronpedia_search.py`:

```bash
cd feature-precursor-detection
python scripts/step2_neuronpedia_search.py
# Writes: data/neuronpedia_search_raw.json, data/shortlist_v1.json
```

Idempotent. Network only; no model loading or GPU needed.

## Saved artifacts

| File | Content |
|---|---|
| `data/neuronpedia_search_raw.json` | All 79 unique candidates from the 20-keyword search, with `idx`, `desc`, `frac_nonzero`, `maxActApprox`, `pos_str`, `similarity` |
| `data/shortlist_v1.json` | The 10 verified candidates with top activation contexts (tokens + per-token values) |
| `data/target_features.json` | The 5 committed picks - machine-readable handoff to Step 3 |

## Neuronpedia links for the chosen features

- 9989 (refusal): https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/9989
- 817 (deception): https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/817
- 12730 (ethics): https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/12730
- 892 (sycophancy-adj.): https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/892
- 1031 (harm): https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/1031
