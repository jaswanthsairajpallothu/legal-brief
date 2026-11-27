# Court Judgment & Legal Case Summarizer (v2.0)

Evidence-linked case briefs for English, text-based Indian court judgments. Every summary statement carries candidate
source passages with page numbers, or is flagged `Needs review` / `No evidence located`.
**Retrieval is not verification.** Only the human audit (`scripts/audit.py`) can support a faithfulness claim.

## Run
```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                   # uses the shipped interim model in models/
pytest -q                                              # 33 tests
```
Hosted: set `LBRIEF_HOSTED=1` (server-side-processing warning). `Dockerfile` + CI included.

## Measured results (all reproducible; see `reports/`)
| What | Result | Caveat |
|---|---|---|
| Section classifier (interim), 5-fold CV **by document**, 50 Indian SC judgments, 9,380 sentences, 4 coarse roles | **Macro-F1 0.620** (0.572 without document-position feature; **0.181** for the old cue-phrase heuristic) | Not LegalSeg. Config chosen on the same CV (mildly optimistic). Per class F1: Reasoning .82, Facts .69, Decision .57, **Arguments .40** |
| Evidence locators on 10 real judgments rendered to multi-page PDFs | **253 / 253** excerpts found on the cited page and resolve to stored source units | PDFs generated from the text; tests the mapping, not arbitrary real-world layouts |
| ROUGE-1/2/L vs practitioner summaries (MILDSum samples, n=10), extractive mode | 0.564 / 0.274 / 0.267 vs lead-sentences baseline 0.524 / 0.242 / 0.256 | n=10; overlap metric only; slightly above trivial, not impressive |
| Tests | 33 pass (parsing, offsets, chunking, linking, flags, export, UI flow, training plumbing) | |
| UI | Viewed in headless Chromium at desktop and 390 px (`docs/screenshots/`) | Evidence panel is sticky on desktop; stacks below the brief on phones |

## What is rule-based and UNEVALUATED
* **Issue** detection ("Whether ..." cues) and **petitioner vs respondent** attribution (party-name cues, then inherit the last party
  within 4 sentences). The JURIX corpus has no Issue or party labels, so there is no ground truth here. Known failure seen on real text:
  an appellant's submission attributed to the respondent. The human audit exists to quantify this.
* Link thresholds (0.20 weak / 0.45 strong) and the 60-sentence section cap are uncalibrated defaults.

## What could not be done in the build environment (network-restricted) and exactly how to do it
| Item | Command |
|---|---|
| LegalSeg classifier (7 labels; the specified model) | Accept HF terms, download `train.csv/val.csv/test.csv`, then `python scripts/train_classifier.py --train train.csv --val val.csv --doc-col Index --max-per-class 60000` and once `python scripts/evaluate_classifier.py --test test.csv`. Schema (`Text`, `Label`, `Index`, `0`=None) taken from the authors' code. Drop the resulting `models/role_classifier.joblib` in place of the interim one |
| Transformer summarizer | `pip install -r requirements-ml.txt`; `python scripts/benchmark_summarizers.py --pdfs ... --models nsi319/legal-pegasus facebook/bart-large-cnn`; choose by human review; `LBRIEF_SUMMARIZER=transformer LBRIEF_MODEL_ID=<winner>`. Backend logic is tested with a stub only; never run with real weights |
| Human claim-level audit | `python scripts/audit.py export --pdfs ...` -> two reviewers fill the CSV -> `python scripts/audit.py report`. In extractive mode statements are verbatim source sentences, so the audit only becomes informative with the Transformer |
| Public deployment | Needs your account: HF Spaces (Docker) with the provided `Dockerfile`; check memory with the real model |

## Reproduce the real-data checks
```bash
python scripts/fetch_public_data.py            # clones Law-AI/semantic-segmentation (MIT) and Law-AI/MILDSum samples into data/external/
python scripts/train_interim.py                # CV + heuristic baseline + trains models/role_classifier.joblib
M=data/external/MILDSum/Data/MILDSum_Samples
python scripts/evaluate_summaries.py --mildsum $M
python scripts/verify_locators.py --mildsum $M
```
Third-party data is fetched, never bundled. Check each repo's current terms; do not redistribute MILDSum files.

## Layout
`app.py` entrypoint + `views/` (New Judgment, Current Case Brief + evidence panel, Methodology) + `ui/` + `assets/style.css` |
`lbrief/` parser, preprocess, classifier, chunking, summarizer, linker, metadata, insights, pipeline, exporter, evaluation |
`scripts/` train_interim, train_classifier, evaluate_classifier, evaluate_summaries, verify_locators, audit, benchmark_summarizers, fetch_public_data |
`docs/worked_examples.py` (edit distance, Laplace smoothing for the viva) | `tests/`
