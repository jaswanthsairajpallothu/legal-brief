---
title: Judora
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
---

# Judora

**Complex judgments. Clear understanding.**

Judora creates structured, evidence-linked case briefs from English, text-based Indian court judgments. Each summary statement includes candidate source passages where available, or is marked `Needs review` / `No evidence located`.

**Retrieval is not verification.** Review the original judgment before relying on any summary. Judora does not provide legal advice.

## Run locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860`). Run the tests with:

```powershell
python -m pytest -q
```

The application uses the shipped interim classifier and extractive summarizer by default. Named-entity detection uses spaCy's general-purpose English model and is not legal validation.

## Hugging Face Spaces

The README metadata configures this repository as a native Gradio Space. Create a Space using the Gradio SDK and connect this repository; `app.py` is the entry point. Uploads to a hosted Space are processed on the server. Do not upload confidential, sealed, privileged, or sensitive documents.

`Dockerfile` is also available for container deployments. Hosted mode can be enabled with `LBRIEF_HOSTED=1`.

## Measured results

| What | Result | Caveat |
|---|---|---|
| Section classifier (interim), 5-fold CV by document, 50 Indian Supreme Court judgments, 9,380 sentences, 4 coarse roles | Macro-F1 0.620 (0.572 without document-position feature; 0.181 for the old cue-phrase heuristic) | Not LegalSeg. Configuration was selected on the same CV folds (mildly optimistic). Per-class F1: Reasoning .82, Facts .69, Decision .57, Arguments .40 |
| Evidence locators on 10 judgments rendered to multi-page PDFs | 253 / 253 excerpts found on the cited page and resolved to stored source units | PDFs were generated from text; this tests mapping, not arbitrary real-world layouts |
| ROUGE-1/2/L vs practitioner summaries (MILDSum samples, n=10), extractive mode | 0.564 / 0.274 / 0.267 vs lead-sentences baseline 0.524 / 0.242 / 0.256 | Small sample; overlap is not a measure of legal faithfulness |
| Interface | Gradio workspace for judgment intake, case brief, source inspection, document insights, and methodology | Evidence is a candidate passage, not proof |

## What is rule-based and unevaluated

- **Issue detection** (for example, "Whether ..." cues) and **petitioner vs respondent attribution** (party-name cues, then inherit the last party within four sentences) have not been evaluated against labeled ground truth. A known failure on real text attributed an appellant's submission to the respondent.
- Link thresholds (0.20 weak / 0.45 strong) and the 60-sentence section cap are uncalibrated defaults.
- General-purpose named-entity recognition may miss legal entities and statutes.

## Optional transformer summarizer

The default extractive summarizer does not require transformer weights. To benchmark an optional transformer, install the ML extras and assess outputs manually:

```powershell
python -m pip install -r requirements-ml.txt
python scripts/benchmark_summarizers.py --pdfs ... --models nsi319/legal-pegasus facebook/bart-large-cnn
```

Choose a model only after human review, then set `LBRIEF_SUMMARIZER=transformer` and `LBRIEF_MODEL_ID=<model-id>`. The transformer backend is tested with a stub; real weights have not been evaluated here.

## Reproduce the real-data checks

```powershell
python scripts/fetch_public_data.py
python scripts/train_interim.py
python scripts/evaluate_summaries.py --mildsum data/external/MILDSum/Data/MILDSum_Samples
python scripts/verify_locators.py --mildsum data/external/MILDSum/Data/MILDSum_Samples
```

Third-party data is fetched, not bundled. Check each repository's current terms and do not redistribute MILDSum files.

## Project layout

- `app.py` and `ui/gradio_app.py`: Judora Gradio frontend and application entry point.
- `lbrief/`: parsing, preprocessing, classification, chunking, summarization, evidence linking, metadata, insights, pipeline, export, and evaluation.
- `scripts/`: training, evaluation, locator verification, audit, and public-data retrieval.
- `tests/`: frontend and backend workflow tests.
