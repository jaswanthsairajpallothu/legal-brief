"""Rhetorical-role classification (FR-06).

Runtime uses ONE trained scikit-learn pipeline (TF-IDF + LogisticRegression) saved by scripts/train_classifier.py.
If no artifact exists, a cue-phrase HEURISTIC is used so the app runs for development. It is NOT the
specified model, it is not evaluated, and the UI must show its version string ('heuristic-cues-v0')."""
from __future__ import annotations
import os
import re
import joblib
import numpy as np
from .config import Config
from .models import SourceUnit, RolePrediction

HEURISTIC_VERSION = "heuristic-cues-v0"

_P = r"(petitioner|appellant|plaintiff|applicant|complainant|writ petitioner)"
_R = r"(respondent|defendant|opposite part|state|prosecution|union of india)"
_SUBMIT = r"(submit|submitted|contend|contended|urge|urged|argue|argued|canvass|canvassed|plead|pleaded|contention)"

CUES = [
    ("Decision", [r"\b(appeal|petition|application|suit|writ petition)s?\b.{0,40}\b(is|are|stand|stands)\b.{0,15}\b(allowed|dismissed|disposed|partly allowed)\b",
                  r"\bin the result\b", r"\bwe (accordingly )?(direct|order|set aside|quash)\b", r"\bno order as to costs\b",
                  r"\bordered accordingly\b", r"\bthe impugned (judgment|order).{0,40}\b(set aside|quashed|upheld|affirmed)\b"]),
    ("Issue", [r"^\s*whether\b", r"\b(question|issue)s?\b.{0,40}\b(arise|arises|for (our )?consideration|for determination|framed)\b",
               r"\bpoints? for determination\b"]),
    ("AoP", [rf"\blearned (senior )?counsel for the {_P}\b", rf"\bthe {_P}\b.{{0,60}}\b{_SUBMIT}\b",
             rf"\b{_SUBMIT}\b.{{0,40}}\bby (the )?{_P}\b", rf"\bon behalf of the {_P}\b.{{0,40}}\b{_SUBMIT}"]),
    ("AoR", [rf"\blearned (senior )?(counsel|advocate|solicitor general|additional solicitor general) (appearing )?for the {_R}\b",
             rf"\bthe {_R}\b.{{0,60}}\b{_SUBMIT}\b", rf"\b{_SUBMIT}\b.{{0,40}}\bby (the )?{_R}\b",
             rf"\bon behalf of the {_R}\b.{{0,40}}\b{_SUBMIT}"]),
    ("Reasoning", [r"\bwe are of the (considered |firm |clear )?(view|opinion)\b", r"\bin our (considered )?(view|opinion)\b",
                   r"\bit is (well )?settled\b", r"\bthis court (has )?(held|observed|is of the view)\b",
                   r"\bwe (find|hold|observe|note|are satisfied)\b", r"\b(therefore|thus|hence)\b.{0,60}\b(we|this court|the court)\b",
                   r"\bin the light of the (above|aforesaid)\b"]),
    ("Facts", [r"\b(brief )?facts\b", r"\b(fir|f\.i\.r)\b", r"\bwas (arrested|born|appointed|convicted|married|employed)\b",
               r"\b(filed|lodged|instituted|preferred)\b.{0,30}\b(suit|petition|complaint|appeal|case|fir)\b",
               r"\bcase of the prosecution\b", r"\bon \d{1,2}(st|nd|rd|th)?[ .]\w+[ ,.]+\d{4}\b", r"\bthe trial court\b.{0,30}\b(convicted|acquitted|decreed)\b"]),
]
_COMPILED = [(lab, [re.compile(p, re.I) for p in pats]) for lab, pats in CUES]
_ARG_CONT = re.compile(rf"\b(further|also|added|in addition)\b.{{0,30}}\b{_SUBMIT}\b|^\s*(he|she|it|they)\b.{{0,20}}\b{_SUBMIT}\b", re.I)


def heuristic_labels(texts: list[str]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    last_arg, since = None, 99
    for t in texts:
        lab, score = "None", 0.0
        for name, pats in _COMPILED:
            hits = sum(1 for p in pats if p.search(t))
            if hits:
                lab, score = name, min(1.0, 0.5 + 0.2 * hits)
                break
        if lab == "None" and last_arg and since <= 4 and _ARG_CONT.search(t):
            lab, score = last_arg, 0.4
        if lab in ("AoP", "AoR"):
            last_arg, since = lab, 0
        else:
            since += 1
        out.append((lab, score))
    return out


def add_position_tokens(texts: list[str]) -> list[str]:
    """Append a relative-document-position token (decile) so a plain TF-IDF pipeline can use position."""
    n = max(1, len(texts))
    return [f"{t} zzpos{min(9, int(10 * i / n))}" for i, t in enumerate(texts)]


_PET = re.compile(rf"\b{_P}\b", re.I)
_RES = re.compile(rf"\b{_R}\b", re.I)
_ISSUE = [p for lab, ps in _COMPILED if lab == "Issue" for p in ps]


def refine_coarse(texts: list[str], coarse: list[str], scores: list[float], gate: float) -> list[tuple[str, float]]:
    """Interim scheme only. coarse in {Facts, Arguments, Reasoning, Decision} -> project labels.
    Low-score sentences -> None (not confidently identified). 'Whether...' cues -> Issue. Arguments -> AoP/AoR by
    party cues, else inherit the last party within 4 sentences, else None. Rule parts are UNEVALUATED."""
    out, last_party, since = [], None, 99
    for t, c, sc in zip(texts, coarse, scores):
        lab = c if sc >= gate else "None"
        if c != "Decision" and any(p.search(t) for p in _ISSUE):
            lab = "Issue"
        elif lab == "Arguments":
            pet, res = len(_PET.findall(t)), len(_RES.findall(t))
            if pet != res:
                lab, last_party, since = ("AoP" if pet > res else "AoR"), ("AoP" if pet > res else "AoR"), 0
            elif last_party and since <= 4:
                lab = last_party
            else:
                lab = "None"
        since += 1
        out.append((lab, sc))
    return out


class RoleClassifier:
    def __init__(self, cfg: Config):
        self.pipeline = None
        self.version = HEURISTIC_VERSION
        self.is_trained = False
        self.is_interim = False
        self.bundle: dict = {}
        if os.path.exists(cfg.classifier_path):
            bundle = joblib.load(cfg.classifier_path)
            self.bundle = bundle
            self.pipeline = bundle["pipeline"]
            self.version = bundle.get("version", "tfidf-lr")
            self.is_trained = True
            self.is_interim = bundle.get("label_scheme") == "coarse-jurix"

    def predict(self, units: list[SourceUnit]) -> list[RolePrediction]:
        texts = [u.normalized_text for u in units]
        if not texts:
            return []
        if self.pipeline is not None:
            feed = add_position_tokens(texts) if self.bundle.get("use_position") else texts
            proba = self.pipeline.predict_proba(feed)
            if self.is_interim:
                idx = np.argmax(proba, axis=1)
                coarse = [str(self.pipeline.classes_[i]) for i in idx]
                scores = [float(proba[n, i]) for n, i in enumerate(idx)]
                ref = refine_coarse(texts, coarse, scores, self.bundle.get("gate", 0.45))
                return [RolePrediction(u.source_id, lab, sc, self.version) for u, (lab, sc) in zip(units, ref)]
            classes = self.pipeline.classes_
            idx = np.argmax(proba, axis=1)
            return [RolePrediction(u.source_id, str(classes[i]), float(proba[n, i]), self.version)
                    for n, (u, i) in enumerate(zip(units, idx))]
        return [RolePrediction(u.source_id, lab, sc, self.version)
                for u, (lab, sc) in zip(units, heuristic_labels(texts))]
