# services/ingest/app/services/nlp/nlp_processor.py
#
# spaCy-Grundpipeline: Lädt das deutsche Sprachmodell und
# analysiert Chunks (Tokenisierung, POS-Tagging, Dependency Parsing).

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()

_DEFAULT_CONFIG = Path(__file__).parents[3] / "nlp_config.yaml"


def load_nlp_config(config_path: Optional[Path] = None) -> dict:
    """Lädt nlp_config.yaml – immer frisch von Disk (kein Caching)."""
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"nlp_config.yaml nicht gefunden: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Ergebnis-Datenklassen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenInfo:
    text:    str
    lemma:   str
    pos:     str
    dep:     str
    head:    str
    is_stop: bool


@dataclass
class SentenceInfo:
    text:   str
    tokens: list[TokenInfo] = field(default_factory=list)
    start:  int = 0
    end:    int = 0


@dataclass
class ChunkAnalysis:
    """Vollständige spaCy-Analyse eines Chunks."""
    chunk_id:   str
    sentences:  list[SentenceInfo] = field(default_factory=list)
    token_count: int = 0
    sent_count:  int = 0
    # Rohes spaCy-Doc – für SVO- und NER-Extraktion weiterverwendet
    doc: object = None


# ─────────────────────────────────────────────────────────────────────────────
# NLPProcessor
# ─────────────────────────────────────────────────────────────────────────────

class NLPProcessor:
    """
    Wrapper um spaCy. Lädt das Modell einmalig (lazy),
    liest Konfiguration bei jedem Aufruf frisch von Disk.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or _DEFAULT_CONFIG
        self._nlp = None

    def _get_nlp(self):
        """Lazy-Loading des spaCy-Modells."""
        if self._nlp is None:
            cfg = load_nlp_config(self.config_path)
            model = cfg["spacy"]["model"]
            components = cfg["spacy"].get("components",
                                          ["tagger", "parser", "lemmatizer"])

            logger.info("Lade spaCy-Modell", model=model)
            try:
                import spacy
                # tok2vec NIEMALS deaktivieren – parser und tagger hängen davon ab.
                # Nur sichere Komponenten deaktivieren die keine Abhängigkeiten haben.
                safe_to_disable = ["senter", "ner", "morphologizer"]
                disable = [p for p in safe_to_disable if p not in components]
                self._nlp = spacy.load(model, disable=disable)
                logger.info("spaCy-Modell geladen",
                            model=model, pipes=self._nlp.pipe_names)
            except OSError:
                raise OSError(
                    f"spaCy-Modell '{model}' nicht gefunden.\n"
                    f"Installation: python -m spacy download {model}"
                )
        return self._nlp

    def analyze(self, chunk_id: str, text: str) -> ChunkAnalysis:
        """
        Analysiert einen einzelnen Chunk.
        Gibt ChunkAnalysis mit Sätzen, Tokens und rohem Doc zurück.
        """
        cfg  = load_nlp_config(self.config_path)
        nlp  = self._get_nlp()
        max_len = cfg["spacy"].get("max_text_length", 100000)

        text = text[:max_len]
        doc  = nlp(text)

        sentences = []
        for sent in doc.sents:
            tokens = [
                TokenInfo(
                    text=t.text, lemma=t.lemma_,
                    pos=t.pos_, dep=t.dep_,
                    head=t.head.text, is_stop=t.is_stop,
                )
                for t in sent
            ]
            sentences.append(SentenceInfo(
                text=sent.text, tokens=tokens,
                start=sent.start_char, end=sent.end_char,
            ))

        return ChunkAnalysis(
            chunk_id=chunk_id,
            sentences=sentences,
            token_count=len(doc),
            sent_count=len(list(doc.sents)),
            doc=doc,
        )

    def analyze_batch(
        self,
        chunks: list[tuple[str, str]],   # [(chunk_id, text), ...]
    ) -> list[ChunkAnalysis]:
        """
        Analysiert eine Liste von Chunks effizient via nlp.pipe().
        """
        cfg        = load_nlp_config(self.config_path)
        nlp        = self._get_nlp()
        batch_size = cfg["spacy"].get("batch_size", 32)
        max_len    = cfg["spacy"].get("max_text_length", 100000)

        texts    = [t[:max_len] for _, t in chunks]
        ids      = [cid for cid, _ in chunks]
        results  = []

        for doc, chunk_id in zip(
            nlp.pipe(texts, batch_size=batch_size), ids
        ):
            sentences = []
            for sent in doc.sents:
                tokens = [
                    TokenInfo(
                        text=t.text, lemma=t.lemma_,
                        pos=t.pos_, dep=t.dep_,
                        head=t.head.text, is_stop=t.is_stop,
                    )
                    for t in sent
                ]
                sentences.append(SentenceInfo(
                    text=sent.text, tokens=tokens,
                    start=sent.start_char, end=sent.end_char,
                ))

            results.append(ChunkAnalysis(
                chunk_id=chunk_id,
                sentences=sentences,
                token_count=len(doc),
                sent_count=len(list(doc.sents)),
                doc=doc,
            ))

        return results
