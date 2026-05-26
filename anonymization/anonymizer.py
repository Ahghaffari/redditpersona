"""
Anonymization for collected RedditPersona data (DEFAULT: OFF).

Three operations:
  * username hashing (HMAC-SHA256, "user_" + 12 hex chars)
  * URL stripping  (replaced with "[URL]")
  * spaCy NER PII stripping  (PERSON entities → "[PERSON]")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable

import config
from config import AppConfig

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


class Anonymizer:
    """Apply anonymization transforms to a collected dataset."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.acfg = cfg.anonymization
        self.data_dir = Path(cfg.data_dir)
        self._nlp = None
        self._username_map: Dict[str, str] = {}

    def run(self) -> Dict[str, str]:
        if not self.acfg.enabled:
            logger.info("Anonymization disabled (acfg.enabled=False) — nothing to do.")
            return {}
        logger.info("Anonymizing dataset under %s", self.data_dir)
        self._anonymize_subreddits()
        self._anonymize_jsonl_field(self.data_dir / "user_activity_matrix.jsonl",
                                    text_fields=[], author_fields=["user"])
        self._anonymize_jsonl_field(self.data_dir / "user_interactions.jsonl",
                                    text_fields=[], author_fields=["from", "to"])
        self._anonymize_user_profiles()
        self._save_username_map()
        logger.info("Anonymization complete: %d unique usernames hashed",
                    len(self._username_map))
        return self._username_map

    @property
    def username_map(self) -> Dict[str, str]:
        return self._username_map


    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.acfg.ner_model)
        return self._nlp

    def _hash_username(self, name: str) -> str:
        if name in self._username_map:
            return self._username_map[name]
        if name in config.IGNORED_AUTHORS:
            self._username_map[name] = name
            return name
        h = "user_" + hashlib.sha256(name.encode()).hexdigest()[:12]
        self._username_map[name] = h
        return h

    def _clean_text(self, text: str) -> str:
        if not text:
            return text
        if self.acfg.remove_urls:
            text = _URL_RE.sub("[URL]", text)
        if self.acfg.strip_pii_ner:
            nlp = self._get_nlp()
            doc = nlp(text[:100_000])
            spans = [(e.start_char, e.end_char) for e in doc.ents
                     if e.label_ == "PERSON"]
            for s, e in reversed(spans):
                text = text[:s] + "[PERSON]" + text[e:]
        return text

    def _anonymize_jsonl_field(
        self,
        path: Path,
        text_fields: Iterable[str],
        author_fields: Iterable[str],
    ):
        if not path.exists():
            return
        out = path.with_suffix(path.suffix + ".tmp")
        n = 0
        with open(path, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if self.acfg.hash_usernames:
                    for af in author_fields:
                        if af in rec and rec[af]:
                            rec[af] = self._hash_username(str(rec[af]))
                for tf in text_fields:
                    if tf in rec and rec[tf]:
                        rec[tf] = self._clean_text(rec[tf])
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        out.replace(path)
        logger.info("  Anonymized %s (%d records)", path.name, n)

    def _anonymize_subreddits(self):
        subs_dir = self.data_dir / "subreddits"
        if not subs_dir.exists():
            return
        for d in sorted(subs_dir.iterdir()):
            if not d.is_dir():
                continue
            self._anonymize_jsonl_field(
                d / "posts.jsonl",
                text_fields=["title", "body"],
                author_fields=["author"],
            )
            self._anonymize_jsonl_field(
                d / "comments.jsonl",
                text_fields=["body"],
                author_fields=["author", "post_author"],
            )

    def _anonymize_user_profiles(self):
        prof_dir = self.data_dir / "user_profiles"
        if not prof_dir.exists():
            return
        for ud in sorted(prof_dir.iterdir()):
            if not ud.is_dir():
                continue
            corpus = ud / "text_corpus.txt"
            if corpus.exists() and self.acfg.strip_pii_ner:
                cleaned = self._clean_text(corpus.read_text(encoding="utf-8"))
                corpus.write_text(cleaned, encoding="utf-8")
            if self.acfg.hash_usernames:
                new_name = self._hash_username(ud.name)
                if new_name != ud.name:
                    target = prof_dir / new_name
                    if target.exists():
                        shutil.rmtree(target)
                    ud.rename(target)

    def _save_username_map(self):
        out = self.data_dir / "username_hash_map.json"
        with open(out, "w") as f:
            json.dump(self._username_map, f, indent=2)
        logger.info("Username map saved to %s", out)
