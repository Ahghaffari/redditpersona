"""
evaluator.py — Phase 5 evaluation orchestration.

Two phases:
  A. GPU — load each adapter, generate replies, compute perplexity, save.
  B. CPU — load saved generations, compute text/distribution metrics.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import AppConfig, EvaluationCfg, ModelSpec, TrainingCfg
from training.dataset import DatasetBuilder, load_assignments
from .generator import ReplyGenerator
from . import metrics as M

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.tcfg: TrainingCfg = cfg.training
        self.ecfg: EvaluationCfg = cfg.evaluation
        self.base_dir = Path(cfg.data_dir)
        self.eval_dir = Path(self.ecfg.output_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> List[dict]:
        path = Path(self.tcfg.output_dir) / "training_manifest.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path) as f:
            return json.load(f)

    def _gen_path(self, model: str, strategy: str, community: str) -> Path:
        return self.eval_dir / "generations" / model / strategy / f"{community}.json"

    def _save_generation(self, data: dict) -> None:
        p = self._gen_path(data["model"], data["strategy"], data["community"])
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    def _load_generation(self, model: str, strategy: str, community: str) -> Optional[dict]:
        p = self._gen_path(model, strategy, community)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    # test-data extraction
    
    def _splits_for(self, builder: DatasetBuilder, strategy: str, community: str):
        if strategy == "baseline_all":
            return builder.build_baseline_dataset()
        try:
            assignments = load_assignments(strategy, self.base_dir)
        except FileNotFoundError:
            return None
        return builder.build_dataset(strategy, community, assignments)

    def _test_messages(
        self, builder: DatasetBuilder, strategy: str, community: str,
    ) -> Optional[List[List[dict]]]:
        """Return up to N test-set message lists for (strategy, community)."""
        splits = self._splits_for(builder, strategy, community)
        if splits is None or "test" not in splits:
            return None
        msgs = list(splits["test"]["messages"])[: self.ecfg.num_test_samples]
        return msgs or None

    # Phase A
    
    def _run_generation_phase(self, builder: DatasetBuilder) -> None:
        manifest = self._load_manifest()
        ok = [e for e in manifest if e.get("status") == "ok"]

        for ms in self.tcfg.models:
            entries = [e for e in ok if e["model"] == ms.short_name]
            if not entries:
                continue

            for entry in entries:
                strategy = entry["strategy"]
                adapter_dir = entry["adapter_dir"]

                if strategy == "baseline_all":
                    if self._load_generation(ms.short_name, strategy, "all"):
                        continue
                    msgs = self._test_messages(builder, strategy, "all")
                    if msgs is None:
                        continue
                    self._generate_and_save(ms, adapter_dir, strategy, "all", msgs)
                    continue

                # strategy adapter — eval per community
                try:
                    assignments = load_assignments(strategy, self.base_dir)
                except FileNotFoundError:
                    logger.warning("No assignments for %s, skipping", strategy)
                    continue
                if strategy == "strategy_1_subreddit":
                    builder._ensure_indexes()
                    comms = [c for c, _ in Counter(
                        x.get("_subreddit") for x in builder._comments
                        if x.get("_subreddit")
                    ).most_common()]
                else:
                    comms = [c for c, _ in Counter(assignments.values()).most_common()]
                if self.tcfg.max_communities > 0:
                    comms = comms[: self.tcfg.max_communities]

                gen = ReplyGenerator(ms, self.tcfg)
                gen.load(adapter_dir)
                try:
                    for community in comms:
                        if self._load_generation(ms.short_name, strategy, community):
                            continue
                        msgs = self._test_messages(builder, strategy, community)
                        if msgs is None:
                            continue
                        self._generate_with(gen, ms, strategy, community, msgs)
                finally:
                    gen.unload()

            # Zero-shot baseline (base model, no adapter) for first community of each strategy
            self._run_zero_shot(ms, builder)

    def _generate_and_save(
        self, ms: ModelSpec, adapter_dir: Optional[str],
        strategy: str, community: str,
        messages_list: List[List[dict]],
    ):
        gen = ReplyGenerator(ms, self.tcfg)
        gen.load(adapter_dir)
        try:
            self._generate_with(gen, ms, strategy, community, messages_list)
        finally:
            gen.unload()

    def _generate_with(
        self, gen: ReplyGenerator, ms: ModelSpec,
        strategy: str, community: str,
        messages_list: List[List[dict]],
    ):
        prompts, refs, fulls = gen.prepare_eval_data(messages_list)
        if not prompts:
            logger.warning(
                "No usable eval data for model=%s strategy=%s community=%s",
                ms.short_name, strategy, community,
            )
            return
        logger.info(
            "Generating: model=%s strategy=%s community=%s (%d prompts)",
            ms.short_name, strategy, community, len(prompts),
        )
        t0 = time.time()
        generated = gen.generate_replies(
            prompts, max_new_tokens=self.ecfg.max_new_tokens,
            temperature=self.ecfg.temperature, top_p=self.ecfg.top_p,
            do_sample=self.ecfg.do_sample,
            batch_size=self.ecfg.generation_batch_size,
        )
        ppl = gen.compute_perplexity(fulls) if fulls else float("nan")
        self._save_generation({
            "model": ms.short_name, "strategy": strategy, "community": community,
            "generated": generated, "references": refs,
            "perplexity": ppl, "elapsed_s": round(time.time() - t0, 1),
        })
        logger.info("Done: ppl=%.2f, %d replies", ppl, len(generated))

    def _run_zero_shot(self, ms: ModelSpec, builder: DatasetBuilder):
        for strategy in self.tcfg.strategies:
            try:
                assignments = load_assignments(strategy, self.base_dir)
            except FileNotFoundError:
                continue
            if strategy == "strategy_1_subreddit":
                builder._ensure_indexes()
                comms = [c for c, _ in Counter(
                    x.get("_subreddit") for x in builder._comments
                    if x.get("_subreddit")
                ).most_common()]
            else:
                comms = [c for c, _ in Counter(assignments.values()).most_common()]
            for comm in comms[:1]:
                tag = f"{strategy}_{comm}"
                if self._load_generation(ms.short_name, "zero_shot", tag):
                    continue
                msgs = self._test_messages(builder, strategy, comm)
                if msgs is None:
                    continue
                self._generate_and_save(ms, None, "zero_shot", tag, msgs)

    # Phase B
    
    def _run_metrics_phase(self) -> List[dict]:
        results: List[dict] = []
        gen_dir = self.eval_dir / "generations"
        if not gen_dir.exists():
            return results
        enabled = set(self.ecfg.metrics)
        all_gens: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(lambda: defaultdict(dict))
        all_refs: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(lambda: defaultdict(dict))

        for model_dir in sorted(gen_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for strat_dir in sorted(model_dir.iterdir()):
                if not strat_dir.is_dir():
                    continue
                for gf in sorted(strat_dir.glob("*.json")):
                    with open(gf) as f:
                        data = json.load(f)
                    generated, refs = data["generated"], data["references"]
                    community = data["community"]
                    ppl = data.get("perplexity", float("nan"))
                    all_gens[model_dir.name][strat_dir.name][community] = generated
                    all_refs[model_dir.name][strat_dir.name][community] = refs

                    row = {
                        "model": model_dir.name, "strategy": strat_dir.name,
                        "community": community, "n_samples": len(generated),
                        "perplexity": round(ppl, 2),
                    }
                    if "distinct" in enabled:
                        row["distinct_1"] = round(M.distinct_n(generated, 1), 4)
                        row["distinct_2"] = round(M.distinct_n(generated, 2), 4)
                    if "vocab_jaccard" in enabled:
                        row["vocab_jaccard"] = round(M.vocabulary_jaccard(generated, refs), 4)
                    if "topic_kl" in enabled:
                        row["topic_kl"] = round(M.topic_kl_divergence(generated, refs), 4)
                    if "sentiment_jsd" in enabled:
                        row["sentiment_jsd"] = round(M.sentiment_jsd(generated, refs), 4)
                    if "bertscore" in enabled:
                        row["bertscore_f1"] = round(M.bertscore_f1(generated, refs), 4)
                    results.append(row)

        if "community_f1" in enabled:
            for model_name, strats in all_gens.items():
                for strategy, ctexts in strats.items():
                    if strategy in ("zero_shot", "baseline_all"):
                        continue
                    f1 = M.community_classification_f1(ctexts)
                    for r in results:
                        if r["model"] == model_name and r["strategy"] == strategy:
                            r["community_f1"] = round(f1, 4)

        if "mauve" in enabled:
            for model_name, strats in all_gens.items():
                for strategy, ctexts in strats.items():
                    pooled_gen, pooled_ref = [], []
                    for community, gens in ctexts.items():
                        pooled_gen.extend(gens)
                        pooled_ref.extend(all_refs[model_name][strategy][community])
                    m = M.mauve_score(pooled_gen, pooled_ref, device_id=0)
                    for r in results:
                        if r["model"] == model_name and r["strategy"] == strategy:
                            r["mauve"] = round(m, 4)
        return results

    # aggregation
    
    def _build_summary(self, per_adapter: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(per_adapter)
        if df.empty:
            return df
        metric_cols = [c for c in df.columns
                       if c not in ("model", "strategy", "community", "n_samples")]
        summary = df.groupby(["model", "strategy"])[metric_cols].mean().round(4).reset_index()
        summary.insert(2, "n_communities",
                       df.groupby(["model", "strategy"]).size().values)
        return summary

    def run(self) -> pd.DataFrame:
        builder = DatasetBuilder(self.cfg)
        logger.info("=" * 60)
        logger.info("PHASE A — Generation + perplexity (GPU)")
        logger.info("=" * 60)
        self._run_generation_phase(builder)
        logger.info("=" * 60)
        logger.info("PHASE B — Text & distribution metrics (CPU)")
        logger.info("=" * 60)
        per_adapter = self._run_metrics_phase()

        raw = self.eval_dir / "per_adapter_results.json"
        with open(raw, "w") as f:
            json.dump(per_adapter, f, indent=2)
        logger.info("Per-adapter results: %s", raw)

        summary = self._build_summary(per_adapter)
        table = self.eval_dir / "results_table.csv"
        summary.to_csv(table, index=False)
        logger.info("Summary table: %s", table)
        if not summary.empty:
            logger.info("\n%s", summary.to_string(index=False))
        return summary


def run_evaluation(cfg: AppConfig) -> pd.DataFrame:
    return EvaluationRunner(cfg).run()
