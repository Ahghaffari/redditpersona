<p align="center">
  <img src="images/redditpersona_logo.svg" alt="RedditPersona logo" height="120"/>
</p>


<p align="center">
  A modular pipeline for <strong>community-conditioned LLM adaptation from Reddit</strong>.<br/>
  Collects Reddit data, builds per-user profiles, partitions users under five community grouping strategies,<br/>
  trains a parameter-efficient LoRA adapter per strategy via QLoRA, and evaluates each adapter<br/>
  on a shared suite of nine generation-quality and persona-fidelity metrics.
</p>



---

## Pipeline

<p align="center">
  <img src="images/redditpersona_pipeline.svg" alt="RedditPersona six-phase pipeline" width="93%"/>
</p>

The pipeline runs in six phases: **(1) data collection** from Reddit via AsyncPRAW; **(2) user profiling**, building structured profiles and text corpora per active user; **(3) community grouping** under five interchangeable strategies; **(4) LLM adaptation** via QLoRA fine-tuning with community identity encoded in the system prompt; **(5) generation** of replies for held-out test conversations; and **(6) evaluation** across fluency, diversity, semantic fidelity, distributional alignment, and community identifiability metrics.

---

## Repository layout

```
Redditpersona/
├── config.py              # module-level constants + AppConfig dataclasses
├── config.yaml            # training / evaluation / anonymization options
├── .env.example           # Reddit API + data paths + (optional) HF token
├── requirements.txt
├── pyproject.toml
├── run.py                 # unified CLI
├── images/
│   ├── redditpersona_logo.svg
│   └── redditpersona_pipeline.svg
│
├── collection/
│   ├── verify_subreddits.py
│   └── collect_subreddits.py
├── profiles/
│   └── build_profiles.py
├── grouping/
│   ├── strategy_subreddit.py      # S1: activity-sorted primary subreddit
│   ├── strategy_graph.py          # S2: Leiden on bipartite user-subreddit projection
│   ├── strategy_semantic.py       # S3: KMeans on EmbeddingGemma text embeddings
│   ├── strategy_hybrid.py         # S4: hybrid of S2 graph + semantic similarity
│   ├── strategy_interaction.py    # S5: Leiden on symmetrised user-user reply graph
│   ├── analyze.py                 # coherence / separation / overlap metrics
│   └── run_all.py                 # orchestrates strategies 1–5
│
├── anonymization/         # HMAC-SHA256 users + URL + spaCy NER
├── training/              # QLoRA SFT (peft + trl), one adapter per strategy
└── evaluation/            # generation, perplexity, MAUVE, BERTScore, …
```

On-disk strategy IDs (output dirs under `data/groupings/`) are `strategy_1_subreddit … strategy_5_interaction`, keeping cross-strategy comparison and the analyzer's strategy list stable.

---

## Grouping strategies

| ID | Module | Method |
|----|--------|--------|
| S1 | `strategy_subreddit.py` | Activity-sorted primary subreddit — subreddit baseline, one community per sub |
| S2 | `strategy_graph.py` | Leiden community detection on bipartite user-subreddit projection |
| S3 | `strategy_semantic.py` | KMeans on EmbeddingGemma text embeddings, *k* chosen by silhouette sweep |
| S4 | `strategy_hybrid.py` | Hybrid of S2 graph similarity and S3 semantic similarity |
| S5 | `strategy_interaction.py` | Leiden on the symmetrised user-user reply graph |

All strategies are consolidated to the top-*K* communities plus an `other` bucket (see `TOP_K_COMMUNITIES` in `config.py`). Training and evaluation use the top-10.

A single execution selects one grouping strategy and trains one adapter on that partition. To compare all five strategies, keep the input data fixed and rerun the pipeline across all strategies, producing one adapter per strategy.

---

## Data flow

| Step | Sub-command | Output artifacts |
|------|-------------|-----------------|
| 1 | `verify` | `data/verified_subreddits.json` |
| 2 | `collect` | `data/subreddits/{sub}/{posts,comments}.jsonl`, `data/user_activity_matrix.jsonl`, `data/user_interactions.jsonl` |
| 3 | `profiles` | `data/user_profiles/{user}/{text_corpus.txt,activity.json,profile.json}`, `data/user_index.json` |
| 4 | `grouping` | `data/groupings/{strategy_id}/{community_assignments.json, …}` |
| 5 | `analysis` | `data/groupings/comparison_table.{json,csv}` |
| 6 | `anonymize` | in-place rewrite |
| 7 | `training` | `{output_dir}/{short_name}/{strategy}/community_pooled/` adapters + `training_manifest.json` |
| 8 | `evaluation` | `{output_dir}/{generations/,per_adapter_results.json,results_table.csv}` |

`DATA_DIR` defaults to `/mnt/data/RedditPersona/reddit_data` and can be overridden via the `DATA_DIR` environment variable (set in `.env`).

---

## Environment

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a Python ≥ 3.10 virtual environment and install dependencies
uv venv .venv --python ">=3.10"
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure secrets and paths
cp .env.example .env
# Edit .env: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
# DATA_DIR, and optionally HF_TOKEN and ANON_SALT
```

---

## Run

```bash
cd Redditpersona

python run.py verify
python run.py collect
python run.py profiles
python run.py grouping --strategies 1 2 3 4 5
python run.py analysis
python run.py anonymize
python run.py training        # uses config.yaml
python run.py evaluation
```

Or chain everything except the long collection step:

```bash
python run.py all
```

A fast single-strategy pipeline test for the training pipeline is available at `training/training_test.py`.

---

## Training

For each strategy, all top-*K* community datasets are **pooled** into a single training set with the community label injected into the system prompt of each example. One LoRA adapter is trained per strategy, yielding **5 strategy adapters + 1 unconditional `baseline_all` adapter** per base model.

Default hyperparameters (see `config.yaml`):

- **Base model:** `ibm-granite/granite-4.1-3b`
- **Quantization:** QLoRA, NF4 4-bit + double-quant, fp16 compute
- **LoRA:** r = 16, α = 32, dropout = 0.05, targets `q/k/v/o_proj`
- **Optimizer:** 1 epoch, batch 2 × accum 8 (effective 16), lr 1e-5, cosine schedule, warmup 0.10
- **Sequence length:** 512; max pooled training samples per strategy: 20,000
- **Split:** 80 / 10 / 10. Two modes via `training.split_method` in `config.yaml`:
  - `random` (default, used in the reported experiments) — splits at the sample level.
  - `user` — splits at the *author* level so the same user never appears in more than one split (prevents author-identity leakage).
- **Format:** OpenAI `messages` format with the tokenizer's own chat template applied automatically — system prompt encodes community identity and grouping strategy; user turn concatenates post title, body, up to three level comments, and the parent comment; assistant turn is the community member's reply, truncated to 512 tokens. This makes the same training script portable across model families (Qwen, Llama, Mistral, Gemma, etc.).

---

## Evaluation

Per-adapter metrics computed on the held-out test split (up to 200 test prompts per adapter, temperature 0.7, top-p 0.9, max 256 new tokens):

| Metric | Description |
|--------|-------------|
| PPL | Token-level perplexity |
| Dist-1 / Dist-2 | Distinct unigram / bigram ratio |
| Vocab-Jacc | TF-IDF vocabulary Jaccard (top-100 terms) |
| Topic-KL | LDA topic KL divergence (10 topics) |
| Sent-JSD | VADER sentiment Jensen-Shannon divergence |
| BERTScore-F1 | Semantic similarity to reference replies |
| MAUVE | Distributional alignment, pooled across communities per strategy |
| Comm-F1 | Community classification F1 via TF-IDF logistic regression *(undefined for `zero_shot` and `baseline_all`)* |

---

## Notes

- To use anonymization, set `ANON_SALT` to a high-entropy secret:
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Heavy artefacts (`data/`, `adapters/`, `evaluation_out/`, `*.log`) and `.env` are git-ignored.

---

## Citation

If you use RedditPersona in your research, please cite our paper (currently under review):

```bibtex
@article{ghaffari2026redditpersona,
  title   = {{RedditPersona}: A Modular Framework for Community-Conditioned {LLM} Adaptation from {Reddit}},
  author  = {Ghaffari, Amirhossein and Goodarzi, Ali and Nguyen, Huong and
             Hosio, Simo and Lov{\'e}n, Lauri and Gilman, Ekaterina},
  year    = {2026},
  note    = {Under review}
}
```
