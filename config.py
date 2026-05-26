"""
RedditPersona — Configuration

Two layers in one file:

1. Module-level CONSTANTS.
   Used by Phase 1 collection, Phase 2 grouping, and Phase 3 analysis.

2. AppConfig DATACLASSES + YAML loader.
   Used by anonymization, Phase 4 training and Phase 5 evaluation
   (loaded from `config.yaml`).

Anything large lives under DATA_DIR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# Layer 1: Grouping constants

# Reddit API credentials
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv(
    "REDDIT_USER_AGENT",
    "RedditPersonaBenchmark/0.1 (research)",
)


# Directory layout
BASE_DIR = Path(__file__).parent

# Heavy outputs go here.
DATA_DIR       = Path(os.getenv("DATA_DIR", "/mnt/data/RedditPersona/reddit_data"))
SUBS_DIR       = DATA_DIR / "subreddits"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOGS_DIR       = DATA_DIR / "logs"
PROFILES_DIR   = DATA_DIR / "user_profiles"

VERIFIED_SUBS_FILE    = DATA_DIR / "verified_subreddits.json"
USER_MATRIX_FILE      = DATA_DIR / "user_activity_matrix.jsonl"
INTERACTION_FILE      = DATA_DIR / "user_interactions.jsonl"
COLLECTION_SUMMARY    = DATA_DIR / "collection_summary.json"


# Target subreddits
SUBREDDITS: dict[str, list[str]] = {
    "civic": [
        "politics", "PoliticalDiscussion", "AskAnAmerican",
        "Ask_Politics", "nonprofit",
    ],
    "culture": ["AskHistorians", "history", "books", "architecture"],
    "economy": [
        "personalfinance", "economics", "finance", "povertyfinance",
        "financialindependence", "Frugal", "lostgeneration", "economy",
        "LateStageCapitalism", "FluentInFinance", "investing", "UKPersonalFinance",
    ],
    "education": [
        "education", "Teachers", "college", "GradSchool", "AskAcademia",
        "learnprogramming", "explainlikeimfive",
    ],
    "environment": [
        "environment", "climate", "climatechange", "sustainability", "ZeroWaste",
        "energy", "RenewableEnergy", "UrbanPlanning", "fuckcars", "transit",
        "bicycling", "cycling", "ecology",
    ],
    "health": [
        "health", "mentalhealth", "depression", "Anxiety", "ADHD", "Fitness",
        "nutrition", "EatCheapAndHealthy", "EOOD", "loseit",
    ],
    "housing": [
        "RealEstate", "FirstTimeHomeBuyer", "homeowners", "Landlord",
        "REBubble", "YIMBY", "homeless", "realestateinvesting",
    ],
    "innovation": [
        "technology", "Futurology", "science", "startups", "smallbusiness",
        "MachineLearning", "datascience",
    ],
    "jobs": [
        "antiwork", "WorkReform", "jobs", "careerguidance", "cscareerquestions",
        "recruitinghell", "overemployed", "legaladvice",
    ],
    "justice": ["law", "SupremeCourt", "Bad_Cop_No_Donut", "AmItheAsshole"],
    "politics_institutions": ["geopolitics", "worldnews", "news", "NeutralPolitics"],
    "demographics": [
        "AskEurope", "immigration", "IWantOut", "expats", "dataisbeautiful",
    ],
    "safety": ["TrueCrime", "ProtectAndServe", "PublicFreakout"],
    "wellbeing": [
        "DecidingToBeBetter", "selfimprovement", "Stoicism", "simpleliving",
        "offmychest", "TrueOffMyChest", "socialskills", "relationship_advice",
        "Meditation", "GetMotivated", "productivity", "internetparents",
        "CasualConversation", "socialanxiety",
    ],
    "cross_cutting": [
        "AskReddit", "LifeProTips", "changemyview", "unpopularopinion",
        "NoStupidQuestions", "TooAfraidToAsk", "self", "Advice",
    ],
}

ALL_SUBREDDITS: list[str] = [s for subs in SUBREDDITS.values() for s in subs]
SUB_TO_CATEGORY: dict[str, str] = {
    s: c for c, subs in SUBREDDITS.items() for s in subs
}


# Verification thresholds
MIN_SUBSCRIBERS: int  = 5_000
REQUIRE_PUBLIC:  bool = True
REQUIRE_SFW:     bool = True


# Collection parameters
POST_SORTS: list[str] = ["hot", "top", "new"]
POSTS_PER_SUBREDDIT: int | None = None
TOP_TIME_FILTER: str = "year"
REPLACE_MORE_LIMIT: int | None = 32
COMMENT_MIN_BODY_CHARS: int = 10
POST_MIN_BODY_CHARS: int = 0
COLLECT_LINK_POSTS: bool = True


# Async / rate-limit / retry
MAX_CONCURRENT_REQUESTS: int = 3
SLEEP_BETWEEN_POSTS: float   = 0.5
SLEEP_BETWEEN_SUBS:  float   = 2.0
BACKOFF_BASE: float = 2.0
MAX_RETRIES:  int   = 5
REQUEST_TIMEOUT: float = 30.0
RATELIMIT_SECONDS: int = 300


# Checkpoint cadence
CHECKPOINT_EVERY_N_POSTS: int = 100


# Activity matrix / interaction graph
BUILD_USER_ACTIVITY_MATRIX: bool = True
BUILD_INTERACTION_GRAPH:    bool = True
IGNORED_AUTHORS: frozenset[str] = frozenset({"[deleted]", "[removed]", "AutoModerator"})


# Anonymization (hard OFF by default)
ANONYMIZE: bool = False
ANON_SALT: str  = os.getenv("ANON_SALT", "")


# Phase 2 (user profiles)
MIN_USER_COMMENTS:   int = 10
MIN_USER_SUBREDDITS: int = 2

# Open file-handle pool size used by the streaming profile builder.
# Tune downward if `ulimit -n` is small.
PROFILE_OPEN_FILE_POOL: int = 1024


# Phase 3 (grouping)
DEFAULT_N_COMMUNITIES: int | None = None
EMBEDDING_MODEL: str    = "google/embeddinggemma-300m"
EMBEDDING_BATCH_SIZE: int = 32
KMEANS_N_CLUSTERS: int  = 100
HYBRID_ALPHA_VALUES: list[float] = [0.5]
HYBRID_ALPHA_DEFAULT: float = 0.5
GRAPH_RESOLUTION: float = 2.0

TOP_K_COMMUNITIES: int = 100

# Strategy 5 (user-interaction Leiden)
LEIDEN_RESOLUTION: float = 1.0
INTERACTION_MIN_EDGE_WEIGHT: int = 1

# Strategy 2 (graph projection — memory-safe controls)
GRAPH_MAX_USERS_PER_SUB: int = 2000
# Drop user-user edges with summed weight below this after projection.
GRAPH_MIN_EDGE_WEIGHT: float = 2.0
# Flush the (row, col, data) accumulator into CSR every N appended pairs.
GRAPH_FLUSH_EVERY_PAIRS: int = 20_000_000


# Phase 4 (analysis)
TOP_TFIDF_TERMS: int = 20
MIN_COMMUNITY_SIZE_FOR_ANALYSIS: int = 5


# Logging
LOG_LEVEL: str = "INFO"


# Layer 2: AppConfig dataclasses (anonymization + training + evaluation)
import yaml

@dataclass
class AnonymizationCfg:
    enabled: bool = False
    hash_usernames: bool = True
    strip_pii_ner: bool = True
    remove_urls: bool = True
    ner_model: str = "en_core_web_sm"


@dataclass
class ModelSpec:
    name: str
    short_name: str


@dataclass
class TrainingCfg:
    models: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec("ibm-granite/granite-4.1-3b", "granite-4.1-3b"),
    ])
    strategies: List[str] = field(default_factory=lambda: [
        "strategy_1_subreddit", "strategy_2_graph", "strategy_3_semantic",
        "strategy_4_hybrid", "strategy_5_interaction",
    ])
    max_communities: int = 8
    max_train_samples: int = 20000
    device: str = "cuda:1"
    output_dir: str = "/mnt/data/RedditPersona/reddit_data/adapters"
    model_cache_dir: str = "/mnt/data/RedditPersona/reddit_data/models"
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    split_method: str = "random"
    max_context_comments: int = 3
    max_seq_length: int = 512
    min_reply_length: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    num_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 25
    save_strategy: str = "no"
    fp16: bool = False
    bf16: bool = False
    hf_token: Optional[str] = None


@dataclass
class EvaluationCfg:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_test_samples: int = 200
    generation_batch_size: int = 4
    output_dir: str = "/mnt/data/RedditPersona/reddit_data/evaluation"
    metrics: List[str] = field(default_factory=lambda: [
        "perplexity", "distinct", "vocab_jaccard",
        "topic_kl", "sentiment_jsd",
        "bertscore", "mauve", "community_f1",
    ])


@dataclass
class AppConfig:
    """Bundle of all dataclass-based configs (training / evaluation / anon).

    The `data_dir` is resolved from the module-level DATA_DIR constant; tests
    and training read everything they need from there.
    """
    data_dir: str = str(DATA_DIR)
    anonymization: AnonymizationCfg = field(default_factory=AnonymizationCfg)
    training:      TrainingCfg      = field(default_factory=TrainingCfg)
    evaluation:    EvaluationCfg    = field(default_factory=EvaluationCfg)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load an AppConfig from a YAML file. Missing keys fall back to defaults."""
    p = Path(path)
    if not p.is_file():
        return AppConfig()

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig()
    cfg.data_dir = raw.get("data_dir", cfg.data_dir)

    if "anonymization" in raw:
        a = raw["anonymization"]
        cfg.anonymization = AnonymizationCfg(**{
            k: a[k] for k in a if k in AnonymizationCfg.__dataclass_fields__
        })

    if "training" in raw:
        t = raw["training"]
        defaults = TrainingCfg()
        models = [ModelSpec(**m) for m in t.get("models", [])] or defaults.models
        kwargs = {k: t[k] for k in t if k in TrainingCfg.__dataclass_fields__ and k != "models"}
        cfg.training = TrainingCfg(models=models, **kwargs)
        cfg.training.hf_token = os.environ.get("HF_TOKEN")

    if "evaluation" in raw:
        e = raw["evaluation"]
        cfg.evaluation = EvaluationCfg(**{
            k: e[k] for k in e if k in EvaluationCfg.__dataclass_fields__
        })

    return cfg
