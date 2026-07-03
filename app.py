from __future__ import annotations

import ast
import json
import os
import queue
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_file
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer
from werkzeug.utils import secure_filename
import gc

# --- Configuration & Paths ---
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
RESULT_DIR = APP_ROOT / "result"
MODEL_PATH = APP_ROOT / "model.pkl"
DOWNLOAD_FILENAME = "submission.csv"
SUPPORTED_DATA_EXTS = {".csv", ".xlsx", ".xls", ".jsonl", ".json"}
NAME_FIELDS = ["candidate_name", "full_name", "name", "display_name", "anonymized_name"]

app = Flask(__name__)

# --- Threading & State Control ---
state_lock = threading.Lock()
task_queue: "queue.Queue[str | None]" = queue.Queue()
processed_files: set[str] = set()
observer: Observer | None = None

is_processing = False
watcher_active = False
log_lines = ["[SYSTEM] Pipeline initialized."]
submission_cache = {"signature": None, "rows": []}

# --- Reasoning Engine Registry ---
_TOKENIZER = None
_SEQ2SEQ_MODEL = None
_REASONING_READY = False


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def add_log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    with state_lock:
        log_lines.append(line)
        del log_lines[:-200]


def _current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_json_field(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        for parser in (json.loads, lambda v: json.loads(v.replace("'", '"')), ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}


def parse_skills_field(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, lambda v: json.loads(v.replace("'", '"')), ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    continue
        parts = re.split(r"[;,]", text)
        return [{"name": part.strip()} for part in parts if part.strip()]
    return []


def extract_candidate_name(row: pd.Series) -> str:
    # 1. Check nested profile dict first — your data stores the name at profile.anonymized_name
    profile = row.get("profile")
    if isinstance(profile, str):
        profile = parse_json_field(profile)
    if isinstance(profile, dict):
        nested_name = profile.get("anonymized_name") or profile.get("name") or profile.get("full_name")
        if isinstance(nested_name, str) and nested_name.strip():
            return nested_name.strip()

    # 2. Fall back to flat top-level columns
    for field in NAME_FIELDS:
        if field in row.index:
            value = row.get(field)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    return cleaned
            elif pd.notna(value):
                cleaned = str(value).strip()
                if cleaned and cleaned.lower() != "nan":
                    return cleaned

    # 3. Last resort
    candidate_id = row.get("candidate_id")
    if pd.notna(candidate_id) and str(candidate_id).strip() and str(candidate_id).strip().lower() != "nan":
        return f"Candidate {candidate_id}"
    return "Anonymous"


def _pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def build_model_wrapper(ranker=None) -> dict:
    return {
        "ranker": ranker,
        "feature_columns": [
            "days_since_active",
            "recency_multiplier",
            "recruiter_response_rate",
            "offer_acceptance_rate",
            "profile_completeness_score_norm",
            "skill_assessment_scores_norm",
            "github_activity_score_norm",
            "skill_match_score",
        ],
        "target_skills": {
            "python",
            "pytorch",
            "tensorflow",
            "llm",
            "transformers",
            "rag",
            "ranking",
            "retrieval",
            "xgboost",
            "lightgbm",
        },
        "skill_tags": {
            "complete": [
                "complete mastery of the AI stack - LLMs, RAG, PyTorch, Transformers",
                "deep, hands-on command of LLMs and the PyTorch/Transformers ecosystem",
                "full coverage of every core skill this role needs",
            ],
            "strong": [
                "strong alignment to core AI frameworks (LLMs, PyTorch, Transformers)",
                "a solid grasp of the key AI frameworks this role depends on",
                "good depth across LLMs, PyTorch, and Transformers",
            ],
            "moderate": [
                "meaningful overlap with the target AI stack",
                "partial coverage of the AI stack this role needs",
                "real exposure to the relevant tools, though not across the board",
            ],
            "low": [
                "foundational AI exposure",
                "early-stage exposure to the core AI tooling",
                "limited overlap with this role's specific stack",
            ],
        },
        "github_tags": {
            "exceptional": [
                "exceptional open-source contributions across multiple repos",
                "a GitHub history like a full-time open-source maintainer's",
                "an unusually prolific public commit history",
            ],
            "consistent": [
                "consistent hands-on development with real shipped code",
                "a steady cadence of shipped side projects",
                "regular commits showing genuine project momentum",
            ],
            "moderate": [
                "moderate GitHub activity with solid practical experience",
                "a reasonable amount of public code",
                "some visible project history",
            ],
            "limited": [
                "limited public GitHub activity",
                "a thin public footprint, possibly due to private repos",
                "little visible on GitHub",
            ],
        },
        "resp_tags": {
            "high": [
                "exceptionally responsive to recruiters",
                "quick to reply to outreach",
                "consistently fast to respond",
            ],
            "medium": [
                "reliably responsive to outreach",
                "easy to reach for a conversation",
                "responsive enough to suggest real interest",
            ],
            "low": [
                "selectively responsive - outreach should be personalized",
                "not consistently reachable",
                "harder to get a response from",
            ],
        },
        "recency_tags": {
            "active_now": [
                "actively job-seeking right now",
                "clearly mid-search at this moment",
                "currently engaged on the platform",
            ],
            "recent_quarter": [
                "active within the last quarter",
                "engaged in an ongoing search recently",
                "showing recent search activity",
            ],
            "stale_6mo": [
                "last active about 6 months ago",
                "quiet for roughly half a year",
                "not seen on the platform in months",
            ],
            "inactive": [
                "not recently active",
                "inactive for a long stretch",
                "uncertain interest given the long inactivity",
            ],
        },
        "profile_tags": {
            "complete": [
                "a fully detailed profile",
                "a profile with essentially nothing missing",
                "thorough profile documentation",
            ],
            "good": [
                "a well-completed profile",
                "enough profile detail for a fair read",
                "a reasonably complete profile",
            ],
            "partial": [
                "a partially complete profile",
                "some gaps worth clarifying on a call",
                "a profile that's still thin in places",
            ],
        },
        "skill_lines": [
            "This candidate has {skill_tag}.",
            "Technically, this candidate brings {skill_tag}.",
            "On the skills front, this candidate offers {skill_tag}.",
            "This candidate's core strength is {skill_tag}.",
            "Right away, {skill_tag} stands out on this profile.",
            "This person demonstrates {skill_tag}.",
        ],
        "evidence_lines": [
            "Their GitHub backs this up with {github_tag}.",
            "On GitHub, they show {github_tag}.",
            "This is supported by {github_tag} on GitHub.",
            "Their track record includes {github_tag}.",
            "Public code confirms this: {github_tag}.",
            "Their portfolio reflects {github_tag}.",
        ],
        "verdict_lines": [
            "Hire: they are {resp_tag} and {recency_tag}, with {profile_tag}.",
            "Recommended for outreach - {resp_tag} and {recency_tag}.",
            "Worth pursuing now: {recency_tag} and {resp_tag}.",
            "This is a strong hire candidate, given they're {resp_tag} and {recency_tag}.",
            "Move forward with this candidate - {resp_tag}, {recency_tag}.",
            "Solid pick: {recency_tag}, {resp_tag}, and {profile_tag}.",
        ],
        "default_reasoning": (
            "This candidate shows a balanced profile for the role, with enough useful signals to justify outreach. "
            "The evidence is practical rather than perfect, so a short conversation would help confirm fit. "
            "Overall, the profile is strong enough to keep in the active hiring queue."
        ),
        "min_valid_length": 50,
        "min_sentences": 2,
        "max_sentences": 4,
        "echo_patterns": re.compile(
            r"(\bcomplete this (short|recruiter)\b|\bfacts:\s|\brecommendation:\s*this candidate\s*$|"
            r"\bdo not repeat\b|\bdo not use numbers\b|\bwrite exactly\b)",
            re.IGNORECASE,
        ),
        "on_topic_patterns": re.compile(
            r"(candidate|engineer|github|recruiter|skill|hire|hiring|role|technical|"
            r"experience|profile|developer|ai\b|stack|portfolio|track record|"
            r"public code|outreach|responsive|pursue|commit|repo)",
            re.IGNORECASE,
        ),
        "junk_patterns": re.compile(
            r"(https?://|www\.|\.com\b|\bemail to\b|\bpost a resume\b|\binput your\b|"
            r"\bapply (for|to)\b|\bnode\.html\b)",
            re.IGNORECASE,
        ),
        "malformed_patterns": re.compile(r"(^\s*:|\s:\s|[a-zA-Z]-[a-zA-Z]?\s*$)"),
        "identifier_leak_patterns": re.compile(
            r"(#\d+|\brank(?:ed)?\s*\d+|\bscore\s*(?:of\s*)?[\d.]+|\bcand[_\-]?\w*\d+|\bcandidate[_\s-]?id\b)",
            re.IGNORECASE,
        ),
        "current_date": pd.Timestamp.now().normalize(),
    }


def save_model_wrapper(model_wrapper: dict) -> None:
    joblib.dump(model_wrapper["ranker"], MODEL_PATH)


def ensure_ranker(model_wrapper: dict):
    if model_wrapper.get("ranker") is not None:
        return model_wrapper["ranker"]
    if MODEL_PATH.exists():
        try:
            model_wrapper["ranker"] = joblib.load(MODEL_PATH)
            add_log("Loaded ranker weights from model.pkl.", "SYSTEM")
            return model_wrapper["ranker"]
        except Exception as exc:
            add_log(f"Failed to load model.pkl: {exc}", "WARN")
    if LEGACY_WRAPPER_PATH.exists():
        try:
            legacy_loaded = joblib.load(LEGACY_WRAPPER_PATH)
            if isinstance(legacy_loaded, dict):
                model_wrapper["ranker"] = legacy_loaded.get("ranker")
            elif hasattr(legacy_loaded, "predict"):
                model_wrapper["ranker"] = legacy_loaded
            if model_wrapper.get("ranker") is not None:
                save_model_wrapper(model_wrapper)
                add_log("Migrated legacy wrapper into model.pkl.", "SYSTEM")
                return model_wrapper["ranker"]
        except Exception as exc:
            add_log(f"Legacy wrapper migration failed: {exc}", "WARN")
    if LEGACY_MODEL_PATH.exists():
        try:
            model_wrapper["ranker"] = joblib.load(LEGACY_MODEL_PATH)
            save_model_wrapper(model_wrapper)
            add_log("Loaded legacy LightGBM ranker into model.pkl.", "SYSTEM")
            return model_wrapper["ranker"]
        except Exception as exc:
            add_log(f"Legacy ranker migration failed: {exc}", "WARN")
    return None


def load_data(file_path: str) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()
    if ext == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if ext == ".json":
        return pd.read_json(file_path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def process_chunk(model_wrapper: dict, df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "redrob_signals" in frame.columns:
        signals_list = [parse_json_field(item[1]) for item in frame["redrob_signals"].items()]
    else:
        signals_list = [{} for _ in range(len(frame))]

    signals_df = pd.DataFrame(signals_list, index=frame.index)
    signal_keys = [
        "last_active_date",
        "recruiter_response_rate",
        "offer_acceptance_rate",
        "github_activity_score",
        "profile_completeness_score",
        "skill_assessment_scores",
    ]
    for key in signal_keys:
        if key in signals_df.columns:
            signal_series = signals_df[key]
        elif key in frame.columns:
            signal_series = frame[key]
        else:
            signal_series = pd.Series(np.nan, index=frame.index)
        signals_df[key] = signal_series

    frame["last_active_date"] = signals_df["last_active_date"]
    frame["recruiter_response_rate"] = signals_df["recruiter_response_rate"].fillna(0.0)
    frame["offer_acceptance_rate"] = signals_df["offer_acceptance_rate"].fillna(0.5)

    if "github_activity_score" in signals_df.columns:
        frame["github_activity_score"] = signals_df["github_activity_score"].fillna(0.0).clip(lower=0)
    elif "github_activity_score_norm" in frame.columns:
        frame["github_activity_score"] = frame["github_activity_score_norm"] * 100.0
    else:
        frame["github_activity_score"] = 0.0

    if "profile_completeness" in signals_df.columns:
        frame["profile_completeness"] = signals_df["profile_completeness_score"].fillna(0.0)
    elif "profile_completeness_score_norm" in frame.columns:
        frame["profile_completeness"] = frame["profile_completeness_score_norm"] * 100.0
    elif "profile_completeness" in frame.columns:
        frame["profile_completeness"] = frame["profile_completeness"].fillna(0.0)
    else:
        frame["profile_completeness"] = 0.0

    def fast_avg_assessment(scores_dict):
        if isinstance(scores_dict, str):
            scores_dict = parse_json_field(scores_dict)
        if not isinstance(scores_dict, dict) or not scores_dict:
            return 0.0
        try:
            return (sum(scores_dict.values()) / len(scores_dict)) / 100.0
        except Exception:
            return 0.0

    frame["skill_assessment_scores_norm"] = signals_df["skill_assessment_scores"].apply(fast_avg_assessment)
    frame["last_active_date"] = pd.to_datetime(frame["last_active_date"], errors="coerce")
    frame["days_since_active"] = (model_wrapper["current_date"] - frame["last_active_date"]).dt.days.fillna(9999)
    frame["days_since_active"] = frame["days_since_active"].clip(lower=0)
    frame["recency_multiplier"] = np.exp(-frame["days_since_active"] / 365.0)
    frame["profile_completeness_score_norm"] = frame["profile_completeness"] / 100.0
    frame["github_activity_score_norm"] = frame["github_activity_score"] / 100.0

    def fast_skill_match(skills_val):
        parsed = parse_skills_field(skills_val)
        count = 0
        for skill in parsed:
            if isinstance(skill, dict):
                name = str(skill.get("name", "")).lower()
            else:
                name = str(skill).lower()
            if any(target in name for target in model_wrapper["target_skills"]):
                count += 1
        return min(count / 5.0, 1.0)

    if "skills" in frame.columns:
        frame["skill_match_score"] = frame["skills"].apply(fast_skill_match)
    else:
        frame["skill_match_score"] = 0.0

    frame["target_relevance_score"] = (
        (frame["skill_match_score"] * 0.4)
        + (frame["recruiter_response_rate"] * 0.3)
        + (frame["github_activity_score_norm"] * 0.3)
    ) * frame["recency_multiplier"]

    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = [f"CAND_{i:06d}" for i in range(len(frame))]

    frame["candidate_name"] = frame.apply(extract_candidate_name, axis=1)

    return frame[model_wrapper["feature_columns"] + ["candidate_id", "candidate_name", "target_relevance_score"]]


def build_context_tags(model_wrapper: dict, rng: random.Random, skills_pct: float, github_score: float, resp_rate: float, comp_score: float, days_active: int):
    if skills_pct >= 100:
        skill_tier = "complete"
    elif skills_pct >= 80:
        skill_tier = "strong"
    elif skills_pct >= 60:
        skill_tier = "moderate"
    else:
        skill_tier = "low"

    if github_score >= 85:
        github_tier = "exceptional"
    elif github_score >= 65:
        github_tier = "consistent"
    elif github_score >= 40:
        github_tier = "moderate"
    else:
        github_tier = "limited"

    if resp_rate >= 80:
        response_tier = "high"
    elif resp_rate >= 50:
        response_tier = "medium"
    else:
        response_tier = "low"

    if days_active < 30:
        recency_tier = "active_now"
    elif days_active < 90:
        recency_tier = "recent_quarter"
    elif days_active < 180:
        recency_tier = "stale_6mo"
    else:
        recency_tier = "inactive"

    if comp_score >= 90:
        profile_tier = "complete"
    elif comp_score >= 70:
        profile_tier = "good"
    else:
        profile_tier = "partial"

    return {
        "skill_tag": _pick(rng, model_wrapper["skill_tags"][skill_tier]),
        "github_tag": _pick(rng, model_wrapper["github_tags"][github_tier]),
        "resp_tag": _pick(rng, model_wrapper["resp_tags"][response_tier]),
        "recency_tag": _pick(rng, model_wrapper["recency_tags"][recency_tier]),
        "profile_tag": _pick(rng, model_wrapper["profile_tags"][profile_tier]),
    }


def build_prompt(model_wrapper: dict, ctx):
    return (
        "Complete this short, direct hiring recommendation for a Founding AI Engineer candidate. "
        "Write exactly 3 short sentences: one on skills, one on GitHub evidence, one final hire verdict. "
        "Do not repeat these instructions. Do not use numbers or labels.\n"
        f"Facts: Skills - {ctx['skill_tag']}. GitHub - {ctx['github_tag']}. Availability - {ctx['recency_tag']}. "
        f"Recruiter responsiveness - {ctx['resp_tag']}.\n"
        "Recommendation: This candidate"
    )


def build_fallback(model_wrapper: dict, rng: random.Random, ctx):
    skill_line = _pick(rng, model_wrapper["skill_lines"]).format(**ctx)
    evidence_line = _pick(rng, model_wrapper["evidence_lines"]).format(**ctx)
    verdict_line = _pick(rng, model_wrapper["verdict_lines"]).format(**ctx)
    return " ".join([skill_line, evidence_line, verdict_line])


def is_valid_reasoning(model_wrapper: dict, text: str) -> bool:
    if not text or len(text) < model_wrapper["min_valid_length"]:
        return False
    sentence_count = len(re.findall(r"[.!?]+", text))
    if sentence_count < model_wrapper["min_sentences"] or sentence_count > model_wrapper["max_sentences"]:
        return False
    if model_wrapper["echo_patterns"].search(text):
        return False
    if model_wrapper["junk_patterns"].search(text):
        return False
    if model_wrapper["malformed_patterns"].search(text):
        return False
    if model_wrapper["identifier_leak_patterns"].search(text):
        return False
    if not model_wrapper["on_topic_patterns"].search(text):
        return False
    return True


def _reasoning_engine():
    global _TOKENIZER, _SEQ2SEQ_MODEL, _REASONING_READY
    if _REASONING_READY:
        return _TOKENIZER, _SEQ2SEQ_MODEL
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        add_log(f"Loading FLAN-T5 reasoning engine on {device}.", "SYSTEM")
        
        load_kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            
        # Try local system load first; fallback seamlessly to online retrieval if unavailable
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
            _SEQ2SEQ_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True, **load_kwargs)
        except Exception:
            add_log("Local FLAN-T5 checkpoints missing. Fetching from Hugging Face Hub...", "INFO")
            _TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=False)
            _SEQ2SEQ_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=False, **load_kwargs)

        if device != "cuda":
            _SEQ2SEQ_MODEL = _SEQ2SEQ_MODEL.to(device)
        _SEQ2SEQ_MODEL.eval()
    except Exception as exc:
        add_log(f"Reasoning engine unavailable, using templates only: {exc}", "WARN")
        _TOKENIZER = None
        _SEQ2SEQ_MODEL = None
    _REASONING_READY = True
    return _TOKENIZER, _SEQ2SEQ_MODEL


def generate_batch_reasoning(model_wrapper: dict, batch_rows: pd.DataFrame) -> list[str]:
    tokenizer, seq2seq_model = _reasoning_engine()
    prompts = []
    fallbacks = []

    for _, row in batch_rows.iterrows():
        candidate_id = str(row.get("candidate_id", ""))
        skills_pct = float(row.get("skill_match_score", 0.0)) * 100
        github_score = float(row.get("github_activity_score_norm", 0.0)) * 100
        resp_rate = float(row.get("recruiter_response_rate", 0.0)) * 100
        comp_score = float(row.get("profile_completeness_score_norm", 0.0)) * 100
        days_active = int(row.get("days_since_active", 9999))
        rng = random.Random(candidate_id)
        ctx = build_context_tags(model_wrapper, rng, skills_pct, github_score, resp_rate, comp_score, days_active)
        prompts.append(build_prompt(model_wrapper, ctx))
        fallbacks.append(build_fallback(model_wrapper, rng, ctx))

    if tokenizer is None or seq2seq_model is None:
        return fallbacks

    try:
        import torch

        device = next(seq2seq_model.parameters()).device
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=180,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = seq2seq_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        results = []
        for index, output in enumerate(outputs):
            reasoning = tokenizer.decode(output, skip_special_tokens=True).strip()
            if reasoning and not reasoning[0].isupper():
                reasoning = "This candidate " + reasoning
            reasoning = re.sub(r"Sentence\s*\d+\s*[:\-]?\s*", "", reasoning, flags=re.IGNORECASE)
            reasoning = re.sub(r"^\d+[\.)]\s*", "", reasoning, flags=re.MULTILINE)
            reasoning = re.sub(r"\*\*|__|##|---", "", reasoning)
            reasoning = re.sub(r"<[^>]+>", "", reasoning)
            reasoning = re.sub(r"\n+", " ", reasoning)
            reasoning = re.sub(r"\s{2,}", " ", reasoning).strip()
            results.append(reasoning if is_valid_reasoning(model_wrapper, reasoning) else fallbacks[index])
        return results
    except Exception as exc:
        add_log(f"Batch inference fallback activated: {exc}", "WARN")
        return fallbacks


def train_ranker(model_wrapper: dict, X: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
    )
    total_rows = len(X)
    group_size = 500 if total_rows >= 500 else max(1, total_rows)
    full_groups = total_rows // group_size
    remainder = total_rows % group_size
    groups = [group_size] * full_groups
    if remainder:
        groups.append(remainder)
    if not groups:
        groups = [1]
    ranker.fit(X, y, group=groups)
    model_wrapper["ranker"] = ranker
    save_model_wrapper(model_wrapper)
    return ranker


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _update_cache(submission: pd.DataFrame) -> None:
    global submission_cache
    output_path = RESULT_DIR / DOWNLOAD_FILENAME
    if output_path.exists():
        stat = output_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    else:
        signature = (time.time_ns(), len(submission))
    with state_lock:
        submission_cache = {"signature": signature, "rows": submission.to_dict(orient="records")}


def leaderboard_score_scale(scores: pd.Series, low: float = 3.0, high: float = 10.0) -> pd.Series:
    if scores.empty:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return pd.Series([low] * len(scores), index=scores.index, dtype=float)
    if abs(maximum - minimum) < 1e-12:
        midpoint = (low + high) / 2.0
        return pd.Series([midpoint] * len(scores), index=scores.index, dtype=float)
    scaled = low + ((scores - minimum) / (maximum - minimum)) * (high - low)
    return scaled.astype(float)


def process_and_save(model_wrapper: dict, input_path: str, output_csv_path: str) -> pd.DataFrame:
    add_log(f"Processing uploaded dataset: {Path(input_path).name}", "SYSTEM")
    raw_df = load_data(input_path)
    processed_df = process_chunk(model_wrapper, raw_df)
    feature_columns = model_wrapper["feature_columns"]
    X = processed_df[feature_columns]
    y = (processed_df["target_relevance_score"] * 10).astype(int)

    ranker = ensure_ranker(model_wrapper)
    if ranker is None:
        add_log("No consolidated ranker found. Training a new one from the uploaded dataset.", "SYSTEM")
        ranker = train_ranker(model_wrapper, X, y)

    predicted = ranker.predict(X)
    processed_df = processed_df.copy()
    processed_df["predicted_score"] = predicted

    top_n = min(100, len(processed_df))
    top_rows = processed_df.sort_values(by="predicted_score", ascending=False).head(top_n).copy()
    top_rows["score"] = leaderboard_score_scale(top_rows["predicted_score"])
    top_rows["rank"] = range(1, top_n + 1)

    batch_size = 20 if _gpu_available() else 5
    reasonings = []
    for batch_start in range(0, len(top_rows), batch_size):
        batch = top_rows.iloc[batch_start : batch_start + batch_size]
        reasonings.extend(generate_batch_reasoning(model_wrapper, batch))

    reasonings = [text if text and text.strip() else model_wrapper["default_reasoning"] for text in reasonings]
    seen = set()
    candidate_ids = top_rows["candidate_id"].astype(str).tolist()
    for index, (candidate_id, text) in enumerate(zip(candidate_ids, reasonings)):
        attempt = 0
        while text in seen and attempt < 20:
            rng = random.Random(f"{candidate_id}_reroll_{attempt}")
            row = top_rows.iloc[index]
            ctx = build_context_tags(
                model_wrapper,
                rng,
                float(row.get("skill_match_score", 0.0)) * 100,
                float(row.get("github_activity_score_norm", 0.0)) * 100,
                float(row.get("recruiter_response_rate", 0.0)) * 100,
                float(row.get("profile_completeness_score_norm", 0.0)) * 100,
                int(row.get("days_since_active", 9999)),
            )
            text = build_fallback(model_wrapper, rng, ctx)
            attempt += 1
        seen.add(text)
        reasonings[index] = text if is_valid_reasoning(model_wrapper, text) else model_wrapper["default_reasoning"]

    top_rows["reasoning"] = reasonings
    submission = top_rows[["candidate_name", "candidate_id", "rank", "score", "reasoning"]]
    submission.to_csv(output_csv_path, index=False, float_format="%.12f")
    _update_cache(submission)
    add_log(f"Saved ranked output to {output_csv_path}", "SUCCESS")
    return submission


def load_unified_model() -> dict:
    if MODEL_PATH.exists():
        try:
            ranker = joblib.load(MODEL_PATH)
            add_log("Loaded ranker weights from model.pkl.", "SYSTEM")
            return build_model_wrapper(ranker=ranker)
        except Exception as exc:
            add_log(f"model.pkl load failed, bootstrapping a fresh runtime container: {exc}", "WARN")

    model_wrapper = build_model_wrapper()
    if LEGACY_WRAPPER_PATH.exists():
        try:
            legacy_loaded = joblib.load(LEGACY_WRAPPER_PATH)
            if isinstance(legacy_loaded, dict):
                model_wrapper["ranker"] = legacy_loaded.get("ranker")
            elif hasattr(legacy_loaded, "predict"):
                model_wrapper["ranker"] = legacy_loaded
            if model_wrapper["ranker"] is not None:
                save_model_wrapper(model_wrapper)
                add_log("Migrated legacy wrapper into model.pkl.", "SYSTEM")
                return model_wrapper
        except Exception as exc:
            add_log(f"Could not migrate legacy wrapper: {exc}", "WARN")
    if LEGACY_MODEL_PATH.exists() and model_wrapper.get("ranker") is None:
        try:
            model_wrapper["ranker"] = joblib.load(LEGACY_MODEL_PATH)
            save_model_wrapper(model_wrapper)
            add_log("Loaded legacy ranker into model.pkl.", "SYSTEM")
        except Exception as exc:
            add_log(f"Could not migrate legacy ranker: {exc}", "WARN")
    if model_wrapper.get("ranker") is not None:
        save_model_wrapper(model_wrapper)
    return model_wrapper


model_wrapper = None


def current_submission_rows() -> list[dict]:
    submission_path = RESULT_DIR / DOWNLOAD_FILENAME
    if not submission_path.exists():
        return []
    try:
        stat = submission_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with state_lock:
            if submission_cache["signature"] == signature:
                return list(submission_cache["rows"])
        df = pd.read_csv(submission_path)
        rows = df.to_dict(orient="records")
        with state_lock:
            submission_cache["signature"] = signature
            submission_cache["rows"] = rows
        return rows
    except Exception as exc:
        add_log(f"Could not read submission.csv: {exc}", "ERROR")
        return []


def process_uploaded_file(file_path: str) -> None:
    global is_processing
    resolved = str(Path(file_path).resolve())
    file_name = Path(resolved).name
    if Path(resolved).suffix.lower() not in SUPPORTED_DATA_EXTS:
        return
    if resolved in processed_files:
        return

    with state_lock:
        is_processing = True
    try:
        output_path = RESULT_DIR / DOWNLOAD_FILENAME
        submission = process_and_save(model_wrapper, resolved, str(output_path))
        processed_files.add(resolved)
        add_log(f"Pipeline finished for {file_name} with {len(submission)} ranked candidates.", "SUCCESS")
    except Exception as exc:
        add_log(f"Failed to process {file_name}: {exc}", "ERROR")
    finally:
        with state_lock:
            is_processing = False


def pipeline_worker() -> None:
    add_log("Background pipeline worker started.", "SYSTEM")
    while True:
        file_path = task_queue.get()
        if file_path is None:
            task_queue.task_done()
            return
        try:
            process_uploaded_file(file_path)
        finally:
            task_queue.task_done()

# --- Utility to ensure file is not held open ---
def force_delete(file_path: Path):
    """Attempt to delete a file by clearing caches and forcing GC."""
    if not file_path.exists():
        return True
    try:
        # Force Python's garbage collector to close potential file handles
        gc.collect()
        file_path.unlink()
        return True
    except PermissionError:
        # If still locked, return error
        return False


class DataCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path).resolve()
        if path.suffix.lower() not in SUPPORTED_DATA_EXTS:
            return
        if str(path) in processed_files:
            return
        add_log(f"Detected new dataset file: {path.name}", "SYSTEM")
        time.sleep(1.5)
        task_queue.put(str(path))


def start_watchdog() -> None:
    global observer, watcher_active
    handler = DataCreatedHandler()
    observer = Observer()
    observer.schedule(handler, str(DATA_DIR), recursive=False)
    observer.daemon = True
    observer.start()
    watcher_active = True
    add_log("Watchdog observer started for data/ (on_created only).", "SYSTEM")


def bootstrap_runtime() -> None:
    ensure_directories()
    global model_wrapper, model_ready
    model_wrapper = load_unified_model()
    model_ready = model_wrapper.get("ranker") is not None
    threading.Thread(target=pipeline_worker, daemon=True).start()
    start_watchdog()

    # Process any files already sitting in data/ at startup instead of
    # silently marking them as done without ever running the pipeline
    for existing in DATA_DIR.iterdir():
        if existing.is_file() and existing.suffix.lower() in SUPPORTED_DATA_EXTS:
            add_log(f"Found existing dataset file at startup: {existing.name}", "SYSTEM")
            task_queue.put(str(existing.resolve()))


def dashboard_payload() -> dict:
    rows = current_submission_rows()
    top_three = rows[:3]
    submission_path = RESULT_DIR / DOWNLOAD_FILENAME
    submission_updated_at = None
    if submission_path.exists():
        submission_updated_at = datetime.fromtimestamp(submission_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "watcher_active": watcher_active,
        "model_ready": model_wrapper is not None and model_wrapper.get("ranker") is not None,
        "processing": is_processing,
        "logs": list(log_lines),
        "submission_exists": (RESULT_DIR / DOWNLOAD_FILENAME).exists(),
        "submission_updated_at": submission_updated_at,
        "rows": rows,
        "top_three": top_three,
        "count": len(rows),
    }


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Recruiting Podium Viewer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Plus Jakarta Sans', 'ui-sans-serif', 'system-ui']
          },
          boxShadow: {
            soft: '0 20px 60px rgba(15, 23, 42, 0.16)'
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(251, 191, 36, 0.14), transparent 24%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .sheet-grid {
      background-image:
        linear-gradient(rgba(148, 163, 184, 0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.12) 1px, transparent 1px);
      background-size: 100% 44px, 160px 100%;
    }
    .podium-shadow { box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18); }
    .podium-gold { background: linear-gradient(180deg, #f8d46c 0%, #f59e0b 100%); }
    .podium-silver { background: linear-gradient(180deg, #e5e7eb 0%, #94a3b8 100%); }
    .podium-bronze { background: linear-gradient(180deg, #fdba74 0%, #ea580c 100%); }
  </style>
</head>
<body class="text-slate-900">
  <div class="mx-auto max-w-7xl px-4 py-6 lg:px-8">
    <div class="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-soft backdrop-blur-xl">
      <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">Dynamic Recruiting Pipeline</p>
          <h1 class="mt-2 text-3xl font-extrabold tracking-tight text-slate-950 sm:text-4xl">Google Sheets-style candidate viewer</h1>
          <p class="mt-2 max-w-3xl text-sm text-slate-600">Drop a file into <span class="font-semibold">data/</span> and the watchdog observer will rank candidates, generate reasoning, and refresh the sheet automatically when processing completes.</p>
        </div>
        <div class="flex flex-wrap gap-3">
          <a id="downloadBtn" href="/download/submission.csv" class="inline-flex items-center rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800">Download submission.csv</a>
          <button id="refreshBtn" class="inline-flex items-center rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50">Refresh now</button>
        </div>
      </div>

      <div class="mt-6 grid gap-4 md:grid-cols-4">
        <div class="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Watcher</div>
          <div id="watcherState" class="mt-2 text-lg font-bold">Starting...</div>
        </div>
        <div class="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Model</div>
          <div id="modelState" class="mt-2 text-lg font-bold">Checking...</div>
        </div>
        <div class="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Rows</div>
          <div id="rowState" class="mt-2 text-lg font-bold">0</div>
        </div>
        <div class="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Pipeline</div>
          <div id="pipelineState" class="mt-2 text-lg font-bold">Idle</div>
        </div>
      </div>

      <div class="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
        <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h2 class="text-lg font-bold">3D Podium Leaderboard</h2>
            <p class="text-sm text-slate-500">Search scans names, candidate IDs, and reasoning; the podium and table both react instantly.</p>
          </div>
          <input id="searchInput" type="search" placeholder="Search names, IDs, or reasoning..." class="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-slate-400 lg:max-w-md" />
        </div>
        <div id="podiumWrap" class="mt-5 flex items-end justify-center gap-3 overflow-x-auto pb-2"></div>
      </div>

      <div class="mt-6 rounded-3xl border border-slate-200 bg-white shadow-sm overflow-hidden">
  <div class="sheet-grid overflow-auto">
    <table class="min-w-full table-fixed border-separate border-spacing-0 text-left text-sm">
      <colgroup>
        <col style="width: 70px" />
        <col style="width: 180px" />
        <col style="width: 150px" />
        <col style="width: 90px" />
        <col />
      </colgroup>
      <thead class="sticky top-0 z-10 bg-slate-950 text-white">
        <tr>
          <th class="px-4 py-3 font-semibold">Rank</th>
          <th class="px-4 py-3 font-semibold">Candidate</th>
          <th class="px-4 py-3 font-semibold">Candidate ID</th>
          <th class="px-4 py-3 font-semibold">Score</th>
          <th class="px-4 py-3 font-semibold">Reasoning</th>
        </tr>
      </thead>
      <tbody id="candidateBody" class="divide-y divide-slate-200 bg-white"></tbody>
    </table>
  </div>
  <div class="flex flex-col gap-3 border-t border-slate-200 bg-slate-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
    <div class="text-sm text-slate-600"><span id="pageSummary">0 rows</span></div>
    <div class="flex items-center gap-2">
      <button id="prevPage" class="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 disabled:opacity-40">Previous</button>
      <span id="pageState" class="min-w-[120px] text-center text-sm font-semibold text-slate-700">Page 1 / 1</span>
      <button id="nextPage" class="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 disabled:opacity-40">Next</button>
    </div>
  </div>
</div>

        <div class="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 class="text-lg font-bold">Status feed</h2>
          <div class="mt-3 max-h-[420px] overflow-auto rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-600" id="logFeed"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = {
      rows: [],
      filteredRows: [],
      page: 1,
      pageSize: 15,
      lastSignature: null,
      lastProcessing: false,
      searchQuery: '',
    };

    const elements = {
      watcherState: document.getElementById('watcherState'),
      modelState: document.getElementById('modelState'),
      rowState: document.getElementById('rowState'),
      pipelineState: document.getElementById('pipelineState'),
      searchInput: document.getElementById('searchInput'),
      podiumWrap: document.getElementById('podiumWrap'),
      logFeed: document.getElementById('logFeed'),
      candidateBody: document.getElementById('candidateBody'),
      pageSummary: document.getElementById('pageSummary'),
      pageState: document.getElementById('pageState'),
      prevPage: document.getElementById('prevPage'),
      nextPage: document.getElementById('nextPage'),
      refreshBtn: document.getElementById('refreshBtn'),
      downloadBtn: document.getElementById('downloadBtn'),
    };

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function scoreText(score) {
      return (score === null || score === undefined || score === '') ? '0' : String(score);
    }

    function podiumTone(rank) {
      if (rank === 1) return 'podium-gold';
      if (rank === 2) return 'podium-silver';
      return 'podium-bronze';
    }

    function displayName(row) {
      return row.candidate_name || row.full_name || row.display_name || row.name || (row.candidate_id ? `Candidate ${row.candidate_id}` : 'Anonymous');
    }

    function renderPodium(items) {
      if (!items.length) {
        elements.podiumWrap.innerHTML = '<div class="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-500">No submission has been generated yet.</div>';
        return;
      }
      const rankMap = new Map(items.slice(0, 3).map((row) => [Number(row.rank), row]));
      const podiumSlots = [2, 1, 3].map((rank) => rankMap.get(rank)).filter(Boolean);
      elements.podiumWrap.innerHTML = podiumSlots.map((row) => {
        const rank = Number(row.rank ?? 0);
        const heightClass = rank === 1 ? 'h-[360px]' : rank === 2 ? 'h-[286px]' : 'h-[238px]';
        const medalLabel = rank === 1 ? 'Gold Crown' : rank === 2 ? 'Silver Medal' : 'Bronze Award';
        const badge = rank === 1 ? 'Crown' : rank === 2 ? 'Silver' : 'Bronze';
        return `
          <div class="w-[260px] shrink-0 ${rank === 1 ? 'order-2' : rank === 2 ? 'order-1' : 'order-3'}">
            <div class="mb-3 flex justify-center">
              <span class="rounded-full px-4 py-1 text-xs font-extrabold uppercase tracking-[0.24em] ${rank === 1 ? 'bg-amber-400 text-white' : rank === 2 ? 'bg-slate-300 text-slate-900' : 'bg-orange-300 text-white'}">${badge} ${medalLabel}</span>
            </div>
            <div class="${heightClass} podium-shadow ${podiumTone(rank)} rounded-t-[2rem] rounded-b-[1.4rem] p-4 text-white transform ${rank === 1 ? 'translate-y-0' : rank === 2 ? 'translate-y-6' : 'translate-y-10'}">
              <div class="flex h-full flex-col justify-between">
                <div>
                  <div class="text-3xl font-black leading-none">${rank}</div>
                  <div class="mt-2 text-sm font-semibold uppercase tracking-[0.25em] opacity-90">${escapeHtml(medalLabel)}</div>
                </div>
                <div class="rounded-2xl bg-white/18 p-4 backdrop-blur-sm">
                  <div class="text-xl font-extrabold leading-tight">${escapeHtml(displayName(row))}</div>
                  <div class="mt-1 text-xs font-semibold uppercase tracking-[0.22em] opacity-90">Candidate ID: ${escapeHtml(row.candidate_id ?? '')}</div>
                  <div class="mt-3 text-sm font-bold">Score: ${escapeHtml(scoreText(row.score))}</div>
                  <p class="mt-3 text-sm leading-6 text-white/95">${escapeHtml(row.reasoning ?? '')}</p>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function filterRows(resetPage = true) {
  state.searchQuery = elements.searchInput.value.trim().toLowerCase();
  const query = state.searchQuery;
  state.filteredRows = state.rows.filter((row) => {
    if (!query) return true;
    const candidateId = String(row.candidate_id ?? '').toLowerCase();
    const candidateName = String(row.candidate_name || row.full_name || row.display_name || row.name || '').toLowerCase();
    const reasoning = String(row.reasoning ?? '').toLowerCase();
    return candidateId.includes(query) || candidateName.includes(query) || reasoning.includes(query);
  });
  if (resetPage) {
    state.page = 1;
  }
  renderLeaderboard();
}

    function renderLeaderboard() {
      const podiumRows = state.filteredRows.slice(0, 3);
      const tableSource = state.filteredRows.slice(3, 100);
      renderPodium(podiumRows);
      const total = tableSource.length;
      const totalPages = Math.max(1, Math.ceil(total / state.pageSize));
      state.page = Math.min(state.page, totalPages);
      const start = (state.page - 1) * state.pageSize;
      const pageRows = tableSource.slice(start, start + state.pageSize);

      elements.candidateBody.innerHTML = pageRows.length ? pageRows.map((row) => {
        const rank = Number(row.rank ?? 0);
        return `
          <tr class="transition hover:bg-slate-50 ${rank <= 3 ? 'bg-amber-50/60' : ''}">
            <td class="px-4 py-4 align-top">
              <span class="inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-700">${escapeHtml(rank)}</span>
            </td>
            <td class="px-4 py-4 align-top font-semibold text-slate-900">${escapeHtml(displayName(row))}</td>
            <td class="px-4 py-4 align-top font-semibold text-slate-700">${escapeHtml(row.candidate_id ?? '')}</td>
            <td class="px-4 py-4 align-top font-mono text-slate-700">${escapeHtml(scoreText(row.score))}</td>
            <td class="px-4 py-4 align-top text-slate-700 leading-6">${escapeHtml(row.reasoning ?? '')}</td>
          </tr>
        `;
      }).join('') : `
        <tr>
          <td colspan="5" class="px-4 py-10 text-center text-sm text-slate-500">No candidates match the current search.</td>
        </tr>
      `;

      elements.pageSummary.textContent = total ? `Showing ${start + 1}-${Math.min(start + state.pageSize, total)} of ${total}` : '0 rows';
      elements.pageState.textContent = `Page ${state.page} / ${totalPages}`;
      elements.prevPage.disabled = state.page <= 1;
      elements.nextPage.disabled = state.page >= totalPages;
    }

    function renderLogs(logs) {
      if (!logs || !logs.length) {
        elements.logFeed.innerHTML = '<div class="text-slate-500">No log entries yet.</div>';
        return;
      }
      elements.logFeed.innerHTML = logs.slice(-80).map((line) => `<div class="mb-1 border-b border-slate-200/80 pb-1 last:border-0">${escapeHtml(line)}</div>`).join('');
      elements.logFeed.scrollTop = elements.logFeed.scrollHeight;
    }

    async function fetchStatus() {
      try {
        const response = await fetch('/api/check-status', { cache: 'no-store' });
        const data = await response.json();
        
        elements.watcherState.textContent = data.watcher_active ? 'Watching data/' : 'Offline';
        elements.modelState.textContent = data.model_ready ? 'Ready' : 'Bootstrapping';
        elements.rowState.textContent = String(data.count ?? 0);
        elements.pipelineState.textContent = data.processing ? 'Processing' : 'Idle';
        
        elements.pipelineState.className = 'mt-2 text-lg font-bold ' + (data.processing ? 'text-amber-600' : 'text-emerald-600');
        elements.watcherState.className = 'mt-2 text-lg font-bold ' + (data.watcher_active ? 'text-emerald-600' : 'text-rose-600');
        elements.modelState.className = 'mt-2 text-lg font-bold ' + (data.model_ready ? 'text-emerald-600' : 'text-amber-600');
        
        renderLogs(data.logs || []);
        state.rows = data.rows || [];
        
        const signature = JSON.stringify([data.count || 0, data.processing || false, data.submission_updated_at || '']);
        const changed = signature !== state.lastSignature;
        state.lastSignature = signature;
        
        if (changed && !data.processing) {
          elements.downloadBtn.classList.remove('opacity-60');
        }
        
        filterRows(false);   // recompute filtered rows on every poll, but keep the user's current page
        state.lastProcessing = !!data.processing;
      } catch (error) {
        elements.pipelineState.textContent = 'Disconnected';
        elements.pipelineState.className = 'mt-2 text-lg font-bold text-rose-600';
      }
    }

    elements.searchInput.addEventListener('input', filterRows);
    elements.prevPage.addEventListener('click', () => {
      state.page = Math.max(1, state.page - 1);
      renderLeaderboard();
    });
    elements.nextPage.addEventListener('click', () => {
      const totalPages = Math.max(1, Math.ceil(Math.max(0, state.filteredRows.slice(3, 100).length) / state.pageSize));
      state.page = Math.min(totalPages, state.page + 1);
      renderLeaderboard();
    });
    elements.refreshBtn.addEventListener('click', fetchStatus);

    fetchStatus();
    setInterval(fetchStatus, 2000);
    async function deleteSubmission() {
        await fetch('/api/delete_result', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: 'submission.csv', folder: 'result'})
        });
        location.reload();
    }
    async function resetSystem() {
        await fetch('/api/reset_all', {method: 'POST'});
        location.reload();
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file_obj = request.files["file"]
    if file_obj.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400
    filename = secure_filename(file_obj.filename)
    if Path(filename).suffix.lower() not in SUPPORTED_DATA_EXTS:
        allowed_types = ", ".join(sorted(ext for ext in SUPPORTED_DATA_EXTS if ext != ".json"))
        return jsonify({"error": f"Unsupported file type. Allowed types: {allowed_types}"}), 400
    target_path = DATA_DIR / filename
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        filename = f"{stem}_{int(time.time())}{suffix}"
        target_path = DATA_DIR / filename
    file_obj.save(str(target_path))
    add_log(f"Manual upload saved to data/: {filename}", "INFO")
    return jsonify({"success": True, "filename": filename}), 200


@app.route("/api/check-status", methods=["GET"])
def check_status():
    return jsonify(dashboard_payload())


@app.route("/status", methods=["GET"])
def legacy_status():
    return jsonify(dashboard_payload())


@app.route("/preview", methods=["GET"])
def preview():
    rows = current_submission_rows()
    return jsonify({"status": "success" if rows else "empty", "candidates": rows[:10]})


@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    global log_lines
    with state_lock:
        log_lines = ["[SYSTEM] Console logs cleared by user."]
    return jsonify({"success": True})


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    safe_name = secure_filename(filename)
    if safe_name != DOWNLOAD_FILENAME:
        return "File not found", 404
    target = RESULT_DIR / safe_name
    if not target.exists():
        return "File not found", 404
    return send_file(str(target), as_attachment=True, download_name=safe_name)


# --- Utility to ensure file is not held open ---
def force_delete(file_path: Path):
    """Attempt to delete a file by clearing caches and forcing GC."""
    if not file_path.exists():
        return True
    try:
        # Force Python's garbage collector to close potential file handles
        gc.collect()
        file_path.unlink()
        return True
    except PermissionError:
        # If still locked, return error
        return False

# --- Management Routes ---

@app.route("/api/delete_result", methods=["POST"])
def delete_result():
    """Endpoint specifically for deleting the output file."""
    target = RESULT_DIR / DOWNLOAD_FILENAME
    
    # 1. Clear memory cache if you have one
    with state_lock:
        # Assuming you have a 'submission_cache' global as in your previous code
        global submission_cache
        submission_cache = {"signature": None, "rows": []}

    # 2. Attempt deletion
    if force_delete(target):
        return jsonify({"success": True, "message": "Result deleted successfully"})
    else:
        return jsonify({"success": False, "message": "File is currently in use by another process"}), 409

@app.route("/api/delete_input", methods=["POST"])
def delete_input():
    data = request.json
    filename = secure_filename(data.get("filename", ""))
    target = DATA_DIR / filename
    
    if force_delete(target):
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "File in use"}), 409

@app.route("/api/reset_all", methods=["POST"])
def reset_all():
    global submission_cache, processed_files, log_lines
    force_delete(RESULT_DIR / DOWNLOAD_FILENAME)
    with state_lock:
        submission_cache = {"signature": None, "rows": []}
        processed_files.clear()
        log_lines = ["[SYSTEM] Pipeline reset by user."]
    return jsonify({"success": True})

def main() -> None:
    bootstrap_runtime()
    add_log("Starting Flask server on http://0.0.0.0:5000", "SYSTEM")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False, use_reloader=False)


if __name__ == "__main__":
    main()