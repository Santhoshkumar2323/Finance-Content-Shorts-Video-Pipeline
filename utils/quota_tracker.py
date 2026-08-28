import json
import os
from filelock import FileLock

import config
from utils.logger_setup import log_event

_LOCK_PATH = config.QUOTA_CONFIG_PATH + ".lock"


class QuotaExceededError(Exception):
    pass


def _load_quota() -> dict:
    if not os.path.exists(config.QUOTA_CONFIG_PATH):
        default = {
            "groq_daily_tokens": 0,
            "groq_max_limit": config.GROQ_DAILY_TOKEN_LIMIT,
        }
        _save_quota(default)
        return default

    with open(config.QUOTA_CONFIG_PATH, "r") as f:
        return json.load(f)


def _save_quota(data: dict) -> None:
    with open(config.QUOTA_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)


def check_quota_before_run(logger=None) -> None:
    with FileLock(_LOCK_PATH):
        quota = _load_quota()

    tokens_used = quota["groq_daily_tokens"]

    if tokens_used >= config.GROQ_DAILY_TOKEN_WARN_THRESHOLD:
        message = (
            f"Groq daily token usage ({tokens_used}) has reached the "
            f"warn threshold ({config.GROQ_DAILY_TOKEN_WARN_THRESHOLD}). "
            f"Halting run to avoid an unhandled quota error mid-pipeline."
        )
        if logger:
            log_event(logger, node="QuotaTracker", status="FAIL", beat="-", message=message)
        raise QuotaExceededError(message)

    if logger:
        log_event(
            logger, node="QuotaTracker", status="SUCCESS", beat="-",
            message=f"Quota check passed -- Groq tokens: {tokens_used}/{config.GROQ_DAILY_TOKEN_LIMIT}",
        )


def record_groq_usage(tokens_used: int) -> None:
    with FileLock(_LOCK_PATH):
        quota = _load_quota()
        quota["groq_daily_tokens"] += tokens_used
        _save_quota(quota)


def reset_daily_quota() -> None:
    default = {
        "groq_daily_tokens": 0,
        "groq_max_limit": config.GROQ_DAILY_TOKEN_LIMIT,
    }
    _save_quota(default)