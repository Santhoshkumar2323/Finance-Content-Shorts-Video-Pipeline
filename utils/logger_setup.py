"""
utils/logger_setup.py

Configures ONE logger with TWO file handlers (DEBUG-level full run log,
ERROR-level failure-only log). Console output is handled ENTIRELY by
explicit print() calls here -- NOT by a logging StreamHandler -- which
is what fixes the original noise problem: a StreamHandler at INFO level
was letting every per-beat log record leak to the terminal in its raw
file-formatted shape, in addition to our own deliberate print() calls,
causing duplicated/messy console output.

Also suppresses third-party library noise (torch/kokoro/huggingface_hub
warnings) up front, since those are unrelated to our own logging and
were adding to the clutter.
"""

import logging
import os
import sys
import threading
import warnings
from datetime import datetime
import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _build_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_logger(run_id: str = None) -> tuple[logging.Logger, str]:
    if run_id is None:
        run_id = _build_run_id()

    logger = logging.getLogger("finance_shorts_pipeline")
    logger.setLevel(logging.DEBUG) 

    if logger.handlers:
        return logger, run_id

    file_path = os.path.join(config.LOGS_DIR, f"run_{run_id}.log")
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] NODE:%(node)-11s STATUS:%(status)-8s beat=%(beat)-6s msg=\"%(message)s\"",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    error_path = os.path.join(config.LOGS_DIR, f"errors_{run_id}.log")
    error_handler = logging.FileHandler(error_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger, run_id


def log_event(logger: logging.Logger, node: str, status: str, message: str,
              beat: str = "-", console_message: str = None):
    extra = {"node": node, "status": status, "beat": beat}

    level = logging.DEBUG
    if status == "FAIL":
        level = logging.ERROR
    elif status in ("RETRY", "SUCCESS", "START"):
        level = logging.INFO if status != "RETRY" else logging.WARNING

    logger.log(level, message, extra=extra)

    if console_message:
        console_symbol = {
            "START": "⠋", "SUCCESS": "✓", "RETRY": "⚠", "FAIL": "✗"
        }.get(status, "•")
        print(f"{console_symbol} {console_message}")


_live_lock = threading.Lock()
_live_order = []  

def live_progress(label: str, current: int, total: int, done: bool = False) -> None:
    with _live_lock:
        text = f"✓ {label:<10} → done" if done else f"⠋ {label:<10} {current}/{total}"

        if label not in _live_order:
            _live_order.append(label)
            print(text)
            return

        idx = _live_order.index(label)
        rows_up = len(_live_order) - idx
        sys.stdout.write(f"\x1b[{rows_up}A\r\x1b[2K{text}\x1b[{rows_up}B\r")
        sys.stdout.flush()


def reset_live_progress() -> None:
    with _live_lock:
        _live_order.clear()