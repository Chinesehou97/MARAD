import os


RESULT_DIR = os.getenv("MARAD_RESULT_DIR") or "result"
TRAIN_RESULT_DIR = os.path.join(RESULT_DIR, "train")
INFER_RESULT_DIR = os.path.join(RESULT_DIR, "infer")
RULES_DIR = os.getenv("MARAD_RULES_DIR") or TRAIN_RESULT_DIR

AGENT_DIR = os.path.join(TRAIN_RESULT_DIR, "agent")
FACT_MEMORY_PATH = os.path.join(AGENT_DIR, "fact_memory.json")
RULE_MEMORY_PATH = os.path.join(AGENT_DIR, "rule_memory.jsonl")


def get_rules_path() -> str:
    return os.path.join(RULES_DIR, "rules.json")


def configure_run(mode: str, output_dir: str | None = None) -> str:
    """Choose an output directory without requiring a dataset layout."""
    global AGENT_DIR, FACT_MEMORY_PATH, RULE_MEMORY_PATH
    defaults = {"train": TRAIN_RESULT_DIR, "infer": INFER_RESULT_DIR,
                "stats": os.path.join(RESULT_DIR, "detection_stats")}
    if mode not in defaults:
        raise ValueError(f"Unknown mode: {mode}")
    directory = output_dir or defaults[mode]
    if mode == "train":
        AGENT_DIR = os.path.join(directory, "agent")
        FACT_MEMORY_PATH = os.path.join(AGENT_DIR, "fact_memory.json")
        RULE_MEMORY_PATH = os.path.join(AGENT_DIR, "rule_memory.jsonl")
    return directory


# Remote API settings are supplied by the caller's environment.
LLM_MODEL_DEFAULT = os.getenv("MARAD_LLM_MODEL", "")
LLM_BASE_URL_DEFAULT = os.getenv("MARAD_LLM_BASE_URL", "")
LLM_API_KEY_ENV = "MARAD_LLM_API_KEY"
