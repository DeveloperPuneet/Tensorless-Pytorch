"""Automatic detection & normalization of "free-form" record shapes.

`data/loader.py` already handles the well-known ``{"text": "..."}`` shape
via ``_TEXT_FIELD_CANDIDATES``. This module widens that net considerably.
Real-world datasets encode text in all kinds of ways -- most commonly as
conversational turns for chat / instruction-tuning:

    {"user": "...", "bot": "..."}
    {"human": "...", "gpt": "..."}
    {"instruction": "...", "input": "...", "output": "..."}   (Alpaca)
    {"prompt": "...", "completion": "..."}
    {"messages": [{"role": "user", "content": "..."}, ...]}    (OpenAI chat)
    {"conversations": [{"from": "human", "value": "..."}, ...]} (ShareGPT)

...or as genuinely arbitrary key/value records that don't match *any*
known convention. Rather than forcing every dataset into one blessed
shape (or erroring out), Tensorless tries a cascade of increasingly
generic strategies and normalizes whatever it finds into plain text
suitable for language-model training:

  1. known single-text-field records          (handled in loader.py)
  2. known turn-list records ("messages" / "conversations" / ...)
  3. known flat conversational pairs (user/bot, instruction/output, ...)
  4. anything else -> flatten the record's fields into readable text

Every step here is best-effort and never raises; if nothing usable is
found, the caller falls back to treating the data as tabular.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Field names (lowercased) that commonly stand in for a given speaker role.
# Order matters only in that it documents intent; matching itself checks
# every alias.
_USER_ALIASES: Tuple[str, ...] = (
    "user", "human", "question", "prompt", "instruction", "input",
    "from_user", "query", "customer",
)
_ASSISTANT_ALIASES: Tuple[str, ...] = (
    "assistant", "bot", "ai", "gpt", "answer", "response", "output",
    "completion", "reply", "chatbot", "agent",
)
_SYSTEM_ALIASES: Tuple[str, ...] = ("system", "context", "system_prompt", "instructions")

# Keys that hold a list of conversational turns.
_TURN_LIST_KEYS: Tuple[str, ...] = ("messages", "conversations", "turns", "dialogue", "dialog", "chat")
# Within a single turn dict, the key that names the speaker...
_TURN_ROLE_KEYS: Tuple[str, ...] = ("role", "from", "speaker", "author")
# ...and the key that holds what they said.
_TURN_CONTENT_KEYS: Tuple[str, ...] = ("content", "value", "text", "message", "utterance")

_ROLE_NAME_TO_LABEL: Dict[str, str] = {}
for _name in _USER_ALIASES + ("user",):
    _ROLE_NAME_TO_LABEL[_name] = "User"
for _name in _ASSISTANT_ALIASES + ("assistant", "model"):
    _ROLE_NAME_TO_LABEL[_name] = "Assistant"
for _name in _SYSTEM_ALIASES:
    _ROLE_NAME_TO_LABEL[_name] = "System"

# Minimum fraction of a (sampled) dataset that must match a strategy
# before we trust that strategy for *every* record.
_MATCH_THRESHOLD = 0.8
_SAMPLE_SIZE = 50


def _label_for_role(role: Any) -> str:
    role_norm = str(role).strip().lower()
    if role_norm in _ROLE_NAME_TO_LABEL:
        return _ROLE_NAME_TO_LABEL[role_norm]
    cleaned = str(role).strip().title()
    return cleaned or "Speaker"


def _first_matching_field(record: Dict[str, Any], aliases: Tuple[str, ...]) -> Optional[str]:
    lower_keys = {str(k).lower(): k for k in record.keys()}
    for alias in aliases:
        if alias in lower_keys:
            key = lower_keys[alias]
            val = record[key]
            if isinstance(val, str) and val.strip():
                return key
    return None


def _find_turn_list_key(record: Dict[str, Any]) -> Optional[str]:
    lower_keys = {str(k).lower(): k for k in record.keys()}
    for cand in _TURN_LIST_KEYS:
        if cand in lower_keys and isinstance(record[lower_keys[cand]], list) and record[lower_keys[cand]]:
            return lower_keys[cand]
    return None


def _format_turn_list(turns: List[Any]) -> Optional[str]:
    lines: List[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        role, content = None, None
        for k, v in turn.items():
            kl = str(k).lower()
            if role is None and kl in _TURN_ROLE_KEYS:
                role = v
            elif content is None and kl in _TURN_CONTENT_KEYS and isinstance(v, str):
                content = v
        if not content or not content.strip():
            return None
        label = _label_for_role(role) if role is not None else "Speaker"
        lines.append(f"{label}: {content.strip()}")
    if not lines:
        return None
    return "\n".join(lines) + "\n"


def _format_flat_pair(record: Dict[str, Any]) -> Optional[str]:
    system_key = _first_matching_field(record, _SYSTEM_ALIASES)
    user_key = _first_matching_field(record, _USER_ALIASES)
    assistant_key = _first_matching_field(record, _ASSISTANT_ALIASES)
    if user_key is None or assistant_key is None or user_key == assistant_key:
        return None

    lines: List[str] = []
    if system_key is not None:
        lines.append(f"System: {record[system_key].strip()}")

    user_text = record[user_key].strip()
    # Alpaca-style {"instruction", "input", "output"}: fold a non-empty
    # "input" field into the user turn instead of silently dropping it.
    rest = {k: v for k, v in record.items() if k not in (user_key, assistant_key, system_key)}
    input_key = _first_matching_field(rest, ("input",))
    if input_key is not None:
        extra = record[input_key].strip()
        if extra:
            user_text = f"{user_text}\n{extra}"
    lines.append(f"User: {user_text}")
    lines.append(f"Assistant: {record[assistant_key].strip()}")
    return "\n".join(lines) + "\n"


def normalize_records_to_text(records: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Try, in order, to interpret `records` as chat/conversation-style
    data and return one formatted training text per record.

    Returns None if the records don't reliably match a known
    conversational shape, in which case the caller should try the
    tabular / generic-flatten paths instead.
    """
    if not records or not all(isinstance(r, dict) for r in records):
        return None
    sample = records[: min(_SAMPLE_SIZE, len(records))]

    # Strategy 1: turn-list records, e.g. "messages": [...] / "conversations": [...]
    votes = sum(1 for r in sample if _find_turn_list_key(r) is not None)
    if votes >= max(1, int(len(sample) * _MATCH_THRESHOLD)):
        texts: List[str] = []
        for r in records:
            key = _find_turn_list_key(r)
            formatted = _format_turn_list(r[key]) if key else None
            if formatted is None:
                break
            texts.append(formatted)
        else:
            return texts

    # Strategy 2: flat conversational pairs (user/bot, instruction/output, prompt/completion, ...)
    votes = sum(1 for r in sample if _format_flat_pair(r) is not None)
    if votes >= max(1, int(len(sample) * _MATCH_THRESHOLD)):
        texts = []
        for r in records:
            formatted = _format_flat_pair(r)
            if formatted is None:
                break
            texts.append(formatted)
        else:
            return texts

    return None


def looks_textual(records: List[Dict[str, Any]]) -> bool:
    """Heuristic used only as a *last resort*, after strategies 1/2 above
    have failed: does this record shape look more like free-form text
    than a structured table? If so, `flatten_records_to_text` kicks in
    instead of forcing the data through the tabular pipeline.
    """
    sample = records[: min(_SAMPLE_SIZE, len(records))]
    total_len = 0
    total_fields = 0
    long_fields = 0
    for r in sample:
        if not isinstance(r, dict):
            return False
        for v in r.values():
            if isinstance(v, str):
                total_fields += 1
                total_len += len(v)
                if len(v) > 60:
                    long_fields += 1
    if total_fields == 0:
        return False
    avg_len = total_len / total_fields
    return avg_len > 40 or (long_fields / total_fields) > 0.3


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _flatten_generic(value: Any, _depth: int = 0) -> str:
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            label = str(k).replace("_", " ").strip().title() or str(k)
            if isinstance(v, (dict, list)) and _depth < 2:
                nested = _flatten_generic(v, _depth + 1)
                if nested:
                    parts.append(f"{label}:\n{_indent(nested)}")
            elif v is not None and str(v).strip() != "":
                parts.append(f"{label}: {v}")
        return "\n".join(parts)
    if isinstance(value, list):
        items = [_flatten_generic(item, _depth + 1) for item in value]
        return "\n".join(i for i in items if i)
    return str(value)


def flatten_records_to_text(records: List[Dict[str, Any]]) -> List[str]:
    """Last-resort normalization: render each record as readable
    'Key: value' text, however unfamiliar its shape, so *something*
    trainable comes out of it instead of an error or a misclassified
    tabular dataset.
    """
    return [_flatten_generic(r) + "\n" for r in records]
