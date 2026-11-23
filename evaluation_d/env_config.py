"""Read the school API key from the project .env file."""

from __future__ import annotations

from pathlib import Path


def load_api_key(env_path: str | Path) -> str:
    """Parse SCHOOL_API_KEY as a literal value with optional quotes."""
    try:
        lines = Path(env_path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise ValueError(
            "Cannot read the project .env. See .env.example for SCHOOL_API_KEY setup."
        ) from None
    values = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw = stripped.split("=", 1)
        if name.strip().removeprefix("export ").strip() != "SCHOOL_API_KEY":
            continue
        raw = raw.strip()
        if raw.startswith(('"', "'")):
            end = raw.find(raw[0], 1)
            if end < 0 or (raw[end + 1 :].strip() and not raw[end + 1 :].strip().startswith("#")):
                raise ValueError("Invalid quoting for SCHOOL_API_KEY in .env")
            value = raw[1:end]
        else:
            value = raw.split(" #", 1)[0].strip()
        values.append(value)
    if len(values) != 1:
        raise ValueError(".env must define SCHOOL_API_KEY exactly once")
    key = values[0]
    if (
        not key
        or key == "your_school_api_key_here"
        or any(character.isspace() or character == "\0" for character in key)
    ):
        raise ValueError("Replace the SCHOOL_API_KEY placeholder in .env with a valid key")
    return key
