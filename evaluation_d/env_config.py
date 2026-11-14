"""只从指定的本地 .env 读取学校 key，不修改进程或系统环境。"""

from __future__ import annotations

from pathlib import Path


def load_api_key(env_path: str | Path) -> str:
    """支持 SCHOOL_API_KEY=value 或引号包围的值；不做变量/命令插值。"""
    try:
        lines = Path(env_path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise ValueError(
            "无法读取项目根 .env；请参考 .env.example 创建并填写 SCHOOL_API_KEY"
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
                raise ValueError(".env 的 SCHOOL_API_KEY 引号格式无效")
            value = raw[1:end]
        else:
            value = raw.split(" #", 1)[0].strip()
        values.append(value)
    if len(values) != 1:
        raise ValueError(".env 必须且只能定义一次 SCHOOL_API_KEY")
    key = values[0]
    if (
        not key
        or key == "your_school_api_key_here"
        or any(character.isspace() or character == "\0" for character in key)
    ):
        raise ValueError("请将 .env 的 SCHOOL_API_KEY 填为有效 key；不要保留占位值")
    return key
