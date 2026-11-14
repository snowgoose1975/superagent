"""只读取测试创建的合成凭据，不接触真实项目 .env。"""

from __future__ import annotations

import os

import pytest
from evaluation_d.env_config import load_api_key


@pytest.mark.parametrize(
    "content",
    [
        "SCHOOL_API_KEY=synthetic=key\n",
        '# comment\nSCHOOL_API_KEY="synthetic=key" # note\n',
        "export SCHOOL_API_KEY='synthetic=key'\n",
        "\ufeffSCHOOL_API_KEY=synthetic=key # note\n",
    ],
)
def test_loads_key_without_exporting_or_executing(tmp_path, content):
    path = tmp_path / "synthetic.env"
    path.write_text(content, encoding="utf-8")
    before = dict(os.environ)
    assert load_api_key(path) == "synthetic=key"
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "content",
    [
        "# no key\n",
        "SCHOOL_API_KEY=\n",
        "SCHOOL_API_KEY=your_school_api_key_here\n",
        "SCHOOL_API_KEY='synthetic key'\n",
        "SCHOOL_API_KEY='synthetic-key\n",
        "SCHOOL_API_KEY=synthetic-key\nSCHOOL_API_KEY=another-key\n",
    ],
)
def test_invalid_key_configuration_has_sanitized_errors(tmp_path, content):
    path = tmp_path / "synthetic.env"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_api_key(path)
    assert "synthetic" not in str(error.value)


def test_missing_file_does_not_search_parent_or_create_file(tmp_path):
    missing = tmp_path / "child" / "synthetic.env"
    with pytest.raises(ValueError, match=".env.example"):
        load_api_key(missing)
    assert not missing.exists()
