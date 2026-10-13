"""Shared test constants: a fake work directory and a fake HF cache snapshot path (never read)."""
from pathlib import Path

from ltbench.spec import tool_command

ROOT = Path('/nonexistent/ltbench-work')
HF_HOME = '/nonexistent/hf'
TOKENIZER = Path(HF_HOME) / 'hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
UNIT = Path('/nonexistent/ltbench-out/unit')


def command(tool, rate=10, directory=UNIT, tokenizer=TOKENIZER, **kwargs):
    kwargs.setdefault('root', ROOT)
    return tool_command(tool, rate, directory, tokenizer, **kwargs)

# The machine-specific part of a manifest.
PATHS = {'root': str(ROOT), 'vllm_env': '/nonexistent/vllm-env', 'hf_home': HF_HOME}
