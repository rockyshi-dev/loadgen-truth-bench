"""Offline construction of the run manifest; never starts the simulator or a tool.

The manifest fixes everything a run uses: tool pins, the tokenizer and its file hashes, the
about-100-token probe prompt, the CPU roles, the randomized schedule, every command line, and
the sha256 of this package's files (a run refuses to start if the code changed since).
"""
import json
import os
from pathlib import Path
import platform

from ltbench.analysis import (CALIBRATION_ATTEMPTS, CALIBRATION_BOUND_NS, CALIBRATION_MARGIN_NS, CALIBRATION_S,
                              COUNTER_READ_PERIOD_NS, CPU_SCAN_PERIOD_S, PROBE_PERIOD_NS)
from ltbench.launch import launch_environment, nofile_limit, server_environment, set_run_prefix
from ltbench.telemetry import clock_state, topology
from ltbench.spec import (PINS, TOOL_NOFILE, digest, hf_home_of, run_prefix, schedule, tool_command,
                          server_command)

CODE = Path(__file__).resolve().parent


def implementation_hashes(code=CODE):
    return {p.name: digest(p) for p in sorted(code.iterdir()) if p.suffix in ('.py', '.sh', '.json')}


def construct(output, tokenizer, seed, root, vllm_env, hf_home=None):
    affinity = topology()
    set_run_prefix(run_prefix(Path(root).resolve()))
    server_env = launch_environment(server_environment(affinity['server']))
    os.environ.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    from transformers import AutoTokenizer
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Path(tokenizer).resolve(strict=True)
    hf_home = str(hf_home) if hf_home else hf_home_of(tokenizer)
    root, vllm_env = Path(root).resolve(), Path(vllm_env).resolve()
    tok = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True, trust_remote_code=False)
    tokens = tok.encode('The quick brown fox walks through the quiet garden. ' * 30, add_special_tokens=False)[:100]
    prompt = tok.decode(tokens, skip_special_tokens=True)
    prompt_tokens = len(tok.encode(prompt, add_special_tokens=False))
    if not 95 <= prompt_tokens <= 105:
        raise ValueError(f'Probe prompt length {prompt_tokens} is not about 100 tokens')
    names = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
             'special_tokens_map.json', 'config.json', 'added_tokens.json')
    hashes = {name: digest(tokenizer / name) for name in names if (tokenizer / name).is_file()}
    if 'tokenizer.json' not in hashes:
        raise ValueError('Expected a local tokenizer.json')
    manifest = {'seed': seed, 'pins': PINS, 'tokenizer': str(tokenizer),
                'root': str(root), 'vllm_env': str(vllm_env), 'hf_home': hf_home,
                'tokenizer_sha256': hashes, 'probe_prompt': prompt, 'probe_prompt_tokens': prompt_tokens,
                'server_argv': server_command(root=root), 'python': platform.python_version(),
                'server_environment': server_env, 'affinity': affinity,
                'probe_transport_idle_limit_s': 1,
                'tool_rlimit_nofile': nofile_limit(TOOL_NOFILE),
                'probe_period_ns': PROBE_PERIOD_NS,
                'counter_read_period_ns': COUNTER_READ_PERIOD_NS,
                'counter_read_placement': 'origin + k x 100 ms + 1.18 s + d + 50 ms',
                'cpu_scan_period_s': CPU_SCAN_PERIOD_S,
                'counter_offsets': {'source': 'measured in the dry run; recorded in its receipt '
                                              'and in every attempt',
                                    'margin_ns': CALIBRATION_MARGIN_NS,
                                    'calibration_s': CALIBRATION_S,
                                    'bound_ns_including_margins': CALIBRATION_BOUND_NS,
                                    'max_attempts': CALIBRATION_ATTEMPTS},
                'clock_state_at_construction': clock_state(),
                'implementation_sha256': implementation_hashes(),
                'schedule': schedule(seed), 'commands': []}
    for row in manifest['schedule']:
        directory = root / 'runs' / f"r{row['round']}-{row['rate']}-{row['tool']}" / 'attempt-1' / 'tool'
        manifest['commands'].append({**row, **tool_command(row['tool'], row['rate'], directory, tokenizer,
                                                           root=root, vllm_env=vllm_env, hf_home=hf_home)})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
