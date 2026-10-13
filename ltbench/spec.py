"""Pinned tools, rates, command lines and the randomized schedule. Builds commands; runs nothing."""
from pathlib import Path
import hashlib
import json
import os
import random
import re

# Model name served by the simulator and sent by every client; the simulator loads no model.
# 18 characters: request and streamed-chunk sizes are part of the measured client work.
MODEL = 'fixed-timing-sim-1'
# One RLIMIT_NOFILE (soft = hard) for every tool process.
TOOL_NOFILE = 65536
DEFAULT_PORT = 8188
DEFAULT_SEED = 20261003
RATES = (10, 50, 200, 500, 1000)
TOOLS = ('vllm', 'guidellm', 'aiperf', 'inference-perf')
PINS = {
    'vllm': ('vllm-project/vllm', 'v0.30.0', 'ced6857afa0ea7b2e3f0846a62e1394e90f15607'),
    'guidellm': ('vllm-project/guidellm', 'v0.8.0', '4545d7cdbab8753d780fa90083259e5a840b6782'),
    'aiperf': ('ai-dynamo/aiperf', 'v0.13.0', '794f8bb75f8582f22e412d7e650fc71ca2a3d21a'),
    'inference-perf': ('kubernetes-sigs/inference-perf', 'v0.7.0', '5804ea6b7ebd2bfceff29cf113311ed3af95146e'),
    'sim': ('llm-d/llm-d-inference-sim', 'v0.11.2', '7639395743c76f1cb37222981b33e8ea56cce69b'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(seed):
    rng = random.Random(seed)
    rows = []
    for round_id in range(1, 4):
        for rate in RATES:
            order = list(TOOLS)
            rng.shuffle(order)
            rows.extend(dict(round=round_id, rate=rate, tool=tool) for tool in order)
    return rows


_SNAPSHOT = re.compile(r'(?:^|/)models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})$')


def hub_snapshot(tokenizer):
    """Return (repo id, commit) of an HF cache path models--<org>--<name>/snapshots/<40-hex>."""
    match = _SNAPSHOT.search(Path(tokenizer).as_posix().rstrip('/'))
    if match is None:
        raise ValueError(f'Tokenizer path is not an HF cache snapshot directory: {tokenizer}')
    org, name, commit = match.groups()
    return f'{org}/{name}', commit


def work_root(value=None):
    """Work directory holding envs/, bin/, setup/, cache/ and config/ (--root or $LTBENCH_ROOT)."""
    value = value or os.environ.get('LTBENCH_ROOT')
    if not value:
        raise ValueError('Give the work directory with --root or LTBENCH_ROOT')
    return Path(value).resolve()


def hf_home_of(tokenizer):
    """HF_HOME that contains an HF cache snapshot <HF_HOME>/hub/models--o--n/snapshots/<commit>."""
    path = Path(tokenizer)
    hub_snapshot(path)
    if path.parents[2].name != 'hub':
        raise ValueError(f'Cannot derive HF_HOME from {tokenizer}; give --hf-home')
    return str(path.parents[3])


def paths(manifest):
    """The machine-specific locations a manifest was constructed with."""
    return {'root': Path(manifest['root']), 'vllm_env': Path(manifest['vllm_env']),
            'hf_home': manifest['hf_home']}


def run_prefix(root):
    """The Python environment of the runner and the probe: the inference-perf environment."""
    return Path(root) / 'envs' / 'inference-perf'


def environment(tool, root, vllm_env=None):
    """Python environment of a tool: vLLM's own environment, the others under <root>/envs/."""
    if tool == 'vllm':
        return Path(vllm_env) if vllm_env else Path(root) / 'envs' / 'vllm'
    return Path(root) / 'envs' / tool


def tool_command(tool, rate, directory, tokenizer, port=DEFAULT_PORT, *, root, vllm_env=None, hf_home=None):
    """Return argv, environment additions and files; never execute a tool."""
    if tool not in TOOLS or rate not in RATES:
        raise ValueError('Unknown tool or rate')
    directory, root = Path(directory), Path(root)
    url = f'http://127.0.0.1:{port}'
    binary = str(environment(tool, root, vllm_env) / 'bin' / tool)
    hf_home = hf_home or hf_home_of(tokenizer)
    env = {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'HF_HOME': str(hf_home)}
    env.update({'XDG_CACHE_HOME': str(root / 'cache'),
                'XDG_CONFIG_HOME': str(root / 'config'),
                'AIPERF_DATASET_MMAP_CACHE_DIR': str(root / 'cache/aiperf-dataset')})
    # No PYTHONPATH and no transport overlay; the tools run unmodified.
    files = {}
    if tool == 'vllm':
        argv = [binary, 'bench', 'serve', '--backend', 'openai', '--base-url', url,
                '--endpoint', '/v1/completions', '--model', MODEL, '--tokenizer', str(tokenizer),
                '--dataset-name', 'random', '--random-input-len', '100',
                '--random-output-len', '50', '--random-range-ratio', '0.0', '--ignore-eos',
                '--request-rate', str(rate), '--burstiness', 'inf', '--num-prompts', str(60 * rate),
                '--num-warmups', '0', '--ready-check-timeout-sec', '0',
                '--percentile-metrics', 'ttft,tpot,itl,e2el,client_queue_time',
                '--metric-percentiles', '50,99', '--save-result', '--save-detailed',
                '--result-dir', str(directory), '--result-filename', 'vllm.json']
    elif tool == 'guidellm':
        env['GUIDELLM__MAX_WORKER_PROCESSES'] = '24'
        argv = [binary, 'run', '--backend',
                f'kind=openai_http,target={url},model={MODEL},request_format=/v1/completions',
                '--profile', f'kind=constant,rate={rate}',
                '--constraint', 'kind=max_duration,seconds=60',
                '--data', 'kind=synthetic_text,prompt_tokens=100,output_tokens=50',
                '--tokenizer', f'kind=huggingface_auto,model={tokenizer}',
                '--output', f'kind=json,path={directory}/guidellm.json']
    elif tool == 'aiperf':
        # aiperf 0.13.0 in offline mode passes --tokenizer to huggingface_hub.snapshot_download
        # as a repo id (aiperf/common/tokenizer.py, _resolve_local_snapshot), which rejects a
        # filesystem path; pass the repo id and the snapshot commit instead. The runner checks
        # that they resolve to this same directory (runner.aiperf_tokenizer_identity).
        repo, revision = hub_snapshot(tokenizer)
        argv = [binary, 'profile', '--model', MODEL, '--url', url,
                '--endpoint-type', 'completions', '--streaming', '--tokenizer', repo,
                '--tokenizer-revision', revision,
                '--synthetic-input-tokens-mean', '100', '--synthetic-input-tokens-stddev', '0',
                '--output-tokens-mean', '50', '--output-tokens-stddev', '0',
                '--extra-inputs', 'ignore_eos:true', '--request-rate', str(rate),
                '--arrival-pattern', 'constant', '--benchmark-duration', '60',
                '--benchmark-grace-period', '5', '--use-server-token-count',
                '--workers-max', '24', '--export-level', 'records',
                '--artifact-dir', str(directory), '--ui', 'none']
    else:
        config = {
            'api': {'type': 'completion', 'streaming': True},
            'server': {'type': 'vllm', 'base_url': url, 'model_name': MODEL, 'ignore_eos': True},
            'tokenizer': {'pretrained_model_name_or_path': str(tokenizer)},
            'data': {'type': 'random', **{f'{name}_distribution': {
                'type': 'fixed', 'min': length, 'max': length, 'mean': length,
                'std_dev': 0, 'total_count': 60 * rate}
                for name, length in [('input', 100), ('output', 50)]}},
            'load': {'type': 'constant', 'stages': [{'rate': rate, 'duration': 60}],
                     'num_workers': 24, 'stage_teardown_grace_seconds': 120},
            'report': {'request_lifecycle': {'summary': True, 'per_stage': True, 'per_request': True,
                       'use_server_output_tokens': True,
                       'per_request_fields': {'computed_metrics': True}}},
            'storage': {'local_storage': {'path': str(directory)}},
        }
        # JSON is a YAML subset; retain exact bytes passed to the YAML reader.
        files['inference-perf.yaml'] = json.dumps(config, indent=2) + '\n'
        argv = [binary, '-c', str(directory / 'inference-perf.yaml')]
    return {'argv': argv, 'env': env, 'files': files,
            'expected_workers': 1 if tool == 'vllm' else 24}


def server_command(port=DEFAULT_PORT, *, root):
    return [str(Path(root) / 'bin/llm-d-inference-sim'), '--model', MODEL, '--port', str(port),
            '--time-to-first-token', '200ms', '--inter-token-latency', '20ms',
            '--time-to-first-token-std-dev', '0', '--inter-token-latency-std-dev', '0',
            '--time-factor-under-load', '1.0', '--max-num-seqs', '8192', '--seed', '42']
