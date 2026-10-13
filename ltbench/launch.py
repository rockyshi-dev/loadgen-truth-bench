"""CPU-only launch identity, inherited environment, child setup and HTTP pool policy."""
import ctypes
import os
from pathlib import Path
import resource
import signal
import socket
import sys

from ltbench.analysis import EvidenceError


# Child processes (simulator, probe, tools) get a minimal, explicit environment: the run's
# Python environment activated (as `conda activate` would put it first on PATH), the
# variables below if the runner has them, and the harness's own additions. Nothing else of
# the launching shell (credentials, SSH_*, other conda state) reaches a child or a file.
SAFE_INHERITED = ('HOME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR',
                  'HF_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
SYSTEM_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
_RUN_PREFIX = None


def set_run_prefix(prefix):
    """The Python environment the runner and probe use (default: the running interpreter's)."""
    global _RUN_PREFIX
    _RUN_PREFIX = None if prefix is None else str(prefix)


def base_environment(inherited):
    prefix = _RUN_PREFIX or sys.prefix
    env = {key: inherited[key] for key in SAFE_INHERITED if key in inherited}
    env.update({'PATH': f'{prefix}/bin:{SYSTEM_PATH}', 'CONDA_PREFIX': prefix, 'CONDA_DEFAULT_ENV': prefix})
    return env


def launch_environment(additions, inherited=None):
    inherited = dict(os.environ if inherited is None else inherited)
    banned = [key for key in inherited if key == 'LD_PRELOAD'
              or key.startswith(('PYTORCH_', 'OMP_', 'VLLM_', 'NCCL_'))
              or (key.startswith('CUDA_') and key != 'CUDA_VISIBLE_DEVICES')]
    if banned:
        raise EvidenceError('Refuse inherited launch overrides: ' + ', '.join(sorted(banned)))
    if additions.get('CUDA_VISIBLE_DEVICES', '') != '':
        raise EvidenceError('Benchmark processes must hide CUDA')
    return {**base_environment(inherited), **additions, 'CUDA_VISIBLE_DEVICES': ''}


PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True)


def nofile_limit(value):
    """The one RLIMIT_NOFILE (soft = hard = value) for every tool."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < value:
        raise EvidenceError(f'RLIMIT_NOFILE hard limit {hard} is below the required {value}')
    return {'soft': value, 'hard': value, 'runner_soft': soft, 'runner_hard': hard}


def child_setup(nofile=None):
    """Runs in the child before exec: parent-death signal and descriptor limit."""
    parent = os.getpid()

    def setup():
        if _LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
            os._exit(70)  # the runner is already gone; never start an orphan
        if nofile is not None:
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    return setup


def server_environment(cpus):
    return {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
            'GOMAXPROCS': str(len(cpus))}


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise EvidenceError(f'Port {port} is unavailable: {exc}') from exc


def process_snapshot(root=Path('/proc')):
    processes = {}
    for path in root.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            text = (path / 'stat').read_text()
            fields = text[text.rfind(')') + 2:].split()
            processes[int(path.name)] = {'state': fields[0], 'ppid': int(fields[1]),
                                        'pgid': int(fields[2]), 'sid': int(fields[3]),
                                        'start_ticks': int(fields[19])}
        except (FileNotFoundError, ProcessLookupError):
            continue
    return processes


def owned_tree(pid, known, snapshot):
    """Keep PID/start identities, including descendants that change groups."""
    owned = {p for p, start in known.items() if p in snapshot and snapshot[p]['start_ticks'] == start}
    # The Popen PID is not reused before it is reaped. Never adopt a reused root.
    if pid in snapshot and (pid not in known or snapshot[pid]['start_ticks'] == known[pid]):
        owned.add(pid)
    while True:
        more = {p for p, row in snapshot.items()
                if (p != pid or p not in known or row['start_ticks'] == known[p])
                and (row['ppid'] in owned or row['pgid'] == pid or row['sid'] == pid)}
        if more <= owned:
            break
        owned |= more
    known.update({p: snapshot[p]['start_ticks'] for p in owned})
    return {p for p in owned if snapshot[p]['state'] != 'Z'}


def listener_identity(pid, port, known, root=Path('/proc')):
    snapshot = process_snapshot(root)
    live = owned_tree(pid, known, snapshot)
    inodes = set()
    for name in ('tcp', 'tcp6'):
        for line in (root / 'net' / name).read_text().splitlines()[1:]:
            columns = line.split()
            if columns[3] == '0A' and int(columns[1].rsplit(':', 1)[1], 16) == port:
                inodes.add(columns[9])
    owners = {}
    for member in live:
        try:
            for fd in (root / str(member) / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target.startswith('socket:[') and target[8:-1] in inodes:
                    owners[str(member)] = target[8:-1]
        except FileNotFoundError:
            continue
    if not inodes or set(owners.values()) != inodes:
        raise EvidenceError('Answering listener is not exclusively owned by the server process tree')
    return {'port': port, 'listener_inodes': sorted(inodes), 'owners': owners,
            'process_start_ticks': {str(p): known[p] for p in live}}


def http_session(timeout, *, limit=100):
    """Aiohttp rejects an idle pooled connection after one second at checkout."""
    import aiohttp
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),
                                 connector=aiohttp.TCPConnector(limit=limit, keepalive_timeout=1.0))
