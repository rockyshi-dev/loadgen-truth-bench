"""CPU-only orchestration of the dry run and the panel.

The probe runs in its own process at 10 Hz on a fixed grid; the tool is supervised by a
waitpid thread with a one-second tick; counter reads are placed on the probe grid at 10 Hz
and bracketed with offsets measured in the dry run (back-to-back reads, bounded range); the
CPU scan runs in its own 0.5 s loop; tool blame uses the send window [f_hi, l_lo]. One failed
attempt is retried instead of aborting the panel; SIGTERM/SIGHUP/SIGINT run the cleanup and
children carry PR_SET_PDEATHSIG.
"""
import asyncio
import fcntl
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

from ltbench.analysis import (CALIBRATION_ATTEMPTS, CALIBRATION_S, CompletionTimeline, EvidenceError,
                      ProbeFailure, build_timeline, calibrate, gate_l, gate_p, gate_s, delivery,
                      comparisons, aggregate, probe_statistics, read_placement_report)
from ltbench.normalize import normalize
from ltbench.launch import (check_port, child_setup, http_session, launch_environment, listener_identity,
                    nofile_limit, owned_tree, process_snapshot, server_environment, set_run_prefix)
from ltbench.spec import (DEFAULT_PORT, PINS, TOOLS, TOOL_NOFILE, digest, environment, hub_snapshot, paths, run_prefix,
                          tool_command, server_command)
from ltbench.telemetry import (ProbeLog, clock_state, counter_reads, cpu_accounting, cpu_scans, read_probe,
                       scrape, topology, workers_started)

DEADLINE_S = 180
# No tool completion seen this long after launch: the run is stopped as
# "did not complete" (the 180 s clock never started; the tool is not sending).
TOOL_HARD_CAP_S = 600
PRE_TOOL_PROBE_COMPLETIONS = 25     # about 2.5 s of probe completions at 10 Hz before the tool starts
P_PHASE_S = 10
SUPERVISION_TICK_S = 1.0
EXPECTED_WORKERS = {'vllm': 1, 'guidellm': 24, 'aiperf': 24, 'inference-perf': 24}
KNOWN_CELLS = ('b.completed', 'b.rate', 'b.ttft.p50', 'b.ttft.p99', 'b.e2e.p50',
               'b.e2e.p99', 'c.ttft.p50', 'c.e2e.p50', 'c.gap', 'c.tokens')


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def probe_requests(path):
    """Request rows of a probe log (its grid record excluded)."""
    return read_probe(path)[1]


def dry_count_bounds(tool, rate):
    """Completed tool requests expected in a dry run at rate R.

    vLLM and inference-perf send exactly 60 R and wait for all of them.
    GuideLLM and AIPerf stop at 60 s: at least the delivery threshold
    0.95 R (60 - 1.18), at most 5% above 60 R.
    """
    if tool in ('vllm', 'inference-perf'):
        return 60 * rate, 60 * rate
    return .95 * rate * (60 - 1.18), 1.05 * 60 * rate


def preflight(manifest, manifest_path=None):
    """Refuse to run unless the loaded manifest still describes this code, its pins,
    the tokenizer bytes and a finished installation."""
    from ltbench.construct import implementation_hashes
    if manifest_path is None or json.loads(Path(manifest_path).read_text()) != manifest:
        raise EvidenceError('The manifest must be loaded from its file')
    if manifest.get('pins') != {k: list(v) for k, v in PINS.items()} \
            or 'tokenizer.json' not in manifest.get('tokenizer_sha256', {}):
        raise EvidenceError('Tool pins or tokenizer identity is incomplete')
    if manifest.get('implementation_sha256') != implementation_hashes():
        raise EvidenceError('Implementation changed since offline construction; construct a new manifest')
    for name, sha in manifest['tokenizer_sha256'].items():
        if digest(Path(manifest['tokenizer']) / name) != sha:
            raise EvidenceError('Tokenizer bytes changed')
    where = paths(manifest)
    if not (where['root'] / 'setup/INSTALL_READY').is_file():
        raise EvidenceError('Installation is incomplete')
    check_environments(where['root'], where['vllm_env'])
    aiperf_tokenizer_identity(manifest['tokenizer'], **paths(manifest))


def check_environments(root, vllm_env, python_prefix=None):
    """The runner runs in the inference-perf environment (the probe uses the same interpreter);
    every tool binary exists in the environment setup.sh verified for it."""
    expected = os.path.realpath(run_prefix(root))
    actual = os.path.realpath(python_prefix or sys.prefix)
    if actual != expected:
        raise EvidenceError(f'Run with {expected}/bin/python (this interpreter is in {actual})')
    for tool in TOOLS:
        env = os.path.realpath(environment(tool, root, vllm_env))
        if not os.path.isfile(os.path.join(env, 'bin', tool)):
            raise EvidenceError(f'{tool}: no executable {env}/bin/{tool}')
        record = Path(root) / 'setup' / f'{tool}-verify.json'
        if not record.is_file():
            raise EvidenceError(f'{tool}: {record} missing; run setup.sh')
        files = json.loads(record.read_text()).get('files', {})
        if not files or any(not os.path.realpath(f['path']).startswith(env + os.sep) for f in files.values()):
            raise EvidenceError(f'{tool}: the environment setup.sh verified is not {env}')


_RESOLVE_SNAPSHOT = ('import os, sys\n'
                     'from huggingface_hub import snapshot_download\n'
                     'print(os.path.realpath(snapshot_download(sys.argv[1], revision=sys.argv[2], '
                     'local_files_only=True)))\n')


def aiperf_tokenizer_identity(tokenizer, root, vllm_env=None, hf_home=None):
    """AIPerf receives a repo id and commit, not the directory: require that, in the
    AIPerf environment with the tool's own offline settings, they resolve to the
    manifest's tokenizer directory (whose bytes preflight has hashed)."""
    repo, revision = hub_snapshot(tokenizer)
    command = tool_command('aiperf', 10, Path(root) / 'unused', tokenizer, root=root, hf_home=hf_home)
    python = str(environment('aiperf', root) / 'bin' / 'python')
    done = subprocess.run([python, '-c', _RESOLVE_SNAPSHOT, repo, revision],
                          env=launch_environment(command['env']), capture_output=True,
                          text=True, timeout=120, check=False)
    expected = os.path.realpath(tokenizer)
    resolved = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ''
    if done.returncode != 0 or resolved != expected:
        raise EvidenceError(f'AIPerf tokenizer {repo}@{revision} resolves to {resolved or "nothing"} '
                            f'(exit {done.returncode}: {done.stderr.strip()[-300:]}), '
                            f'not the manifest tokenizer {expected}')
    return {'repo': repo, 'revision': revision, 'resolved': resolved, 'python': python}


class OwnedProcess:
    def __init__(self, argv, env, cpus, directory, name, nofile=None):
        environment = launch_environment(env)
        self.known = {}
        self.stopped = False
        self.directory, self.name = directory, name
        self.log = (directory / f'{name}.log').open('x')
        self.process = None
        self.argv = ['taskset', '-c', ','.join(map(str, cpus)), *argv]
        try:
            write_json(directory / f'{name}.manifest.json', {
                'argv': self.argv, 'environment': environment, 'cpus': cpus,
                'rlimit_nofile': None if nofile is None else {'soft': nofile, 'hard': nofile},
                'parent_death_signal': 'SIGKILL', 'uses_gpu': False})
            self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True,
                                            env=environment, preexec_fn=child_setup(nofile))
            self.group_alive()
            write_json(directory / f'{name}.process.json', {'pid': self.process.pid,
                       'pgid': self.process.pid, 'argv': self.argv, 'env_additions': env,
                       'created_monotonic_ns': time.monotonic_ns()})
        except BaseException:
            self.stop()
            raise

    @property
    def pid(self):
        return self.process.pid

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.process is not None:
            self.group_alive()  # retain escaped descendants before the leader exits
            deadline = time.monotonic() + 60
            # The group may be signalled only while its leader is unreaped: after
            # the reap the pgid can be reused. Retained descendants are
            # signalled by PID with their start time checked.
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self.signal_descendants(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)  # leader still unreaped here
                self.process.wait(timeout=10)
            # Wait for every retained descendant, even after a setsid/reparent.
            if self.group_alive():
                self.signal_descendants(signal.SIGKILL)
                while self.group_alive() and time.monotonic() < deadline:
                    time.sleep(.05)
                if self.group_alive():
                    raise EvidenceError(f'Owned process tree {self.process.pid} survived 60 s cleanup')
        self.log.close()

    def group_alive(self):
        return bool(owned_tree(self.process.pid, self.known, process_snapshot()))

    def signal_descendants(self, sig):
        snapshot = process_snapshot()
        for pid in owned_tree(self.process.pid, self.known, snapshot) - {self.process.pid}:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


async def ready(url, process):
    import aiohttp
    from ltbench.spec import MODEL
    async with http_session(2) as session:
        for _ in range(120):
            if process.process.poll() is not None:
                raise EvidenceError('Simulator exited before readiness')
            try:
                async with session.get(url + '/health') as response:
                    if response.status != 200:
                        await asyncio.sleep(.5)
                        continue
                async with session.get(url + '/v1/models') as response:
                    response.raise_for_status()
                    models = await response.json()
                if {row['id'] for row in models.get('data', [])} != {MODEL}:
                    raise EvidenceError('Ready endpoint reports another model')
                if process.process.poll() is not None:
                    raise EvidenceError('Simulator exited during readiness')
                identity = listener_identity(process.process.pid, urlparse(url).port, process.known)
                write_json(process.directory / f'{process.name}.ready.json',
                           {**identity, 'models': models})
                return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(.5)
    raise EvidenceError('Simulator did not become ready')


class OnlineClock:
    """Live upper bounds only, for the 180 s deadline. Its origin f_lo is the
    latest scrape start at which the tool cannot yet have completed a request:
    a lower bound on the tool's first completion that keeps advancing until the
    tool completes something. A sample is evaluated once every probe end that U
    needs must already be logged."""
    def __init__(self, baseline, calibration, probe_path):
        self.timeline = CompletionTimeline(baseline, calibration)
        self.log = ProbeLog(probe_path)
        self.ends = []
        self.consumed = 0       # rows of self.log.rows already in self.ends
        self.pending = []
        self.settle_ns = max(self.timeline.u_offset, self.timeline.l_offset, 0) + 500_000_000

    def add(self, sample):
        self.pending.append(sample)

    def refresh(self, now_ns):
        import bisect
        # Consume every row the log has accumulated, not only the rows this call's
        # refresh() returns: wait_probe_warm also refreshes this log, and the warm-up
        # completions must reach the clock or U overcounts the tool from the start.
        self.log.refresh()
        for row in self.log.rows[self.consumed:]:
            if row.get('done') and not row.get('error'):
                bisect.insort(self.ends, int(row['end_ns']))
        self.consumed = len(self.log.rows)
        while self.pending and self.pending[0].get('end_ns', now_ns) + self.settle_ns <= now_ns:
            self.timeline.observe(self.pending.pop(0), self.ends, online=True)
        return self.timeline.f_lo_ns


def probe_process(url, prompt, directory, affinity, name='probe'):
    (directory / f'{name}-prompt.txt').write_text(prompt)
    argv = [sys.executable, str(Path(__file__).resolve().with_name('probe.py')), '--url', url,
            '--prompt-file', str(directory / f'{name}-prompt.txt'),
            '--output', str(directory / f'{name}.jsonl')]
    return OwnedProcess(argv, {}, affinity['probe'], directory, name)


async def wait_probe_warm(probe, log, instruments=()):
    deadline = time.monotonic() + 15
    while len(log.rows) < PRE_TOOL_PROBE_COMPLETIONS:
        log.refresh()
        if probe.process.poll() is not None:
            raise EvidenceError('Probe exited during warm-up')
        for task in instruments:
            if task.done():
                task.result()
                raise EvidenceError('Telemetry stopped during probe warm-up')
        if time.monotonic() > deadline:
            raise EvidenceError('Probe did not log its first completions within 15 s')
        await asyncio.sleep(.1)


async def stop_probe(probe):
    if probe.process.poll() is None:
        os.kill(probe.pid, signal.SIGTERM)
    try:
        code = await asyncio.to_thread(probe.process.wait, 60)
    except subprocess.TimeoutExpired as exc:
        raise EvidenceError('Probe did not drain within 60 s of its stop signal') from exc
    await asyncio.to_thread(probe.stop)
    if code != 0:
        raise EvidenceError(f'Probe process exited with {code}')


async def supervise(tool, clock, probe, telemetry_task, launch_ns):
    """A waitpid thread reports the tool's exit; everything else (descendant
    retention, instrument health, deadline) runs on a one-second tick."""
    waiter = asyncio.ensure_future(asyncio.to_thread(tool.process.wait))
    while True:
        done, _ = await asyncio.wait({waiter}, timeout=SUPERVISION_TICK_S)
        now = time.monotonic_ns()
        if done:
            return {'timed_out': False, 'tool_exit_ns': now, 'deadline_start_ns': clock.timeline.f_lo_ns,
                    'deadline_rule': None}
        tool.group_alive()
        if telemetry_task.done():
            telemetry_task.result()  # propagate a collector failure immediately
            raise EvidenceError('Telemetry stopped before the tool')
        if probe.process.poll() is not None:
            raise EvidenceError('Probe exited before the tool')
        first = clock.refresh(now)
        # The 180 s limit runs from f_lo once a read has shown a certain tool
        # completion (L >= 1); until then the 600 s no-completion cap from launch
        # applies. A single ambiguous read (U >= 1, L = 0) does not switch rules,
        # so a tool that never completes anything is still stopped.
        certain = clock.timeline.certain_completion
        basis = rule = start = None
        if certain and first is not None and now - first >= DEADLINE_S * 1_000_000_000:
            basis, rule, start = 'f_lo + 180 s', 'f_lo + 180 s', first
        elif not certain and now - launch_ns >= TOOL_HARD_CAP_S * 1_000_000_000:
            basis = f'no certain tool completion (L >= 1) within {TOOL_HARD_CAP_S} s of launch'
            rule, start = f'launch + {TOOL_HARD_CAP_S} s without a certain completion', launch_ns
        if basis:
            await asyncio.to_thread(tool.stop)
            await waiter
            return {'timed_out': True, 'tool_exit_ns': time.monotonic_ns(),
                    'deadline_start_ns': start, 'deadline_basis': basis, 'deadline_rule': rule,
                    'f_lo_ns': first, 'possible_completion': clock.timeline.possible_completion}


# The pinned simulator's stream writer can call sendError on a RequestCtx that
# fasthttp has already released (sim pkg/communication/http.go:416, 579-585,
# 630-634) after a client cancels streams. Pooled contexts are left with status
# 500 and a later /metrics request on a new connection returns 500 with a valid
# body. Such a read is retried; its counters are never used.
SCRAPE_ATTEMPTS = 20
SCRAPE_BACKOFF_S = (.05, .1, .2, .3)          # then .3 s: about 5.5 s in all
GATHER_ERROR = 'An error has occurred while serving metrics'


async def scrape_checked(session, url, failure_log, allow_uninstantiated=False):
    """One /metrics read that must return 200, with bounded retries.

    Every non-200 attempt is appended to failure_log (time, status, first 300
    bytes). A promhttp gather error is a real server fault and fails at once.
    """
    from ltbench.telemetry import ScrapeStatusError
    for attempt in range(SCRAPE_ATTEMPTS):
        try:
            return await scrape(session, url, allow_uninstantiated)
        except ScrapeStatusError as exc:
            with Path(failure_log).open('a') as stream:
                stream.write(json.dumps({'attempt': attempt + 1, 'start_ns': exc.start_ns,
                                         'end_ns': exc.end_ns, 'status': exc.status,
                                         'body_head': exc.body[:300]}) + '\n')
            if exc.body.startswith(GATHER_ERROR):
                raise EvidenceError(f'/metrics gather error (HTTP {exc.status}); attempt not used') from exc
            if attempt + 1 < SCRAPE_ATTEMPTS:
                await asyncio.sleep(SCRAPE_BACKOFF_S[min(attempt, len(SCRAPE_BACKOFF_S) - 1)])
    raise EvidenceError(f'/metrics did not return 200 in {SCRAPE_ATTEMPTS} attempts; attempt not used')


async def final_counters(session, url, failure_log):
    last = None
    for _ in range(30):
        current = await scrape_checked(session, url, failure_log)
        if current['counters']['inflight'] == 0 and last is not None and current['counters'] == last['counters']:
            return current
        last = current
        await asyncio.sleep(1)
    raise EvidenceError('Server did not drain to stable counters')


def idle_tool_command(seconds=P_PHASE_S):
    return {'argv': [sys.executable, '-c', f'import time; time.sleep({seconds})'],
            'env': {}, 'files': {}, 'expected_workers': 0}


async def run_one(row, directory, manifest, affinity, disconnect_counted, calibration,
                  port=DEFAULT_PORT, *, command=None, worker_check=False):
    """One fresh server/probe/tool attempt. Called only after the top-level hold."""
    directory.mkdir(parents=True, exist_ok=False)
    tool_dir = directory / 'tool'
    tool_dir.mkdir()
    if command is None:
        command = tool_command(row['tool'], row['rate'], tool_dir, manifest['tokenizer'], port,
                               **paths(manifest))
        if row['tool'] == 'aiperf':
            # Recorded with the argv in command.json; not passed to the tool.
            command['tokenizer_identity'] = aiperf_tokenizer_identity(manifest['tokenizer'], **paths(manifest))
    for name, data in command['files'].items():
        (tool_dir / name).write_text(data)
    write_json(directory / 'command.json', command)
    write_json(directory / 'clock.json', clock_state())
    url = f'http://127.0.0.1:{port}'
    env = server_environment(affinity['server'])
    launch_environment(command['env'])
    limits = nofile_limit(TOOL_NOFILE)
    server = tool = probe = None
    telemetry_task = None
    stop_telemetry = asyncio.Event()
    probe_path = directory / 'probe.jsonl'
    try:
        check_port(port)
        server = OwnedProcess(server_command(port, root=paths(manifest)['root']), env, affinity['server'], directory, 'server')
        await ready(url, server)
        async with http_session(3) as session:
            before = await scrape_checked(session, url, directory / 'before-scrape-failures.jsonl',
                                          allow_uninstantiated=True)
            write_json(directory / 'before.json', before)
            clock = OnlineClock(before['counters']['requests'], calibration, probe_path)
            clock.add(before)
            groups = {'server': {'pgids': {server.pid}, 'known': server.known},
                      'probe': {'pgids': {os.getpgrp()}, 'known': {}},
                      'tool': {'pgids': set(), 'known': {}}}
            # Counter reads (10 Hz, placed on the probe grid) and the CPU scan
            # (0.5 s) are separate loops; either failing stops the attempt.
            telemetry_task = asyncio.gather(
                counter_reads(url, stop_telemetry, directory / 'counter-reads.jsonl', probe_path,
                              on_sample=clock.add,
                              allow_uninstantiated=before['counters']['requests'] == 0),
                cpu_scans(groups, stop_telemetry, directory / 'cpu-scans.jsonl',
                          connections_port=port if worker_check else None))
            probe = probe_process(url, manifest['probe_prompt'], directory, affinity)
            groups['probe'] = {'pgids': {os.getpgrp(), probe.pid}, 'known': probe.known}
            await wait_probe_warm(probe, clock.log, (telemetry_task,))
            launch_ns = time.monotonic_ns()
            tool = OwnedProcess(command['argv'], command['env'], affinity['tool'], directory, 'tool',
                                nofile=TOOL_NOFILE)
            groups['tool'] = {'pgids': {tool.pid}, 'known': tool.known}
            outcome = await supervise(tool, clock, probe, telemetry_task, launch_ns)
            exit_code = tool.process.poll()
            await stop_probe(probe)
            after = await final_counters(session, url, directory / 'after-scrape-failures.jsonl')
            write_json(directory / 'after.json', after)
            stop_telemetry.set()
            reads, scans = await telemetry_task
        grid, probe_rows = read_probe(probe_path)
        length = gate_l(before['counters'], after['counters'], probe_rows)  # ProbeFailure first
        server_ok = gate_s(probe_rows)
        if grid is None:
            raise EvidenceError('Probe log has no grid record')
        completed = build_timeline(before['counters']['requests'], [before, *reads, after],
                                   probe_rows, calibration)
        write_json(directory / 'completion-samples.json', completed.rows)
        consistency = completed.check(launch_ns, length['requests'])
        span = completed.span(length['requests'])
        write_json(directory / 'completion-span.json', span)
        placement = read_placement_report(reads, probe_rows, calibration)
        write_json(directory / 'counter-read-placement.json', placement)
        # The send window is [f_hi, l_lo], where the tool is certainly
        # completing requests; busiest-thread maxima are over 1 s intervals in it.
        send_window = None
        if span['f_hi_ns'] is not None and span['l_lo_ns'] is not None \
                and span['l_lo_ns'] - span['f_hi_ns'] >= 1_000_000_000:
            send_window = (span['f_hi_ns'], span['l_lo_ns'])
        accounting = cpu_accounting(scans, affinity, window=send_window)
        busiest = accounting['busiest_tool_thread_window_fraction'] if send_window else None
        timed_out = outcome['timed_out']
        result = {**row, 'attempt_path': str(directory), 'did_not_complete': timed_out,
                  'exit_code': exit_code, 'L': length, 'S': server_ok, 'Q': accounting,
                  'probe': {k: v for k, v in probe_statistics(probe_rows).items() if k != 'tokens'},
                  'calibration': calibration, 'bracket_consistency': consistency,
                  'deadline_start_ns': outcome['deadline_start_ns'],
                  'deadline_basis': outcome.get('deadline_basis'),
                  'deadline_rule': outcome.get('deadline_rule'),
                  'first_certain_completion_read_end_ns': getattr(
                      getattr(clock, 'timeline', None), 'first_certain_end_ns', None),
                  'tool_launch_ns': launch_ns, 'tool_exit_ns': outcome['tool_exit_ns'],
                  'tool_rlimit_nofile': limits, 'completion_span': span,
                  'max_bracket_width_requests': span['max_bracket_width_requests'],
                  'probe_grid': grid, 'read_placement_summary': placement['summary'],
                  'send_window_ns': list(send_window) if send_window else None,
                  'busiest_tool_thread_send_window': busiest,
                  'valid': length['pass'] and server_ok and accounting['pass'], 'cells': {}}
        if worker_check:
            result['workers_started'] = workers_started(scans, row['tool'], tool.pid)
        def attribution():
            # Decision rule: "tool" only if gate S passed and the busiest
            # thread exceeded 0.9 of a core inside [f_hi, l_lo]; otherwise (also
            # with no send window) "unexplained".
            return 'tool' if server_ok and busiest is not None and busiest > .9 else 'unexplained'
        if timed_out:
            result['delivered'] = False
            result['shortfall_attribution'] = attribution()
        elif exit_code != 0:
            # A tool that exits non-zero has not delivered the load. The
            # attempt keeps its gate validity, so the cell reads "not delivered"
            # (not "not measured"); no summary cells are read from its exports.
            result['delivered'] = False
            result['status'] = f'tool exited with status {exit_code}; not delivered'
            result['shortfall_attribution'] = 'tool exited non-zero'
        elif length['requests'] == 0:
            result['delivered'] = False
            result['status'] = 'no completed tool requests; no summary cells'
        else:
            result['delivered'] = delivery(row['tool'], row['rate'], length['requests'], span)
            if length['pass'] and server_ok:
                data = normalize(row['tool'], tool_dir)
                result['cells'] = comparisons(row['tool'], data, length['requests'], probe_rows,
                                              disconnect_counted)
                result['corrected_metrics'] = data['corrected']
                result['normalized'] = True
            # Tool-side saturation is judged on the busiest thread inside [f_hi, l_lo].
            result['shortfall_attribution'] = attribution() if result['delivered'] is False else None
        write_json(directory / 'raw-gate-calculations.json', result)
        return result
    finally:
        stop_telemetry.set()
        if telemetry_task is not None:
            try:
                await telemetry_task
            except Exception:
                pass
        try:
            if tool is not None:
                tool.stop()
        finally:
            try:
                if probe is not None:
                    probe.stop()
            finally:
                if server is not None:
                    server.stop()


async def disconnect_check(url, prompt, directory):
    """Dry-run-only experiment: cancel one stream after its first content event."""
    from ltbench.probe import SSE
    from ltbench.spec import MODEL
    async with http_session(10) as session:
        before = await scrape_checked(session, url, directory / 'disconnect-scrape-failures.jsonl')
        body = {'model': MODEL, 'prompt': prompt, 'max_tokens': 50, 'ignore_eos': True,
                'stream': True, 'stream_options': {'include_usage': True}}
        async with session.post(url + '/v1/completions', json=body) as response:
            response.raise_for_status()
            decoder = SSE()
            async for chunk in response.content.iter_any():
                if any(p != b'[DONE]' and any(c.get('text') for c in json.loads(p).get('choices', [])) for p in decoder.feed(chunk)):
                    break
        after = await final_counters(session, url, directory / 'disconnect-scrape-failures.jsonl')
    delta = after['counters']['requests'] - before['counters']['requests']
    if delta not in (0, 1):
        raise EvidenceError('Disconnect check is contaminated')
    write_json(directory / 'disconnect.json', {'before': before, 'after': after,
                                              'counted': bool(delta), 'body': body})
    return bool(delta)


def _without_prometheus(row, phase):
    return dict({k: v for k, v in row.items() if k != 'prometheus'}, phase=phase)


async def calibration_attempt(session, url, manifest, output, affinity, number):
    """One calibration: a pre-start read, back-to-back reads for CALIBRATION_S
    while the 10 Hz probe runs, then the drain and a post-drain read. Only the
    back-to-back reads bound increments (analysis.calibrate)."""
    name = f'calibration-{number}'
    failures = output / f'{name}-scrape-failures.jsonl'
    scrapes = [_without_prometheus(await scrape_checked(session, url, failures, allow_uninstantiated=True),
                                   'pre-start')]
    probe = probe_process(url, manifest['probe_prompt'], output, affinity, f'{name}-probe')
    try:
        stop_at = time.monotonic() + CALIBRATION_S
        while time.monotonic() < stop_at:
            if probe.process.poll() is not None:
                raise EvidenceError('Calibration probe exited early')
            # A non-200 read is retried and logged; its counters are never used.
            scrapes.append(_without_prometheus(
                await scrape_checked(session, url, failures, allow_uninstantiated=True), 'back-to-back'))
        await stop_probe(probe)
        scrapes.append(_without_prometheus(await final_counters(session, url, failures), 'post-drain'))
    finally:
        probe.stop()
    with (output / f'{name}-scrapes.jsonl').open('x') as stream:
        for row in scrapes:
            stream.write(json.dumps(row) + '\n')
    result = calibrate(scrapes, probe_requests(output / f'{name}-probe.jsonl'))
    result['attempt'] = number
    return result


async def calibration_phase(manifest, output, affinity, port):
    """Probe-only traffic with back-to-back counter reads. A
    calibration whose lead range including both margins exceeds 60 ms is
    repeated at most twice; every attempt is recorded and the first that passes
    is used. Returns the offsets and whether a disconnected request is counted."""
    url = f'http://127.0.0.1:{port}'
    check_port(port)
    server = OwnedProcess(server_command(port, root=paths(manifest)['root']), server_environment(affinity['server']),
                          affinity['server'], output, 'calibration-server')
    try:
        await ready(url, server)
        attempts = []
        chosen = None
        async with http_session(3) as session:
            for number in range(1, CALIBRATION_ATTEMPTS + 1):
                attempts.append(await calibration_attempt(session, url, manifest, output, affinity, number))
                if attempts[-1]['bound_pass']:
                    chosen = attempts[-1]
                    break
        write_json(output / 'calibration-attempts.json', attempts)
        if chosen is None:
            raise EvidenceError(f'Calibration lead range exceeded {attempts[-1]["bound_ns"] / 1e6:.0f} ms in '
                                f'all {len(attempts)} attempts; the dry run fails')
        write_json(output / 'calibration.json', chosen)
        counted = await disconnect_check(url, manifest['probe_prompt'], output)
        return chosen, counted
    finally:
        server.stop()


PLACEMENT_KEYS = ('grid_reads', 'grid_ambiguous_fraction', 'grid_u_side_fraction',
                  'grid_during_read_fraction', 'grid_l_side_fraction',
                  'grid_longest_ambiguous_run_reads', 'max_start_late_ns', 'skipped_grid_instants',
                  'errored_reads', 'd_online_offline_mismatches')


def placement_digest(result):
    summary = result.get('read_placement_summary') or {}
    return {key: summary.get(key) for key in PLACEMENT_KEYS}


def dry_run_checks(tool, result):
    """What the dry run must show before a receipt is issued."""
    low, high = dry_count_bounds(tool, 10)
    count = result['L']['requests']
    expected = EXPECTED_WORKERS[tool]
    workers = result.get('workers_started', [])
    checks = {
        'exit_zero': result['exit_code'] == 0 and not result['did_not_complete'],
        'gate_L': result['L']['pass'] and count > 0,
        'count_in_bounds': low <= count <= high, 'count': count, 'count_bounds': [low, high],
        'workers_started': len(workers), 'expected_workers': expected,
        'workers_match': len(workers) == expected,
        'normalized_and_compared': bool(result.get('normalized'))
                                   and set(KNOWN_CELLS) <= set(result.get('cells', {})),
        # Counter-read placement, reported for reading before a panel; not a gate.
        'read_placement': placement_digest(result),
    }
    checks['pass'] = all(checks[k] for k in ('exit_zero', 'gate_L', 'count_in_bounds',
                                              'workers_match', 'normalized_and_compared'))
    return checks


async def dry_run(manifest, output, affinity, port, manifest_sha256):
    """No scientific result is emitted by this path."""
    output.mkdir(parents=True, exist_ok=False)
    launch_environment({})
    write_json(output / 'clock.json', clock_state())
    calibration, counted = await calibration_phase(manifest, output, affinity, port)
    # Gate P under the full harness (telemetry, supervision, idle dummy tool).
    idle = await run_one({'tool': 'idle-dummy', 'rate': 0, 'round': 0}, output / 'p-phase',
                         manifest, affinity, counted, calibration, port, command=idle_tool_command())
    probe_rows = probe_requests(output / 'p-phase' / 'probe.jsonl')
    if idle['L']['requests'] != 0 or not gate_p(probe_rows):
        raise EvidenceError('P failed under the full harness: fix the instrument before any tool run')
    checks = {'p-phase': {'read_placement': placement_digest(idle)}}
    for tool in ('vllm', 'guidellm', 'aiperf', 'inference-perf'):
        result = await run_one({'tool': tool, 'rate': 10, 'round': 0}, output / tool,
                               manifest, affinity, counted, calibration, port, worker_check=True)
        checks[tool] = dry_run_checks(tool, result)
        if not checks[tool]['pass']:
            write_json(output / 'dry-run-checks.json', checks)
            raise EvidenceError(f'Dry run failed for {tool}: {checks[tool]}')
    write_json(output / 'dry-run-checks.json', checks)
    write_json(output / 'dry-run-receipt.json', {
        'P': True, 'P_statistics': idle['probe'], 'P_harness': 'full (telemetry, supervision, idle dummy tool)',
        'calibration': calibration, 'disconnect_counted': counted, 'checks': checks,
        'manifest_sha256': manifest_sha256})


async def panel(manifest, output, affinity, dry_receipt, port):
    accepted = []
    failures = []
    dropped = None
    calibration = dry_receipt['calibration']
    write_json(output / 'calibration.json', calibration)
    for row in manifest['schedule']:
        if dropped is not None and row['rate'] >= dropped:
            continue
        failures_s = failures_q = failures_i = 0
        attempt = 0
        while True:
            attempt += 1
            name = f"r{row['round']}-{row['rate']}-{row['tool']}"
            path = output / name / f'attempt-{attempt}'
            try:
                result = await run_one(row, path, manifest, affinity,
                                       dry_receipt['disconnect_counted'], calibration, port)
            except ProbeFailure as exc:
                # A failed or timed-out probe request is a gate S failure.
                result = {**row, 'attempt_path': str(path), 'S': False, 'valid': False,
                          'probe_failure': str(exc)}
            except Exception as exc:
                failures_i += 1
                record = {**row, 'attempt_path': str(path), 'instrument_failure': f'{type(exc).__name__}: {exc}'}
                failures.append(record)
                if path.is_dir():
                    write_json(path / 'attempt-error.json', record)
                if failures_i > 1:
                    raise EvidenceError(f'Instrument failed twice for {name}: {exc}') from exc
                continue
            if 'probe_failure' in result:
                write_json(path / 'attempt-error.json', result)
            elif not result['L']['pass']:
                raise EvidenceError('L failed; stop for flag correction, do not silently relabel results')
            if not result['S']:
                failures_s += 1
                if failures_s >= 2:
                    dropped = row['rate']
                    break
                continue
            if not result['Q']['pass']:
                failures_q += 1
                if failures_q <= 2:
                    continue
                result['valid'] = False
                result['status'] = 'not measured'
            accepted.append(result)
            break
    candidate = aggregate(accepted, dry_receipt['P'], dropped)
    write_json(output / 'accepted-attempts.json', accepted)
    write_json(output / 'instrument-failures.json', failures)
    write_json(output / 'decision.json', candidate)


async def guarded(coroutine):
    """SIGTERM, SIGHUP and SIGINT cancel the work so every finally block
    stops the tool, probe and server before the runner exits."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    received = []

    def handler(sig):
        received.append(sig)
        task.cancel()
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        loop.add_signal_handler(sig, handler, sig)
    try:
        return await coroutine
    except asyncio.CancelledError:
        if received:
            raise EvidenceError(f'Stopped by signal {received[0].name}; owned processes cleaned up') from None
        raise


def run(mode, manifest_path, output, dry_receipt=None, port=DEFAULT_PORT):
    """Start a dry run or a panel. `output` must not exist."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    # The runner's own environment (inherited by the probe and the simulator): offline, no GPU.
    os.environ.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HOME': manifest.get('hf_home', ''),
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    set_run_prefix(run_prefix(paths(manifest)['root']))
    preflight(manifest, manifest_path)
    launch_environment({})
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Refuse to overwrite an output directory')
    root = paths(manifest)['root']
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        affinity = topology()
        os.sched_setaffinity(0, affinity['probe'])
        if mode == 'dry-run':
            asyncio.run(guarded(dry_run(manifest, output, affinity, port, digest(manifest_path))))
        else:
            receipt = json.loads(Path(dry_receipt).read_text()) if dry_receipt else {}
            if (not receipt.get('P') or receipt.get('manifest_sha256') != digest(manifest_path)
                    or 'calibration' not in receipt):
                raise EvidenceError('Passing dry-run receipt (with calibration) for the same manifest is required')
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / 'affinity.json', affinity)
            write_json(output / 'clock.json', clock_state())
            asyncio.run(guarded(panel(manifest, output, affinity, receipt, port)))


