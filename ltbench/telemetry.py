"""Linux CPU/process accounting (0.5 s loop) and counter reads placed on the probe grid.

Counter reads run at 10 Hz, each due at grid origin + k x 100 ms +
1.18 s + d + 50 ms (analysis.ReadPlacement); the due instant is when the /metrics
request is sent. The per-process CPU scan runs in its own 0.5 s loop, in a worker
thread, so its /proc walk is never on the counter read's path.
"""
import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ltbench.analysis import (COUNTER_READ_PERIOD_NS, CPU_SCAN_PERIOD_S, EvidenceError,
                      ReadPlacement)
from ltbench.launch import http_session, launch_environment


GIL_SWITCH_INTERVAL_S = .0002


def topology():
    text = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,ONLINE'], text=True)
    sockets = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        cpu, core, socket, online = line.split(',')
        if online == 'Y':
            sockets.setdefault(int(socket), {}).setdefault(int(core), []).append(int(cpu))
    if len(sockets.get(0, {})) < 26 or len(sockets.get(1, {})) < 20:
        raise EvidenceError('Node does not have the required CPU topology')
    def select(socket, begin, end):
        groups = list(sorted(sockets[socket].items()))[begin:end]
        return sorted(cpu for _, cpus in groups for cpu in cpus)
    return {'tool': select(0, 0, 26), 'server': select(1, 0, 16),
            'probe': select(1, 16, 20), 'lscpu': text,
            'meaning': 'physical cores with both SMT siblings assigned to the same role'}


FIELDS = ('user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal')


def cpu_ticks():
    """Keep each /proc/stat field; irq/softirq/steal are reported separately."""
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        if not re.match(r'cpu\d+ ', line):
            continue
        name, *values = line.split()
        nums = dict(zip(FIELDS, map(int, values[:8])))  # guest is inside user
        nums['total'] = sum(nums.values())
        result[int(name[3:])] = nums
    return result


def _stat(path):
    raw = path.read_text()
    return raw[raw.rfind(')') + 2:].split()


def _server_socket_inodes(port):
    inodes = set()
    for name in ('tcp', 'tcp6'):
        for line in Path('/proc/net', name).read_text().splitlines()[1:]:
            columns = line.split()
            if columns[3] == '01' and int(columns[2].rsplit(':', 1)[1], 16) == port:
                inodes.add(columns[9])
    return inodes


def processes(groups, thread_roles=('tool',), connections_port=None):
    """Owned processes by role. groups[role] = {'pgids': set, 'known': {pid: start}}.

    The known map holds descendants that left the process group (setsid).
    Per-thread CPU is read for thread_roles. With connections_port, count
    each process's established sockets to the server (dry-run worker check).
    """
    inodes = _server_socket_inodes(connections_port) if connections_port else None
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = _stat(path / 'stat')
            pid, pgid, start = int(path.name), int(fields[2]), int(fields[19])
            role = next((k for k, v in groups.items() if pgid in v['pgids']
                         or v.get('known', {}).get(pid) == start), None)
            if role is None:
                continue
            row = {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': start, 'role': role,
                   'ticks': int(fields[11]) + int(fields[12]),
                   'allowed_cpus': sorted(os.sched_getaffinity(pid)),
                   'argv': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
            if role in thread_roles:
                threads = {}
                for task in (path / 'task').iterdir():
                    try:
                        t = _stat(task / 'stat')
                        threads[f'{task.name}:{t[19]}'] = int(t[11]) + int(t[12])
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                row['threads'] = threads
            if inodes is not None:
                count = 0
                for fd in (path / 'fd').iterdir():
                    try:
                        target = os.readlink(fd)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    count += target.startswith('socket:[') and target[8:-1] in inodes
                row['server_connections'] = count
            result[f'{pid}:{start}'] = row
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            pass
    return result


def counters(text, allow_uninstantiated=False):
    names = {'vllm:request_success_total': 'requests',
             'vllm:generation_tokens_total': 'tokens',
             'vllm:num_requests_running': 'inflight'}
    result = {key: 0.0 for key in names.values()}
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.match(r'^([^ {]+)(?:\{.*\})?\s+([^ ]+)', line)
        if match and match[1] in names:
            key = names[match[1]]
            result[key] += float(match[2])
            seen.add(key)
    # Pinned metrics.go:421 instantiates the inflight gauge at startup. The two
    # CounterVec families have no child (not even HELP/TYPE) until first use.
    for name, key in names.items():
        if key not in seen and name not in text:
            if allow_uninstantiated and key != 'inflight' and 'inflight' in seen:
                continue
            raise EvidenceError(f'Missing metric family {name}')
    return result


class ScrapeStatusError(EvidenceError):
    """A /metrics response other than 200; its counters are never used."""
    def __init__(self, status, body, start_ns, end_ns):
        super().__init__(f'/metrics returned HTTP {status}')
        self.status, self.body, self.start_ns, self.end_ns = status, body, start_ns, end_ns


async def scrape(session, url, allow_uninstantiated=False):
    """start_ns is the read instant used for lower time bounds."""
    start = time.monotonic_ns()
    async with session.get(url + '/metrics') as response:
        raw = await response.text()
        if response.status != 200:
            raise ScrapeStatusError(response.status, raw, start, time.monotonic_ns())
    return {'start_ns': start, 'end_ns': time.monotonic_ns(),
            'counters': counters(raw, allow_uninstantiated), 'prometheus': raw,
            'allow_uninstantiated': allow_uninstantiated}


class ProbeLog:
    """Incremental reader of the probe process's JSONL log (whole lines only).

    The probe's first line is its grid record (`grid_origin_ns`, the planned
    send of its first request, written before that send); it is kept in `grid`
    and is not a request row."""
    def __init__(self, path):
        self.path, self.offset, self.rows, self.grid = Path(path), 0, [], None

    def refresh(self):
        try:
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                data = stream.read()
        except FileNotFoundError:
            return []
        complete = data[:data.rfind(b'\n') + 1]
        self.offset += len(complete)
        new = []
        for line in complete.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if 'grid_origin_ns' in row:
                self.grid = row
            else:
                new.append(row)
        self.rows.extend(new)
        return new


def read_probe(path):
    """(grid record or None, request rows) of a complete probe log."""
    log = ProbeLog(path)
    log.refresh()
    return log.grid, log.rows


async def cpu_scans(groups, stop, output, period=CPU_SCAN_PERIOD_S, connections_port=None):
    """Per-CPU ticks and owned-process/thread CPU every `period` s (its own
    loop). The /proc walk runs in a worker thread; a late scan restarts the grid
    from now instead of bursting."""
    def scan(stream):
        # The /proc walk, the JSON dump and the write all run in the worker thread.
        row = {'monotonic_ns': time.monotonic_ns(),
               'boottime_ns': time.clock_gettime_ns(time.CLOCK_BOOTTIME),
               'cpu': cpu_ticks(),
               'processes': processes(groups, connections_port=connections_port)}
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        return row
    rows = []
    epoch = time.monotonic()
    index = 0
    with output.open('x') as stream:
        while not stop.is_set():
            rows.append(await asyncio.to_thread(scan, stream))
            index += 1
            delay = epoch + index * period - time.monotonic()
            if delay < 0:
                epoch, index, delay = time.monotonic(), 0, 0
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
    return rows


async def _sleep_until(deadline_ns, stop):
    """True if stopped before the deadline."""
    delay = (deadline_ns - time.monotonic_ns()) / 1e9
    if delay <= 0:
        return stop.is_set()
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return True
    except asyncio.TimeoutError:
        return False


async def counter_reads(url, stop, output, probe_log_path, on_sample=None, allow_uninstantiated=True,
                        period_ns=COUNTER_READ_PERIOD_NS, refine=3):
    """Counter reads at 10 Hz, placed by the probe's observed completion phase.

    Before the probe's grid record exists there are no probe requests, so reads
    run on a free 100 ms grid from the loop's start ('pre-origin'). Afterwards
    read k is due at origin + k x 100 ms + 1.18 s + d + 50 ms (ReadPlacement),
    with d re-estimated from the live probe log just before the send. A late
    read is never followed by catch-up reads: the next read is the next grid
    instant after now. A non-200 or failed read is recorded with its times and
    an error and is not retried here (its counters are never used); the next
    read stays on the grid.
    """
    launch_environment({})
    # The CPU scan shares this interpreter's GIL from its worker thread; a short
    # switch interval bounds how long it can hold back a due read (measured
    # offline: worst case about 25 ms at the 5 ms default, 2.5 ms at 0.2 ms).
    sys.setswitchinterval(GIL_SWITCH_INTERVAL_S)
    log = ProbeLog(probe_log_path)
    placement = None
    rows = []
    loop_start = time.monotonic_ns()
    pre_k = 0
    last_k = None

    def update():
        nonlocal placement
        new = log.refresh()
        if placement is None and log.grid is not None:
            placement = ReadPlacement(log.grid['grid_origin_ns'], log.grid.get('period_ns', period_ns))
            new = log.rows
        if placement is not None:
            for row in new:
                placement.add(row)
    with output.open('x') as stream:
        async with http_session(3) as session:
            while not stop.is_set():
                update()
                now = time.monotonic_ns()
                if placement is None:
                    pre_k = max(pre_k, -(-(now - loop_start) // period_ns))
                    plan = {'phase': 'pre-origin', 'k': pre_k, 'due_ns': loop_start + pre_k * period_ns,
                            'd_ns': None, 'd_samples': 0, 'converged': None, 'skipped': 0}
                    pre_k += 1
                else:
                    plan = dict(placement.next_due(now, last_k), phase='grid',
                                grid_origin_ns=placement.origin_ns)
                    plan['skipped'] = 0 if last_k is None else plan['k'] - last_k - 1
                if await _sleep_until(plan['due_ns'], stop):
                    break
                if plan['phase'] == 'grid':
                    # Completions logged while waiting move d; re-place up to `refine` times.
                    for _ in range(refine):
                        update()
                        again = placement.due(plan['k'])
                        if again['due_ns'] == plan['due_ns']:
                            break
                        plan.update(again)
                        if again['due_ns'] <= time.monotonic_ns() or await _sleep_until(again['due_ns'], stop):
                            break
                    if stop.is_set():
                        break
                    last_k = plan['k']
                row = dict(plan)
                started = time.monotonic_ns()
                try:
                    row.update(await scrape(session, url, allow_uninstantiated))
                    if row['counters']['requests'] > 0:
                        allow_uninstantiated = False
                except Exception as exc:
                    row.update(start_ns=started, end_ns=time.monotonic_ns(),
                               error=f'{type(exc).__name__}: {exc}')
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if on_sample is not None:
                    on_sample(row)
    return rows


def _cpu(sample, cpu):
    # JSON reload converts integer dictionary keys to strings.
    return sample['cpu'].get(cpu, sample['cpu'].get(str(cpu)))


def cpu_accounting(rows, affinity, window=None):
    """Gate Q and tool-CPU blame.

    foreign = (user+nice+system on the role's CPUs - owned process CPU) / all
    ticks; irq, softirq, steal and iowait are reported separately and not gated.
    A process seen for the first time is credited with all its CPU if it started
    after the previous sample (its starttime), otherwise with nothing.
    per-process and per-thread CPU over every interval of at least 1 s that
    lies inside `window` = (from_ns, to_ns), the tool's send window.
    """
    rows = [r for r in rows if 'cpu' in r and 'processes' in r]
    if len(rows) < 3:
        raise EvidenceError('Incomplete CPU telemetry')
    hz = os.sysconf('SC_CLK_TCK')
    tick_ns = 1e9 / hz
    roles = ('tool', 'server', 'probe')
    totals = {role: {**{f: 0 for f in FIELDS}, 'total': 0, 'owned': 0} for role in roles}
    per_cpu = []
    unattributed = []
    cumulative = {}            # entity -> credited ticks so far
    history = [dict()]         # row index -> snapshot of tool entities' cumulative ticks
    for previous, current in zip(rows, rows[1:]):
        duration = (current['monotonic_ns'] - previous['monotonic_ns']) / 1e9
        if duration <= 0 or duration > 2.5:
            raise EvidenceError('Missing telemetry interval')
        per_cpu_row = {}
        for role in roles:
            counts = totals[role]
            for cpu in affinity[role]:
                old, new = _cpu(previous, cpu), _cpu(current, cpu)
                if old is None or new is None:
                    raise EvidenceError('Missing assigned CPU')
                delta = {f: new[f] - old[f] for f in (*FIELDS, 'total')}
                if delta['total'] <= 0 or min(delta.values()) < 0:
                    raise EvidenceError('CPU counters reset or stopped')
                for f, value in delta.items():
                    counts[f] += value
                per_cpu_row[str(cpu)] = (delta['total'] - delta['idle'] - delta['iowait']) / delta['total']
        for identity, proc in current['processes'].items():
            role = proc['role']
            if not set(proc['allowed_cpus']) <= set(affinity[role]):
                raise EvidenceError(f'Owned {role} process escaped CPU affinity')
            old = previous['processes'].get(identity)
            new_since = proc['start_ticks'] * tick_ns >= previous['boottime_ns'] - tick_ns
            if old is not None:
                delta = max(0, proc['ticks'] - old['ticks'])
            elif new_since:
                delta = proc['ticks']
            else:
                delta = 0
                unattributed.append(identity)
            totals[role]['owned'] += delta
            if role == 'tool':
                key = 'process ' + identity
                cumulative[key] = cumulative.get(key, 0) + delta
                old_threads = (old or {}).get('threads', {})
                for tid, ticks in proc.get('threads', {}).items():
                    if tid in old_threads:
                        add = max(0, ticks - old_threads[tid])
                    else:  # a new thread of a known process, or of a new process
                        add = ticks if (old is not None or new_since) else 0
                    key = f'thread {identity}:{tid}'
                    cumulative[key] = cumulative.get(key, 0) + add
        history.append(dict(cumulative))
        per_cpu.append(per_cpu_row)
    times = [r['monotonic_ns'] for r in rows]
    elapsed = (times[-1] - times[0]) / 1e9
    windows = {'process': (0, None), 'thread': (0, None)}
    lo, hi = window if window else (times[0], times[-1])
    intervals = 0
    j = 0
    for i, t in enumerate(times):
        if t < lo:
            continue
        j = max(j, i + 1)
        while j < len(times) and times[j] - t < 1e9:
            j += 1
        if j >= len(times) or times[j] > hi:
            break
        seconds = (times[j] - t) / 1e9
        intervals += 1
        for key, value in history[j].items():
            fraction = (value - history[i].get(key, 0)) / hz / seconds
            kind = key.split(' ', 1)[0]
            if fraction > windows[kind][0]:
                windows[kind] = (fraction, key)
    foreign = {role: max(0, v['user'] + v['nice'] + v['system'] - v['owned']) / v['total']
               for role, v in totals.items()}
    separate = {role: {f: v[f] / v['total'] for f in ('irq', 'softirq', 'steal', 'iowait')}
                for role, v in totals.items()}
    whole = {k: v for k, v in cumulative.items() if k.startswith('process ')}
    return {'pass': all(value <= .02 for value in foreign.values()),
            'foreign_fraction_by_role': foreign,
            'irq_softirq_steal_iowait_by_role': separate,
            'per_cpu_utilization': per_cpu,
            'busiest_tool_process_core_fraction': max(whole.values(), default=0) / hz / elapsed,
            # no 1 s interval inside the window gives no value, not 0.
            'busiest_tool_process_window_fraction': windows['process'][0] if intervals else None,
            'busiest_tool_process_window_identity': windows['process'][1],
            'busiest_tool_thread_window_fraction': windows['thread'][0] if intervals else None,
            'blame_intervals': intervals,
            'busiest_tool_thread_window_identity': windows['thread'][1],
            'blame_window_ns': [lo, hi],
            'tool_process_cpu_seconds': {k[8:]: v / hz for k, v in whole.items()},
            'unattributed_first_seen': unattributed,
            'process_accounting': 'conservative: CPU of exited/unseen processes is not subtracted'}


def workers_started(rows, tool, leader_pid):
    """Distinct tool processes that held a connection to the server.

    The tool's leader process is excluded for multi-process tools (it may make
    a validation request); AIPerf workers are the processes titled
    'aiperf worker_*' (base_service.py:118-122)."""
    seen = set()
    for row in rows:
        for identity, proc in row.get('processes', {}).items():
            if proc['role'] != 'tool' or not proc.get('server_connections'):
                continue
            if tool != 'vllm' and proc['pid'] == leader_pid:
                continue
            if tool == 'aiperf' and not proc['argv'].startswith('aiperf worker_'):
                continue
            seen.add(identity)
    return sorted(seen)


def clock_state():
    """NTP/chrony state at run time (GuideLLM stamps wall-clock time)."""
    result = {'realtime_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    for name, argv in (('chronyc_tracking', ['chronyc', '-n', 'tracking']),
                       ('timedatectl', ['timedatectl', 'show']),
                       ('ntpq', ['ntpq', '-pn'])):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=5)
            result[name] = {'returncode': done.returncode, 'stdout': done.stdout,
                            'stderr': done.stderr}
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            result[name] = {'error': f'{type(exc).__name__}: {exc}'}
    return result
