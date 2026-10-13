"""Probe grid, placed 10 Hz counter reads, separate CPU loop,
bounded calibration, send window, GuideLLM nearest-rank classification, W both ways.
CPU only; no server, no network traffic (HTTP and processes are faked)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import bisect
import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ltbench import analysis as a
from ltbench import construct
from ltbench import normalize
from ltbench import probe as probe_module
from ltbench import runner
from ltbench import telemetry
from tests.helpers import PATHS
from tests.test_brackets import ZERO, attempt_patches, sample
from tests.test_core import probe_row

S = 1_000_000_000
MS = 1_000_000


class ProbeGrid(unittest.TestCase):
    def test_grid_origin_is_logged_before_the_first_send_and_planned_sends_are_on_it(self):
        async def exercise(path):
            stop = asyncio.Event()
            seen = []

            async def fake_request(session, url, prompt, index, planned_ns):
                # What is on disk at the moment of the send.
                seen.append((index, planned_ns, time.monotonic_ns(), path.read_text().splitlines()))
                if index == 3:
                    stop.set()
                return {'index': index, 'planned_ns': planned_ns, 'done': True, 'error': None,
                        'end_ns': time.monotonic_ns()}
            context = AsyncMock()
            context.__aenter__.return_value = Mock()
            with patch.object(probe_module, 'request', side_effect=fake_request), \
                 patch.object(probe_module, 'http_session', return_value=context):
                await probe_module.run_probe('http://x', 'p', stop, path)
            return seen
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.jsonl'
            seen = asyncio.run(exercise(path))
            grid, rows = telemetry.read_probe(path)
        first_lines = seen[0][3]
        self.assertEqual(len(first_lines), 1)                     # only the grid record before send 0
        header = json.loads(first_lines[0])
        self.assertEqual(header, grid)
        self.assertEqual(grid['period_ns'], 100 * MS)
        self.assertLess(grid['written_ns'], grid['grid_origin_ns'])
        for index, planned, sent, _ in seen:
            self.assertEqual(planned, grid['grid_origin_ns'] + index * 100 * MS)
            self.assertGreaterEqual(sent, planned)
        self.assertEqual(sorted(r['index'] for r in rows), [0, 1, 2, 3])

    def test_probe_log_reader_keeps_grid_record_out_of_request_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.jsonl'
            path.write_text(json.dumps({'grid_origin_ns': 5, 'period_ns': 100 * MS}) + '\n'
                            + json.dumps(probe_row()) + '\n')
            log = runner.ProbeLog(path)
            self.assertEqual(len(log.refresh()), 1)
            self.assertEqual(log.grid['grid_origin_ns'], 5)
            self.assertEqual(runner.probe_requests(path), [probe_row()])


def completion(planned, delay_ns):
    return {'planned_ns': planned, 'end_ns': planned + a.REQUEST_NS + delay_ns, 'done': True, 'error': None}


class Placement(unittest.TestCase):
    def test_due_instant_formula_and_default_d(self):
        p = a.ReadPlacement(0)
        first = p.due(0)
        self.assertEqual(first['d_ns'], 30 * MS)                      # before any completion
        self.assertEqual(first['due_ns'], 1_180 * MS + 30 * MS + 50 * MS)
        for k in range(20):
            p.add(completion(k * 100 * MS, 35 * MS))
        plan = p.due(20)
        self.assertEqual(plan['d_ns'], 35 * MS)
        self.assertEqual(plan['due_ns'], 20 * 100 * MS + 1_180 * MS + 35 * MS + 50 * MS)
        self.assertTrue(plan['converged'])

    def test_d_uses_only_completions_logged_in_the_two_seconds_before_the_due_instant(self):
        p = a.ReadPlacement(0)
        for k in range(60):                       # first 3 s at 20 ms, then 40 ms
            p.add(completion(k * 100 * MS, (20 if k < 30 else 40) * MS))
        late = p.due(55)                          # window holds only 40 ms completions
        self.assertEqual(late['d_ns'], 40 * MS)
        self.assertLessEqual(late['d_samples'], 20)
        early = p.due(10)
        self.assertEqual(early['d_ns'], 20 * MS)
        # A completion logged after the due instant is not used.
        q = a.ReadPlacement(0)
        q.add(completion(0, 20 * MS))
        q.add(completion(100 * MS, 300 * MS))     # ends after read 0's due instant
        self.assertEqual(q.due(0)['d_ns'], 20 * MS)
        self.assertEqual(q.due(0)['d_samples'], 1)

    def test_no_catch_up_after_a_late_read(self):
        p = a.ReadPlacement(0)
        base0 = p.due(0)['due_ns']
        nxt = p.next_due(base0 + 250 * MS, last_k=0)  # read 0 finished 250 ms late
        self.assertEqual(nxt['k'], 3)                 # the next grid instant after now, not 1 and 2
        self.assertGreater(nxt['due_ns'], base0 + 250 * MS)
        self.assertEqual(p.next_due(base0 - 1)['k'], 0)
        self.assertEqual(p.next_due(base0)['k'], 1)   # strictly after now


class CounterReadLoop(unittest.TestCase):
    def run_loop(self, seconds, scrape_delay=lambda n: 0, with_cpu=False, origin_after=.15):
        async def exercise(directory):
            stop = asyncio.Event()
            log = directory / 'probe.jsonl'
            log.write_text('')
            loop = asyncio.get_running_loop()
            origin = time.monotonic_ns() + int(origin_after * S)

            def write_grid():
                with log.open('a') as stream:
                    stream.write(json.dumps({'grid_origin_ns': origin, 'period_ns': 100 * MS}) + '\n')

            def write_completion(k):
                with log.open('a') as stream:
                    stream.write(json.dumps(completion(origin + k * 100 * MS, 33 * MS)) + '\n')
            loop.call_later(origin_after * .5, write_grid)
            k = 0
            while origin + k * 100 * MS + a.REQUEST_NS + 33 * MS < time.monotonic_ns() + seconds * S:
                end = origin + k * 100 * MS + a.REQUEST_NS + 33 * MS
                loop.call_at(loop.time() + (end - time.monotonic_ns()) / 1e9, write_completion, k)
                k += 1
            calls = []

            async def fake_scrape(session, url, allow):
                calls.append(1)
                start = time.monotonic_ns()
                await asyncio.sleep(scrape_delay(len(calls)))
                return {'start_ns': start, 'end_ns': time.monotonic_ns(),
                        'counters': {'requests': 0}, 'prometheus': 'x'}
            context = AsyncMock()
            context.__aenter__.return_value = Mock()
            loop.call_later(seconds, stop.set)

            def slow_processes(*args, **kwargs):
                # An 80 ms /proc walk (a measured median of 47-93 ms), holding the GIL throughout.
                t = time.perf_counter()
                while time.perf_counter() - t < .08:
                    sum(int(c) for c in '1234567890')
                return {}
            tasks = [telemetry.counter_reads('http://x', stop, directory / 'reads.jsonl', log)]
            if with_cpu:
                tasks.append(telemetry.cpu_scans({}, stop, directory / 'cpu.jsonl'))
            with patch.object(telemetry, 'scrape', side_effect=fake_scrape), \
                 patch.object(telemetry, 'http_session', return_value=context), \
                 patch.object(telemetry, 'processes', side_effect=slow_processes):
                return await asyncio.gather(*tasks), origin
        with tempfile.TemporaryDirectory() as tmp:
            return asyncio.run(exercise(Path(tmp)))

    def test_reads_follow_the_grid_and_record_due_start_end_and_d(self):
        (reads,), origin = self.run_loop(2.2)
        pre = [r for r in reads if r['phase'] == 'pre-origin']
        grid = [r for r in reads if r['phase'] == 'grid']
        self.assertTrue(pre and grid)
        for r in grid:
            for key in ('due_ns', 'start_ns', 'end_ns', 'd_ns', 'k'):
                self.assertIn(key, r)
            self.assertEqual(r['due_ns'], origin + r['k'] * 100 * MS + a.REQUEST_NS + r['d_ns'] + 50 * MS)
            self.assertGreaterEqual(r['start_ns'], r['due_ns'])
            self.assertLess(r['start_ns'] - r['due_ns'], 15 * MS)
        self.assertEqual([r['k'] for r in grid], list(range(grid[0]['k'], grid[0]['k'] + len(grid))))
        self.assertTrue(any(r['d_ns'] == 33 * MS for r in grid))       # after completions are logged
        self.assertTrue(all(r['d_ns'] in (30 * MS, 33 * MS) for r in grid))
        starts = [r['start_ns'] for r in reads]
        gaps = [b - a_ for a_, b in zip(starts, starts[1:])]
        self.assertLess(max(gaps), 200 * MS)                          # 10 Hz, also across the origin

    def test_late_read_is_not_followed_by_catch_up_reads(self):
        (reads,), _ = self.run_loop(2.4, scrape_delay=lambda n: .25 if n == 18 else 0)
        grid = [r for r in reads if r['phase'] == 'grid']
        starts = [r['start_ns'] for r in grid]
        # No burst after the late read; the next read is the next grid instant.
        self.assertGreater(min(b - a_ for a_, b in zip(starts, starts[1:])), 90 * MS)
        late = next(i for i, r in enumerate(grid) if r['end_ns'] - r['start_ns'] > 200 * MS)
        self.assertGreaterEqual(grid[late + 1]['skipped'], 2)
        self.assertGreater(grid[late + 1]['due_ns'], grid[late]['end_ns'])

    def test_cpu_scan_loop_is_off_the_counter_read_path(self):
        (reads, scans), _ = self.run_loop(2.2, with_cpu=True)
        grid = [r for r in reads if r['phase'] == 'grid']
        self.assertTrue(all(r['start_ns'] - r['due_ns'] < 15 * MS for r in grid),
                        [(r['start_ns'] - r['due_ns']) / MS for r in grid])
        self.assertAlmostEqual(__import__('sys').getswitchinterval(), telemetry.GIL_SWITCH_INTERVAL_S, 9)
        times = [s['monotonic_ns'] for s in scans]
        self.assertTrue(all(400 * MS < b - a_ < 650 * MS for a_, b in zip(times, times[1:])))

    def test_errored_read_is_recorded_and_the_grid_continues(self):
        async def exercise(directory):
            stop = asyncio.Event()
            log = directory / 'probe.jsonl'
            log.write_text(json.dumps({'grid_origin_ns': time.monotonic_ns() - 2 * S, 'period_ns': 100 * MS}) + '\n')
            calls = []

            async def fake_scrape(session, url, allow):
                calls.append(1)
                if len(calls) == 2:
                    raise telemetry.ScrapeStatusError(500, 'body', 1, 2)
                return {'start_ns': time.monotonic_ns(), 'end_ns': time.monotonic_ns(),
                        'counters': {'requests': 0}, 'prometheus': ''}
            context = AsyncMock()
            context.__aenter__.return_value = Mock()
            asyncio.get_running_loop().call_later(.45, stop.set)
            with patch.object(telemetry, 'scrape', side_effect=fake_scrape), \
                 patch.object(telemetry, 'http_session', return_value=context):
                return await telemetry.counter_reads('http://x', stop, directory / 'r.jsonl', log)
        with tempfile.TemporaryDirectory() as tmp:
            reads = asyncio.run(exercise(Path(tmp)))
        self.assertIn('error', reads[1])
        self.assertEqual(reads[2]['k'], reads[1]['k'] + 1)


def placed_run(rate=10, tool_start_s=5., lead_ns=2 * MS, read_ns=2 * MS, seconds_after=4):
    """Synthetic attempt at 10 Hz: probe ends 20-40 ms after planned + 1.18 s,
    the counter leads the probe log by lead_ns, a perfect tool sends 60 R
    requests; reads are placed exactly as counter_reads places them."""
    origin = 0
    n = 60 * rate
    tool = [int((tool_start_s + 1.18 + i / rate) * S) for i in range(n)]
    stop_probe = tool[-1] + 2 * S
    probe_rows = []
    k = 0
    while origin + k * 100 * MS < stop_probe:
        delay = (20 + (k * 7919) % 21) * MS
        probe_rows.append(completion(origin + k * 100 * MS, delay))
        k += 1
    counted = sorted([r['end_ns'] - lead_ns for r in probe_rows] + tool)
    placement = a.ReadPlacement(origin)
    by_end = sorted(probe_rows, key=lambda r: r['end_ns'])
    j = 0
    reads = []
    t = -500 * MS
    while t < origin:                                   # pre-origin reads
        reads.append({'phase': 'pre-origin', 'start_ns': t, 'end_ns': t + read_ns})
        t += 100 * MS
    last_k = None
    now = t
    while now < stop_probe + seconds_after * S:
        plan = placement.next_due(now, last_k)
        while j < len(by_end) and by_end[j]['end_ns'] < plan['due_ns']:
            placement.add(by_end[j])
            j += 1
        plan = dict(placement.due(plan['k']), phase='grid', grid_origin_ns=origin)
        start = plan['due_ns']
        reads.append(dict(plan, start_ns=start, end_ns=start + read_ns))
        last_k, now = plan['k'], start + read_ns
    for r in reads:
        r['counters'] = {'requests': bisect.bisect_right(counted, (r['start_ns'] + r['end_ns']) // 2)}
    return {'reads': reads, 'probe': probe_rows, 'tool': tool, 'n': n}


class TimelineAtTenHertz(unittest.TestCase):
    def test_brackets_gap_check_and_delivery_hold_with_placed_reads(self):
        calibration = {'u_offset_ns': -8 * MS, 'l_offset_ns': 12 * MS}
        for rate in (10, 500):
            with self.subTest(rate=rate):
                run = placed_run(rate)
                before = {'start_ns': -600 * MS, 'end_ns': -599 * MS, 'counters': {'requests': 0}}
                timeline = a.build_timeline(0, [before, *run['reads']], run['probe'], calibration)
                timeline.check(int(5 * S), run['n'])
                span = timeline.span(run['n'])
                true = (run['tool'][-1] - run['tool'][0]) / S
                self.assertLessEqual(span['lower_s'], true)
                self.assertGreaterEqual(span['upper_s'], true)
                self.assertLess(span['upper_s'] - span['lower_s'], .5)       # ~0.1 s per end
                self.assertIs(a.delivery('vllm', rate, run['n'], span), True)
                report = a.read_placement_report(run['reads'], run['probe'], calibration)
                self.assertEqual(report['summary']['ambiguous_reads'], 0)
                self.assertEqual(report['summary']['d_online_offline_mismatches'], 0)
                grid = [r for r in report['reads'] if r['phase'] == 'grid']
                self.assertGreaterEqual(min(r['nearest_probe_completion_distance_ns'] for r in grid), 10 * MS)

    def test_gap_bound_still_rejects_a_missing_second(self):
        run = placed_run(10)
        reads = [r for r in run['reads'] if not (20 * S < r['start_ns'] < 21.2 * S)]
        with self.assertRaisesRegex(a.EvidenceError, 'gap exceeds'):
            a.build_timeline(0, reads, run['probe'], ZERO)

    def test_placement_report_distance_and_ambiguity(self):
        reads = [{'phase': 'grid', 'k': 0, 'due_ns': 100, 'start_ns': 100, 'end_ns': 110, 'd_ns': 0,
                  'grid_origin_ns': 0}]
        probe = [{'planned_ns': -a.REQUEST_NS + 90, 'end_ns': 90, 'done': True, 'error': None},
                 {'planned_ns': -a.REQUEST_NS + 140, 'end_ns': 140, 'done': True, 'error': None}]
        report = a.read_placement_report(reads, probe, {'u_offset_ns': -5, 'l_offset_ns': 40})
        row = report['reads'][0]
        self.assertEqual(row['nearest_probe_completion_distance_ns'], 10)
        self.assertEqual(row['nearest_probe_completion_minus_start_ns'], -10)
        self.assertEqual(row['probe_completions_in_uncertain_window'], 1)        # 140 <= 110 + 40
        self.assertIn('offline', report['distance_computed'])


class CpuLoopWindow(unittest.TestCase):
    def test_half_second_scans_and_startup_outside_the_send_window(self):
        from tests.test_harness import AFFINITY, HZ, proc, row, ticks
        rows = []
        busy = 0
        for i in range(41):                   # 20 s at 0.5 s
            t = i / 2
            if t <= 3:                        # startup: one thread saturated for 3 s
                busy = int(t * HZ)
            n = int(t * HZ)
            rows.append(row(t, [ticks(user=busy, idle=n - busy), ticks(idle=n), ticks(idle=n)],
                            {'7:1': proc('tool', 7, busy, threads={'7:1': busy})}))
        whole = telemetry.cpu_accounting(rows, AFFINITY, window=(0, 20 * S))
        sending = telemetry.cpu_accounting(rows, AFFINITY, window=(5 * S, 18 * S))
        self.assertGreater(whole['busiest_tool_thread_window_fraction'], .9)
        self.assertEqual(sending['busiest_tool_thread_window_fraction'], 0)


class AttemptWindow(unittest.TestCase):
    def attempt(self, tool_requests=True, accounting=None, outcome=None):
        # One probe request (ends at 0.05 s); a tool completing 60 requests from 4 s.
        p = dict(probe_row(start_ns=0, end_ns=50 * MS), planned_ns=-a.REQUEST_NS)
        events = [sample(.1 * k, 1) for k in range(1, 30)]
        if tool_requests:
            events += [sample(3 + .1 * k, 1 + min(k, 600) // 10) for k in range(0, 640)]
            final = 61
        else:
            events += [sample(3 + .1 * k, 1) for k in range(0, 40)]
            final = 1
        before, after = sample(0, 0), sample(3 + 64.5 if tool_requests else 7.5, final)
        before['counters'].update(tokens=0, inflight=0)
        after['counters'].update(requests=final, tokens=50 * final, inflight=0)
        outcome = outcome or {'timed_out': False, 'tool_exit_ns': 67 * S, 'deadline_start_ns': 1}
        accounting = accounting or {'pass': True, 'busiest_tool_thread_window_fraction': .42}

        async def exercise(directory):
            patches = attempt_patches(self, directory, before, after, events, [p], outcome, accounting)
            for item in patches:
                item.start()
            try:
                with patch.object(runner.time, 'monotonic_ns', return_value=0):
                    result = await runner.run_one({'tool': 'vllm', 'rate': 10, 'round': 1}, directory,
                                                  {'tokenizer': '/synthetic', 'probe_prompt': 'text', **PATHS},
                                                  {'server': [1], 'tool': [0], 'probe': [2]}, False, ZERO)
                return result, runner.cpu_accounting.call_args, \
                    json.loads((directory / 'counter-read-placement.json').read_text())
            finally:
                for item in reversed(patches):
                    item.stop()
        with tempfile.TemporaryDirectory() as tmp:
            return asyncio.run(exercise(Path(tmp) / 'attempt'))

    def test_blame_window_is_f_hi_to_l_lo_and_value_is_published(self):
        result, call, placement = self.attempt()
        span = result['completion_span']
        self.assertEqual(call.kwargs['window'], (span['f_hi_ns'], span['l_lo_ns']))
        self.assertEqual(result['send_window_ns'], [span['f_hi_ns'], span['l_lo_ns']])
        self.assertEqual(result['busiest_tool_thread_send_window'], .42)
        self.assertIn('summary', placement)
        self.assertEqual(result['read_placement_summary'], placement['summary'])
        self.assertEqual(result['probe_grid']['grid_origin_ns'], 0)

    def test_no_send_window_gives_no_busiest_value(self):
        result, call, _ = self.attempt(tool_requests=False)
        self.assertIsNone(call.kwargs['window'])
        self.assertIsNone(result['busiest_tool_thread_send_window'])


class CalibrationAttempts(unittest.TestCase):
    def run_phase(self, passes):
        attempts = [dict(ZERO, bound_pass=ok, bound_ns=60 * MS, attempt=i + 1) for i, ok in enumerate(passes)]

        async def exercise(output):
            server = Mock()
            context = AsyncMock()
            context.__aenter__.return_value = Mock()
            with patch.object(runner, 'check_port'), \
                 patch.object(runner, 'OwnedProcess', return_value=server), \
                 patch.object(runner, 'ready', new=AsyncMock()), \
                 patch.object(runner, 'http_session', return_value=context), \
                 patch.object(runner, 'server_environment', return_value={}), \
                 patch.object(runner, 'calibration_attempt', new=AsyncMock(side_effect=attempts)) as attempt, \
                 patch.object(runner, 'disconnect_check', new=AsyncMock(return_value=True)):
                try:
                    result = await runner.calibration_phase({'probe_prompt': 'p', **PATHS}, output, {'server': [1]}, 8188)
                except a.EvidenceError as exc:
                    result = exc
                return result, attempt.call_count
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result, calls = asyncio.run(exercise(output))
            recorded = json.loads((output / 'calibration-attempts.json').read_text())
            used = json.loads((output / 'calibration.json').read_text()) if (output / 'calibration.json').exists() else None
        return result, calls, recorded, used

    def test_first_passing_attempt_is_used_and_every_attempt_recorded(self):
        result, calls, recorded, used = self.run_phase([False, True, True])
        self.assertEqual(calls, 2)
        self.assertEqual([r['attempt'] for r in recorded], [1, 2])
        self.assertEqual(used['attempt'], 2)
        self.assertEqual(result[0]['attempt'], 2)

    def test_three_failures_fail_the_dry_run(self):
        result, calls, recorded, used = self.run_phase([False, False, False])
        self.assertIsInstance(result, a.EvidenceError)
        self.assertRegex(str(result), 'all 3 attempts')
        self.assertEqual(calls, 3)
        self.assertEqual(len(recorded), 3)
        self.assertIsNone(used)

    def test_attempt_tags_pre_start_back_to_back_and_post_drain_reads(self):
        async def exercise(output):
            reads = iter(range(100))

            async def fake_scrape(session, url, log, allow_uninstantiated=False):
                await asyncio.sleep(.01)
                n = next(reads)
                return {'start_ns': n, 'end_ns': n, 'counters': {'requests': 0}, 'prometheus': 'big'}

            def fake_probe(url, prompt, directory, affinity, name):
                (directory / f'{name}.jsonl').write_text(json.dumps({'grid_origin_ns': 0}) + '\n'
                                                         + json.dumps(probe_row()) + '\n')
                owned = Mock()
                owned.process.poll.return_value = None
                return owned
            with patch.object(runner, 'CALIBRATION_S', .05), \
                 patch.object(runner, 'scrape_checked', side_effect=fake_scrape), \
                 patch.object(runner, 'final_counters', new=AsyncMock(return_value={
                     'start_ns': 999, 'end_ns': 999, 'counters': {'requests': 1}, 'prometheus': 'big'})), \
                 patch.object(runner, 'probe_process', side_effect=fake_probe), \
                 patch.object(runner, 'stop_probe', new=AsyncMock()), \
                 patch.object(runner, 'calibrate', return_value={'bound_pass': True}) as calibrate:
                await runner.calibration_attempt(Mock(), 'http://x', {'probe_prompt': 'p'}, output, {}, 2)
                return calibrate.call_args.args
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            scrapes, probe_rows = asyncio.run(exercise(output))
            on_disk = [json.loads(line) for line in (output / 'calibration-2-scrapes.jsonl').read_text().splitlines()]
        phases = [s['phase'] for s in scrapes]
        self.assertEqual(phases[0], 'pre-start')
        self.assertEqual(phases[-1], 'post-drain')
        self.assertTrue(set(phases[1:-1]) == {'back-to-back'} and len(phases) > 3)
        self.assertNotIn('prometheus', on_disk[0])
        self.assertEqual(probe_rows, [probe_row()])          # grid record excluded


def guidellm_cell(reported, exported, n, q):
    return a.guidellm_percentile(reported, exported, n, q)['value']


class GuidellmRank(unittest.TestCase):
    def test_nearest_rank_one_above_and_other(self):
        xs = [float(i) for i in range(1, 101)]           # rank r has value r
        self.assertEqual(guidellm_cell(50., xs, 100, 50), 'differs by documented definition')
        above = a.guidellm_percentile(51., xs, 100, 50)
        self.assertEqual((above['value'], above['known_issue']), ('wrong', 'vllm-project/guidellm#1193'))
        self.assertEqual(above['label'], 'wrong (known)')
        self.assertEqual(guidellm_cell(49., xs, 100, 50), 'wrong')
        self.assertEqual(guidellm_cell(50.5, xs, 100, 50), 'wrong')     # numpy's value is not GuideLLM's

    def test_rank_is_exact_not_floating_point(self):
        xs = [float(i) for i in range(1, 101)]
        self.assertEqual(int(__import__('math').ceil(.07 * 100)), 8)      # the float trap
        self.assertEqual(a.guidellm_percentile(7., xs, 100, 7)['rank'], 7)
        self.assertEqual(guidellm_cell(7., xs, 100, 7), 'differs by documented definition')
        self.assertEqual(a.guidellm_percentile(99., xs, 100, 99)['rank'], 99)
        self.assertEqual(guidellm_cell(99., xs, 100, 99), 'differs by documented definition')
        self.assertEqual(a.guidellm_percentile(100., xs, 100, 99)['known_issue'], 'vllm-project/guidellm#1193')

    def test_ties_read_as_the_rank_value(self):
        xs = [1., 2., 2., 3.]
        self.assertEqual(guidellm_cell(2., xs, 4, 50), 'differs by documented definition')

    def test_count_mismatch_band(self):
        # Export holds 101 values, summary counts 100 (m = 1): rank 99 of the summary.
        xs = [float(i) for i in range(1, 102)]
        for value in (98., 99., 100.):
            self.assertEqual(guidellm_cell(value, xs, 100, 99), 'differs by documented definition', value)
        self.assertEqual(guidellm_cell(101., xs, 100, 99), 'wrong')
        self.assertEqual(guidellm_cell(97., xs, 100, 99), 'wrong')
        self.assertIsNone(a.guidellm_percentile(100., xs, 100, 99)['known_issue'])

    def test_round_3_rate_500_shape(self):
        # 25,852 exported, 25,851 counted: the reported p99 is rank ceil(0.99 x 25,851).
        xs = [float(i) for i in range(1, 25_853)]
        result = a.guidellm_percentile(25_593., xs, 25_851, 99)
        self.assertEqual(result['rank'], 25_593)
        self.assertEqual(result['value'], 'differs by documented definition')


def guidellm_normalized(ttft_ms, e2e_s, reported_ttft, reported_e2e, summary=None, export=None):
    n = len(ttft_ms) if summary is None else summary
    return {'completed': n, 'incomplete': 0, 'rate': 1.0, 'duration_s': None,
            'requests': [{'ttft_s': t / 1000, 'e2e_s': e} for t, e in zip(ttft_ms, e2e_s)],
            'all_start_times': [], 'output_tokens': [50] * len(ttft_ms),
            'ttft_p50_s': reported_ttft[50] / 1000, 'ttft_p99_s': reported_ttft[99] / 1000,
            'e2e_p50_s': reported_e2e[50], 'e2e_p99_s': reported_e2e[99], 'gap_p50_s': .02,
            'percentile_native': {
                'ttft': {'unit': 'ms', 'summary_count': n, 'values': ttft_ms, 'reported': reported_ttft},
                'e2e': {'unit': 's', 'summary_count': n, 'values': e2e_s, 'reported': reported_e2e}},
            'counts': {'export_successful': len(ttft_ms) if export is None else export,
                       'summary_successful': n, 'export_incomplete': 0, 'summary_incomplete': 0},
            'corrected': {}}


class GuidellmCells(unittest.TestCase):
    def cells(self, **kwargs):
        ttft = [200. + i for i in range(100)]
        e2e = [1.2 + i / 1000 for i in range(100)]
        normalized = guidellm_normalized(ttft, e2e, {50: 249., 99: 298.}, {50: e2e[49], 99: e2e[99]}, **kwargs)
        probe = [probe_row(ttft_s=.2, e2e_s=1.2)]
        return a.comparisons('guidellm', normalized, 100, probe, False)

    def test_comparisons_classify_by_documented_rank_and_keep_numpy_value_and_difference(self):
        cells = self.cells()
        self.assertEqual(cells['b.ttft.p50']['value'], 'differs by documented definition')
        self.assertEqual(cells['b.ttft.p50']['value_numpy_reference'], 'wrong')
        self.assertAlmostEqual(cells['b.ttft.p50']['difference'], (249. - 249.5) / 1000)
        self.assertEqual(cells['b.e2e.p99']['known_issue'], 'vllm-project/guidellm#1193')
        self.assertEqual(cells['b.e2e.p99']['value'], 'wrong')
        self.assertEqual(cells['b.guidellm.count']['value'], 'recorded')
        self.assertEqual(cells['b.guidellm.count']['difference'], 0)

    def test_count_mismatch_is_its_own_entry(self):
        cells = self.cells(summary=99, export=100)
        self.assertEqual(cells['b.guidellm.count']['difference'], 1)
        self.assertEqual(cells['b.guidellm.count']['summary_count_ttft'], 99)

    def test_normalize_reads_native_values_and_summary_counts(self):
        def request(ttft_ms, latency):
            return {'time_to_first_token_ms': ttft_ms, 'request_latency': latency, 'output_tokens': 50,
                    'info': {'timings': {'request_start': 10.0, 'first_token_iteration': 10.0 + ttft_ms / 1000,
                                         'request_end': 10.0 + latency}}}
        stats = lambda count, p50, p99: {'successful': {'count': count, 'mean': 1,
                                                        'percentiles': {'p50': p50, 'p99': p99}}}
        doc = {'benchmarks': [{'requests': {'successful': [request(201., 1.21), request(203., 1.23)],
                                            'incomplete': [{}], 'errored': []},
                               'metrics': {'request_totals': {'successful': 1, 'incomplete': 2},
                                           'requests_per_second': {'successful': {'mean': 9.8}},
                                           'output_token_count': {'successful': {'mean': 50}},
                                           'time_to_first_token_ms': stats(1, 201., 203.),
                                           'request_latency': stats(1, 1.21, 1.23),
                                           'inter_token_latency_ms': stats(1, 20., 20.),
                                           'request_dispatch_delay': {}, 'request_scheduled_latency': {}}}]}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'guidellm.json').write_text(json.dumps(doc))
            n = normalize.normalize('guidellm', tmp)
        self.assertEqual(n['percentile_native']['ttft']['values'], [201., 203.])
        self.assertEqual(n['percentile_native']['ttft']['summary_count'], 1)
        self.assertEqual(n['percentile_native']['e2e']['reported'], {50: 1.21, 99: 1.23})
        self.assertEqual(n['counts'], {'export_successful': 2, 'summary_successful': 1,
                                       'export_incomplete': 1, 'summary_incomplete': 2})


class Aggregate(unittest.TestCase):
    def records(self):
        records = []
        for tool in ('vllm', 'guidellm', 'aiperf', 'inference-perf'):
            for rate in (10, 50, 200, 500, 1000):
                for round_id in (1, 2, 3):
                    cells = {name: {'value': 'agrees'} for name in runner.KNOWN_CELLS}
                    if tool == 'inference-perf':
                        cells['c.gap'] = {'value': 'no entry'}
                    if tool == 'guidellm':
                        cells['b.rate'] = {'value': 'no entry'}
                        for name in ('b.ttft.p50', 'b.ttft.p99', 'b.e2e.p50', 'b.e2e.p99'):
                            cells[name] = {'value': 'differs by documented definition',
                                           'value_numpy_reference': 'wrong'}
                        cells['b.guidellm.count'] = {'value': 'recorded', 'difference': 0}
                    records.append({'tool': tool, 'rate': rate, 'round': round_id, 'valid': True,
                                    'delivered': True, 'cells': cells,
                                    'busiest_tool_thread_send_window': .1 * round_id,
                                    'shortfall_attribution': None, 'did_not_complete': False,
                                    'exit_code': 0})
        return records

    def test_w_under_both_rules_and_busiest_thread_published(self):
        candidate = a.aggregate(self.records(), True)
        self.assertEqual(candidate['W'], 0)
        self.assertEqual(candidate['W_numpy_reference'], 4)
        self.assertEqual(candidate['decision'], 'CLEAN')
        self.assertIn('numpy', candidate['W_note'])
        self.assertEqual(candidate['cells_numpy_reference']['guidellm/10/b.ttft.p50'], 'wrong')
        self.assertEqual(candidate['cells']['guidellm/10/b.ttft.p50'], 'differs by documented definition')
        self.assertEqual([r['busiest_thread_core_fraction'] for r in candidate['busiest_thread']['vllm/500']],
                         [.1, .2, .30000000000000004])
        self.assertEqual(len(candidate['busiest_thread']), 20)
        self.assertEqual(len(candidate['guidellm_count_entries']['10']), 3)


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'calibration_reads.json'
PROMETHEUS = """# HELP vllm:generation_tokens_total Total number of generated tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="fixed-timing-sim-1"} %d
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="fixed-timing-sim-1"} 0
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finish_reason="length",model_name="fixed-timing-sim-1"} %d
"""


def tag_phases(rows):
    """The fixture's first read was before the probe started, its last after the drain."""
    return [dict(r, phase='pre-start' if i == 0 else 'post-drain' if i == len(rows) - 1 else 'back-to-back')
            for i, r in enumerate(rows)]


class RealCounters(unittest.TestCase):
    """Real counters are floats (telemetry.counters); calibrate must accept them."""

    def test_recorded_bracket_edges_give_the_expected_offsets(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertIsInstance(fixture['scrapes'][0]['counters']['requests'], float)
        cal = a.calibrate(tag_phases(fixture['scrapes']), fixture['probe'])
        self.assertEqual(cal['u_offset_ns'], -17_690_677)          # -17.69 ms
        self.assertEqual(cal['l_offset_ns'], 31_984_736)           # +31.98 ms
        self.assertEqual(cal['lead_range_with_margins_ns'], 49_675_413)
        self.assertTrue(cal['bound_pass'])
        self.assertEqual((cal['probe_requests_used'], cal['used_increments']), (219, [1, 219]))

    def test_parsed_prometheus_text_feeds_calibrate(self):
        from tests.test_calibration import calibration_scrapes
        scrapes, rows = calibration_scrapes(2 * MS, seconds=3)
        parsed = []
        for r in scrapes:
            n = r['counters']['requests']
            counters = telemetry.counters(PROMETHEUS % (50 * n, n))
            self.assertIsInstance(counters['requests'], float)
            parsed.append(dict(r, counters=counters))
        cal = a.calibrate(parsed, rows)
        self.assertEqual(cal['probe_requests_used'], len(rows))
        self.assertLessEqual(cal['measured_lead_min_ns'], 2 * MS)
        self.assertGreaterEqual(cal['measured_lead_max_ns'], 2 * MS)
        bad = [dict(r) for r in parsed]
        bad[5] = dict(bad[5], counters=dict(bad[5]['counters'], requests=bad[5]['counters']['requests'] + .5))
        with self.assertRaisesRegex(a.EvidenceError, 'Invalid calibration completion counter'):
            a.calibrate(bad, rows)


class TimedOutAttribution(unittest.TestCase):
    """A timed-out attempt gets a shortfall attribution by the decision rule."""
    attempt = AttemptWindow.attempt

    def run_case(self, busiest, s_passes=True, tool_requests=True):
        accounting = {'pass': True, 'busiest_tool_thread_window_fraction': busiest}
        outcome = {'timed_out': True, 'tool_exit_ns': 67 * S, 'deadline_start_ns': 1,
                   'deadline_basis': 'f_lo + 180 s'}
        with patch.object(runner, 'gate_s', return_value=s_passes):
            return self.attempt(tool_requests, accounting=accounting, outcome=outcome)[0]

    def test_rule(self):
        tool = self.run_case(.95)
        self.assertIs(tool['delivered'], False)
        self.assertEqual(tool['shortfall_attribution'], 'tool')
        self.assertEqual(self.run_case(.42)['shortfall_attribution'], 'unexplained')
        self.assertEqual(self.run_case(.95, s_passes=False)['shortfall_attribution'], 'unexplained')
        self.assertEqual(self.run_case(.95, tool_requests=False)['shortfall_attribution'], 'unexplained')


class PlacementSummary(unittest.TestCase):
    """U side / during / L side split and longest run, carried into the dry-run checks."""

    def test_split_and_longest_run(self):
        cal = {'u_offset_ns': -20, 'l_offset_ns': 30}
        reads = [{'phase': 'grid', 'k': k, 'due_ns': 100 * k, 'start_ns': 100 * k, 'end_ns': 100 * k + 5,
                  'd_ns': 0, 'grid_origin_ns': None} for k in range(1, 7)]
        ends = [190, 303, 430, 505]        # before read 2, during read 3, after read 4, outside read 5
        probe = [{'planned_ns': e - a.REQUEST_NS, 'end_ns': e, 'done': True, 'error': None} for e in ends]
        report = a.read_placement_report(reads, probe, cal)
        rows = report['reads']
        self.assertEqual([r['u_side_completions'] for r in rows], [0, 1, 0, 0, 0, 0])
        self.assertEqual([r['during_read_completions'] for r in rows], [0, 0, 1, 0, 1, 0])
        self.assertEqual([r['l_side_completions'] for r in rows], [0, 0, 0, 1, 0, 0])
        summary = report['summary']
        self.assertEqual(summary['grid_longest_ambiguous_run_reads'], 4)
        self.assertAlmostEqual(summary['grid_ambiguous_fraction'], 4 / 6)
        self.assertAlmostEqual(summary['grid_u_side_fraction'], 1 / 6)
        self.assertAlmostEqual(summary['grid_l_side_fraction'], 1 / 6)
        result = {'exit_code': 0, 'did_not_complete': False, 'L': {'requests': 600, 'pass': True},
                  'workers_started': ['1:1'], 'normalized': True,
                  'cells': {name: {} for name in runner.KNOWN_CELLS}, 'read_placement_summary': summary}
        checks = runner.dry_run_checks('vllm', result)
        self.assertTrue(checks['pass'])                  # placement is reported, not gated
        self.assertEqual(checks['read_placement']['grid_longest_ambiguous_run_reads'], 4)
        self.assertEqual(set(checks['read_placement']), set(runner.PLACEMENT_KEYS))


class ShortWindow(unittest.TestCase):
    def test_window_without_a_one_second_interval_gives_none(self):
        from tests.test_harness import AFFINITY, HZ, proc, row, ticks
        rows = [row(t / 2, [ticks(user=n, idle=n), ticks(idle=2 * n), ticks(idle=2 * n)],
                    {'7:1': proc('tool', 7, n, threads={'7:1': n})})
                for t, n in ((i, i * HZ // 2) for i in range(12))]
        short = telemetry.cpu_accounting(rows, AFFINITY, window=(int(2.2 * S), int(3.4 * S)))
        self.assertIsNone(short['busiest_tool_thread_window_fraction'])
        self.assertEqual(short['blame_intervals'], 0)
        idle = telemetry.cpu_accounting(rows, AFFINITY, window=(0, int(5.5 * S)))
        self.assertGreater(idle['blame_intervals'], 0)


class WarmupReachesOnlineClock(unittest.TestCase):
    """The real wait_probe_warm -> OnlineClock -> supervise path.
    Every probe completion read during warm-up must reach the online clock, so
    the 180 s deadline cannot start before the tool is launched and the 600 s
    no-completion cap can fire."""

    def scenario(self, tool_completes):
        now = time.monotonic_ns()
        # The synthetic attempt lies in the past: 20 s back when the tool completes
        # (well inside 180 s), 700 s back when it does not (past the 600 s cap).
        base = now - (20 if tool_completes else 700) * S
        launch = base + 5 * S
        cal = {'u_offset_ns': -18 * MS, 'l_offset_ns': 32 * MS}
        probe_ends = [base + k * 100 * MS + a.REQUEST_NS + 30 * MS for k in range(60)]
        tool_ends = ([launch + 1_500 * MS + i * 100 * MS for i in range(20)] if tool_completes else [])
        counted = sorted([e - 2 * MS for e in probe_ends] + tool_ends)
        samples = [{'start_ns': base + j * 100 * MS + 80 * MS, 'end_ns': base + j * 100 * MS + 82 * MS,
                    'counters': {'requests': float(bisect.bisect_right(counted, base + j * 100 * MS + 81 * MS))}}
                   for j in range(90)]
        return base, launch, cal, probe_ends, samples

    def run_path(self, tool_completes):
        base, launch, cal, probe_ends, samples = self.scenario(tool_completes)

        async def exercise(path):
            with path.open('w') as stream:
                stream.write(json.dumps({'grid_origin_ns': base, 'period_ns': 100 * MS}) + '\n')
                for k, end in enumerate(probe_ends):
                    stream.write(json.dumps({'index': k, 'planned_ns': base + k * 100 * MS, 'end_ns': end,
                                             'done': True, 'error': None}) + '\n')
            before = {'start_ns': base - 50 * MS, 'end_ns': base - 49 * MS, 'counters': {'requests': 0.0}}
            clock = runner.OnlineClock(0, cal, path)
            clock.add(before)
            for sample in samples:
                clock.add(sample)
            probe = Mock()
            probe.process.poll.return_value = None
            await runner.wait_probe_warm(probe, clock.log)          # the real warm-up, consuming rows
            self.assertGreaterEqual(len(clock.log.rows), runner.PRE_TOOL_PROBE_COMPLETIONS)
            import threading
            released = threading.Event()
            tool = Mock()
            tool.process.wait = lambda: released.wait(5)
            tool.stop = Mock(side_effect=released.set)
            if tool_completes:
                threading.Timer(.3, released.set).start()            # the tool exits normally
            telemetry_task = Mock()
            telemetry_task.done.return_value = False
            with patch.object(runner, 'SUPERVISION_TICK_S', .05):
                try:
                    outcome = await runner.supervise(tool, clock, probe, telemetry_task, launch)
                finally:
                    released.set()
            return outcome, clock
        with tempfile.TemporaryDirectory() as tmp:
            return asyncio.run(exercise(Path(tmp) / 'probe.jsonl')), launch, base

    def test_deadline_starts_after_tool_launch(self):
        (outcome, clock), launch, base = self.run_path(tool_completes=True)
        self.assertEqual(len(clock.ends), 60)                   # no probe completion was dropped
        self.assertFalse(outcome['timed_out'])
        self.assertIsNotNone(outcome['deadline_start_ns'])
        self.assertGreaterEqual(outcome['deadline_start_ns'], launch)
        self.assertLess(outcome['deadline_start_ns'], launch + 1_500 * MS)   # before the first tool completion
        self.assertTrue(clock.timeline.possible_completion)

    def test_no_tool_completion_lets_the_600_s_cap_fire(self):
        (outcome, clock), launch, base = self.run_path(tool_completes=False)
        self.assertFalse(clock.timeline.possible_completion)    # probe completions alone are explained
        self.assertTrue(outcome['timed_out'])
        self.assertIn('no certain tool completion (L >= 1) within 600 s', outcome['deadline_basis'])


class CapLiveness(unittest.TestCase):
    """One U-side ambiguous read before any tool completion
    must not disable the 600 s cap; and the clock must follow a probe log that
    grows between ticks with a half-written last line. Real ProbeLog, OnlineClock
    and supervise; only the tool process and the telemetry task are mocked."""
    CAL = {'u_offset_ns': -18 * MS, 'l_offset_ns': 32 * MS}

    @staticmethod
    def tool(exit_after=None):
        import threading
        released = threading.Event()
        tool = Mock()
        tool.process.wait = lambda: released.wait(30)
        tool.stop = Mock(side_effect=released.set)
        if exit_after is not None:
            threading.Timer(exit_after, released.set).start()
        return tool, released

    def test_one_ambiguous_read_then_nothing_ends_at_the_600_s_cap(self):
        now = time.monotonic_ns()
        base = now - 610 * S                      # launch 605 s ago; reads run up to now
        launch = base + 5 * S
        probe_ends = [base + k * 100 * MS + 10 * MS for k in range(6100)]       # phase +10 ms
        counted = [e - 2 * MS for e in probe_ends]                            # counter leads by 2 ms
        starts = [base + j * 100 * MS + 80 * MS for j in range(6095)]         # phase +80 ms: unambiguous
        odd = 80                                    # launch + 3 s: one read 5 ms after a probe end
        starts[odd] = base + odd * 100 * MS + 15 * MS

        def read(start):
            return {'start_ns': start, 'end_ns': start + 2 * MS,
                    'counters': {'requests': float(bisect.bisect_right(counted, start + MS))}}

        async def exercise(path):
            with path.open('w') as stream:
                stream.write(json.dumps({'grid_origin_ns': base, 'period_ns': 100 * MS}) + '\n')
                for k, end in enumerate(probe_ends):
                    stream.write(json.dumps({'index': k, 'end_ns': end, 'done': True, 'error': None}) + '\n')
            clock = runner.OnlineClock(0, self.CAL, path)
            clock.add({'start_ns': base - 50 * MS, 'end_ns': base - 49 * MS, 'counters': {'requests': 0.0}})
            for start in starts:
                clock.add(read(start))
            probe = Mock()
            probe.process.poll.return_value = None
            await runner.wait_probe_warm(probe, clock.log)
            tool, released = self.tool()
            task = Mock()
            task.done.return_value = False
            with patch.object(runner, 'SUPERVISION_TICK_S', .05):
                try:
                    # Under the pre-fix rule this never returns (f_lo trails now).
                    outcome = await asyncio.wait_for(runner.supervise(tool, clock, probe, task, launch), 5)
                finally:
                    released.set()
            return outcome, clock, tool
        with tempfile.TemporaryDirectory() as tmp:
            outcome, clock, tool = asyncio.run(exercise(Path(tmp) / 'probe.jsonl'))
        ambiguous = [r for r in clock.timeline.rows if r['tool_completed_upper'] >= 1]
        self.assertEqual(len(ambiguous), 1)                 # the scenario has exactly one such read
        self.assertTrue(clock.timeline.possible_completion)
        self.assertFalse(clock.timeline.certain_completion)
        self.assertTrue(outcome['timed_out'])
        self.assertEqual(outcome['deadline_rule'], 'launch + 600 s without a certain completion')
        self.assertIn('no certain tool completion (L >= 1) within 600 s', outcome['deadline_basis'])
        self.assertEqual(outcome['deadline_start_ns'], launch)
        tool.stop.assert_called_once()

    def test_log_growing_between_ticks_with_a_half_written_line(self):
        async def exercise(path):
            loop = asyncio.get_running_loop()
            base = time.monotonic_ns() + 100 * MS
            launch = base + 300 * MS
            n = 25
            probe_ends = [base + k * 100 * MS + 10 * MS for k in range(n)]
            tool_ends = [launch + 1_000 * MS + i * 100 * MS for i in range(8)]
            counted = sorted([e - 2 * MS for e in probe_ends] + tool_ends)
            stream = path.open('w')
            stream.write(json.dumps({'grid_origin_ns': base, 'period_ns': 100 * MS}) + '\n')
            stream.flush()
            clock = runner.OnlineClock(0, self.CAL, path)
            clock.add({'start_ns': base - 50 * MS, 'end_ns': base - 49 * MS, 'counters': {'requests': 0.0}})

            def at(t_ns, fn, *args):
                loop.call_at(loop.time() + (t_ns - time.monotonic_ns()) / 1e9, fn, *args)

            def write(text):
                stream.write(text)
                stream.flush()
            for k, end in enumerate(probe_ends):          # each line arrives in two pieces, 40 ms apart
                line = json.dumps({'index': k, 'end_ns': end, 'done': True, 'error': None}) + '\n'
                at(end + MS, write, line[:len(line) // 2])
                at(end + 41 * MS, write, line[len(line) // 2:])

            def add_read(start):
                clock.add({'start_ns': start, 'end_ns': start + 2 * MS,
                           'counters': {'requests': float(bisect.bisect_right(counted, start + MS))}})
            for j in range(n):
                at(base + j * 100 * MS + 80 * MS, add_read, base + j * 100 * MS + 80 * MS)
            probe = Mock()
            probe.process.poll.return_value = None
            tool, released = self.tool(exit_after=(base + n * 100 * MS - time.monotonic_ns()) / 1e9)
            task = Mock()
            task.done.return_value = False
            partial_seen = []
            original = clock.log.refresh

            def spy():
                rows = original()
                partial_seen.append(clock.log.path.stat().st_size > clock.log.offset)
                return rows
            clock.log.refresh = spy
            with patch.object(runner, 'SUPERVISION_TICK_S', .03):
                try:
                    outcome = await runner.supervise(tool, clock, probe, task, launch)
                finally:
                    released.set()
            await asyncio.sleep(.7)
            clock.refresh(time.monotonic_ns())               # settle the last reads
            stream.close()
            return outcome, clock, launch, tool_ends, n, partial_seen
        with tempfile.TemporaryDirectory() as tmp:
            outcome, clock, launch, tool_ends, n, partial_seen = asyncio.run(exercise(Path(tmp) / 'probe.jsonl'))
        self.assertTrue(any(partial_seen))                   # a tick did meet a half-written line
        self.assertEqual(len(clock.ends), n)                 # every completion reached the clock once
        self.assertEqual(clock.ends, sorted(set(clock.ends)))
        self.assertFalse(outcome['timed_out'])
        self.assertIsNone(outcome['deadline_rule'])
        self.assertTrue(clock.timeline.certain_completion)
        self.assertGreaterEqual(clock.timeline.f_lo_ns, launch)
        self.assertLess(clock.timeline.f_lo_ns, tool_ends[0])
        self.assertTrue(all(r['tool_completed_upper'] >= 0 for r in clock.timeline.rows))


class Bookkeeping(unittest.TestCase):
    def test_construct_manifest_records_paths_and_code_hashes(self):
        source = inspect.getsource(construct.construct)
        for key in ("'root'", "'vllm_env'", "'hf_home'", "'implementation_sha256'"):
            self.assertIn(key, source)

    def test_old_offset_tag_rule_is_gone(self):
        self.assertFalse(hasattr(a, 'guidellm_1193'))
        self.assertEqual(runner.PRE_TOOL_PROBE_COMPLETIONS, 25)
        self.assertEqual(a.CALIBRATION_S, 25)


if __name__ == '__main__':
    unittest.main()
