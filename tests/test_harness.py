"""Quiet gate, tool blame, known-issue tags, dry-run checks, panel retries, sampling, clock and
/metrics status handling. Synthetic data, no traffic."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ltbench import analysis as a
from ltbench import normalize
from ltbench import runner
from ltbench import telemetry
from tests.helpers import PATHS
from tests.test_brackets import ZERO, sample
from tests.test_core import probe_row

HZ = os.sysconf('SC_CLK_TCK')
S = 1_000_000_000


def ticks(user=0, system=0, idle=0, softirq=0, irq=0):
    row = {'user': user, 'nice': 0, 'system': system, 'idle': idle, 'iowait': 0,
           'irq': irq, 'softirq': softirq, 'steal': 0}
    row['total'] = sum(row.values())
    return row


def proc(role, pid, ticks_, start=1, threads=None, argv='tool', connections=0):
    row = {'pid': pid, 'ppid': 1, 'start_ticks': start, 'role': role, 'ticks': ticks_,
           'allowed_cpus': [{'tool': 0, 'server': 1, 'probe': 2}[role]], 'argv': argv,
           'server_connections': connections}
    if threads is not None:
        row['threads'] = threads
    return row


AFFINITY = {'tool': [0], 'server': [1], 'probe': [2]}


def row(t, cpus, processes):
    return {'monotonic_ns': int(t * S), 'boottime_ns': int((1000 + t) * S),
            'cpu': {cpu: value for cpu, value in enumerate(cpus)}, 'processes': processes}


class QuietGate(unittest.TestCase):
    def test_softirq_is_reported_not_counted_as_foreign(self):
        rows = []
        for k in range(3):
            n = k * HZ
            rows.append(row(k, [ticks(idle=n), ticks(user=n // 2, softirq=3 * n // 10, idle=n - n // 2 - 3 * n // 10),
                                ticks(idle=n)],
                            {'9:1': proc('server', 9, n // 2)}))
        result = telemetry.cpu_accounting(rows, AFFINITY)
        self.assertTrue(result['pass'])
        self.assertEqual(result['foreign_fraction_by_role']['server'], 0)
        self.assertAlmostEqual(result['irq_softirq_steal_iowait_by_role']['server']['softirq'], .3, 2)

    def test_first_seen_process_is_credited_by_its_starttime(self):
        def rows_with(start_ticks):
            rows = [row(0, [ticks(idle=0), ticks(idle=0), ticks(idle=0)], {}),
                    row(1, [ticks(idle=HZ), ticks(idle=HZ), ticks(idle=HZ)], {})]
            busy = ticks(user=HZ // 2, idle=HZ + HZ // 2)
            rows.append(row(2, [busy, ticks(idle=2 * HZ), ticks(idle=2 * HZ)],
                            {'7:5': proc('tool', 7, HZ // 2, start=start_ticks)}))
            return telemetry.cpu_accounting(rows, AFFINITY)
        started_inside = rows_with(int(1001.5 * HZ))      # after the previous sample (boottime 1001 s)
        self.assertEqual(started_inside['foreign_fraction_by_role']['tool'], 0)
        started_before = rows_with(int(999 * HZ))          # existed but was never seen
        self.assertGreater(started_before['foreign_fraction_by_role']['tool'], .2)
        self.assertEqual(started_before['unattributed_first_seen'], ['7:5'])


class ToolBlame(unittest.TestCase):
    def test_per_thread_one_second_window_inside_send_window(self):
        # One thread saturates a core for one second out of ten; the whole-run
        # process average is diluted far below 0.9.
        rows = []
        thread_a = 0
        for k in range(11):
            if k == 5:
                thread_a += int(.95 * HZ)
            n = k * HZ
            rows.append(row(k, [ticks(user=thread_a, idle=n - thread_a), ticks(idle=n), ticks(idle=n)],
                            {'7:1': proc('tool', 7, thread_a, threads={'7:1': thread_a, '8:1': 0})}))
        result = telemetry.cpu_accounting(rows, AFFINITY, window=(0, 10 * S))
        self.assertLess(result['busiest_tool_process_core_fraction'], .2)
        self.assertAlmostEqual(result['busiest_tool_thread_window_fraction'], .95, 2)
        self.assertIn('7:1:7:1', result['busiest_tool_thread_window_identity'])
        outside = telemetry.cpu_accounting(rows, AFFINITY, window=(6 * S, 10 * S))
        self.assertEqual(outside['busiest_tool_thread_window_fraction'], 0)

    def test_workers_counted_from_connected_non_leader_processes(self):
        rows = [{'processes': {
            '10:1': proc('tool', 10, 0, argv='python guidellm', connections=1),
            '11:1': proc('tool', 11, 0, argv='python -c spawn_main', connections=2),
            '12:1': proc('tool', 12, 0, argv='python -c spawn_main', connections=0),
            '13:1': proc('tool', 13, 0, argv='aiperf worker_ab12', connections=1),
            '14:1': proc('server', 14, 0, connections=5)}}]
        self.assertEqual(telemetry.workers_started(rows, 'guidellm', 10), ['11:1', '13:1'])
        self.assertEqual(telemetry.workers_started(rows, 'aiperf', 10), ['13:1'])
        self.assertEqual(telemetry.workers_started(rows, 'vllm', 10), ['10:1', '11:1', '13:1'])


class KnownIssueTags(unittest.TestCase):
    def test_aiperf_1492_needs_phase_window_arithmetic(self):
        reference = 600 / 60.0
        by_window = 600 / 61.5
        tagged = a.cell(by_window, reference, .01)
        self.assertEqual(a.aiperf_1492(tagged, {'phase_window_s': 61.5}, 600), 'ai-dynamo/aiperf#1492')
        self.assertIsNone(a.aiperf_1492(tagged, {'phase_window_s': None}, 600))
        other = a.cell(9.5, reference, .01)
        self.assertIsNone(a.aiperf_1492(other, {'phase_window_s': 61.5}, 600))
        above = a.cell(600 / 58.0, reference, .01)
        self.assertIsNone(a.aiperf_1492(above, {'phase_window_s': 58.0}, 600))

    def test_aiperf_normalize_reads_phase_window(self):
        summary = {'request_count': {'avg': 1}, 'request_throughput': {'avg': 1},
                   'benchmark_duration': {'avg': 1}, 'output_token_count': {'avg': 50},
                   'time_to_first_token': {'p50': 200, 'p99': 200},
                   'request_latency': {'p50': 1180, 'p99': 1180}, 'inter_token_latency': {'p50': 20},
                   'start_time': '2026-10-05T10:00:00.000000', 'end_time': '2026-10-05T10:01:01.500000'}
        record = {'metadata': {'request_start_ns': 1}, 'metrics': {
            'time_to_first_token': {'value': 200}, 'request_latency': {'value': 1180},
            'output_token_count': {'value': 50}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'profile_export_aiperf.json').write_text(json.dumps(summary))
            (path / 'profile_export.jsonl').write_text(json.dumps(record) + '\n')
            self.assertEqual(normalize.normalize('aiperf', path)['phase_window_s'], 61.5)


class DryRunChecks(unittest.TestCase):
    def result(self, count, workers, cells=True):
        return {'exit_code': 0, 'did_not_complete': False, 'L': {'requests': count, 'pass': True},
                'workers_started': [f'{i}:1' for i in range(workers)], 'normalized': cells,
                'cells': {name: {} for name in runner.KNOWN_CELLS} if cells else {}}

    def test_count_workers_and_normalize(self):
        self.assertTrue(runner.dry_run_checks('vllm', self.result(600, 1))['pass'])
        self.assertFalse(runner.dry_run_checks('vllm', self.result(599, 1))['pass'])
        self.assertTrue(runner.dry_run_checks('aiperf', self.result(590, 24))['pass'])
        self.assertFalse(runner.dry_run_checks('aiperf', self.result(558, 24))['pass'])
        self.assertFalse(runner.dry_run_checks('aiperf', self.result(631, 24))['pass'])
        self.assertFalse(runner.dry_run_checks('guidellm', self.result(590, 23))['pass'])
        self.assertFalse(runner.dry_run_checks('inference-perf', self.result(600, 24, cells=False))['pass'])


class PanelAttempts(unittest.TestCase):
    def run_panel(self, outcomes):
        manifest = {'schedule': [{'round': 1, 'rate': 10, 'tool': 'vllm'}]}
        good = {'round': 1, 'rate': 10, 'tool': 'vllm', 'L': {'pass': True}, 'S': True,
                'Q': {'pass': True}, 'valid': True}

        async def attempt(row, path, *args, **kwargs):
            path.mkdir(parents=True)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return dict(good, **outcome)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(runner, 'run_one', side_effect=attempt):
                try:
                    asyncio.run(runner.panel(manifest, output, AFFINITY,
                                             {'P': True, 'calibration': ZERO, 'disconnect_counted': False}, 8188))
                except a.EvidenceError as exc:
                    return exc, None
            accepted = json.loads((output / 'accepted-attempts.json').read_text())
            failures = json.loads((output / 'instrument-failures.json').read_text())
            return accepted, failures

    def test_probe_failure_is_a_gate_s_failure_for_that_attempt(self):
        accepted, failures = self.run_panel([a.ProbeFailure('timeout'), {}])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(failures, [])
        accepted, _ = self.run_panel([a.ProbeFailure('timeout'), a.ProbeFailure('timeout')])
        self.assertEqual(accepted, [])  # two S failures drop the rate

    def test_instrument_failure_is_rerun_once(self):
        accepted, failures = self.run_panel([a.EvidenceError('scrape gap'), {}])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(failures), 1)
        error, _ = self.run_panel([a.EvidenceError('scrape gap'), OSError('disk')])
        self.assertRegex(str(error), 'Instrument failed twice')


class Sampling(unittest.TestCase):
    def test_probe_log_reader_returns_whole_lines_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.jsonl'
            log = runner.ProbeLog(path)
            self.assertEqual(log.refresh(), [])
            path.write_text(json.dumps(probe_row()) + '\n{"partial":')
            self.assertEqual(len(log.refresh()), 1)
            with path.open('a') as stream:
                stream.write(' 1}\n')
            self.assertEqual(log.refresh(), [{'partial': 1}])

    def test_online_clock_waits_for_probe_log_before_evaluating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.jsonl'
            path.write_text('')
            clock = runner.OnlineClock(0, ZERO, path)
            clock.add(sample(1, 0))
            clock.add(sample(1.5, 1))
            self.assertEqual(clock.refresh(int(1.4 * S)), None)     # nothing settled yet
            path.write_text(json.dumps(probe_row(end_ns=int(1.4 * S))) + '\n')
            self.assertEqual(clock.refresh(int(3 * S)), int(1.5 * S))  # probe explains the increase
            self.assertFalse(clock.timeline.possible_completion)


class Clock(unittest.TestCase):
    def test_clock_state_records_missing_tools(self):
        with patch.object(telemetry.subprocess, 'run', side_effect=FileNotFoundError('chronyc')):
            state = telemetry.clock_state()
        self.assertIn('error', state['chronyc_tracking'])
        self.assertIn('realtime_ns', state)


class MeasuredOffsets(unittest.TestCase):
    def attempt(self, exit_code, thread, process=.0):
        from tests.test_brackets import attempt_patches
        p = probe_row(start_ns=500_000_000, end_ns=1_400_000_000)
        # Tool completes one request at ~2 s and another at ~70 s: span too long.
        events = [sample(.5 * k, 2 if k >= 4 else (1 if k >= 3 else 0)) for k in range(1, 140)]
        before, after = sample(0, 0), sample(70.5, 3)
        before['counters'].update(tokens=0, inflight=0)
        after['counters'].update(tokens=150, inflight=0)
        events.append(dict(sample(70, 3)))
        outcome = {'timed_out': False, 'tool_exit_ns': 70 * S, 'deadline_start_ns': 1}
        accounting = {'pass': True, 'busiest_tool_thread_window_fraction': thread,
                      'busiest_tool_process_window_fraction': process}

        async def exercise(directory):
            patches = attempt_patches(self, directory, before, after, events, [p], outcome, accounting)
            for item in patches:
                item.start()
            runner.OwnedProcess.return_value.process.poll.return_value = exit_code
            try:
                with patch.object(runner.time, 'monotonic_ns', return_value=0):
                    return await runner.run_one({'tool': 'vllm', 'rate': 10, 'round': 1}, directory,
                                                {'tokenizer': '/synthetic', 'probe_prompt': 'text', **PATHS},
                                                AFFINITY, False, ZERO)
            finally:
                for item in reversed(patches):
                    item.stop()
        with tempfile.TemporaryDirectory() as tmp:
            return asyncio.run(exercise(Path(tmp) / 'attempt'))

    def test_nonzero_exit_is_not_delivered_not_not_measured(self):
        result = self.attempt(exit_code=1, thread=.1)
        self.assertIs(result['delivered'], False)
        self.assertTrue(result['valid'])
        self.assertEqual(result['cells'], {})
        self.assertNotIn('instrument_error', result)
        records = [dict(result, round=n) for n in (1, 2, 3)]
        self.assertIs(a.aggregate(records, True)['delivery']['vllm/10'], False)

    def test_saturation_is_judged_on_the_busiest_thread(self):
        self.assertEqual(self.attempt(0, thread=.95)['shortfall_attribution'], 'tool')
        self.assertEqual(self.attempt(0, thread=.5, process=.99)['shortfall_attribution'], 'unexplained')

    def test_calibration_records_its_rule(self):
        from tests.test_calibration import calibration_from_trace
        self.assertEqual(calibration_from_trace(0)['rule'],
                         'back-to-back reads while the probe runs; bounded lead range')

class MetricsStatus(unittest.TestCase):
    """Pooled-context 500s after cancelled streams (sim http.go:416, 579-585, 630-634)."""
    BODY = 'vllm:num_requests_running{model_name="fixed-timing-sim-1"} 0\n'

    def session(self, responses):
        class Response:
            def __init__(self, status, body):
                self.status, self.body = status, body

            async def text(self):
                return self.body

        class Context:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *args):
                return False
        session = Mock()
        session.get.side_effect = [Context(Response(*r)) for r in responses]
        return session

    def run_checked(self, responses, final=False):
        async def exercise(log):
            session = self.session(responses)
            with patch.object(runner.asyncio, 'sleep', new=AsyncMock()):
                if final:
                    return await runner.final_counters(session, 'http://x', log)
                return await runner.scrape_checked(session, 'http://x', log, allow_uninstantiated=True)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'after-scrape-failures.jsonl'
            try:
                result = asyncio.run(exercise(log))
            except a.EvidenceError as exc:
                result = exc
            lines = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        return result, lines

    def test_transient_500s_then_200_succeed_and_are_logged(self):
        result, lines = self.run_checked([(500, self.BODY)] * 3 + [(200, self.BODY)])
        self.assertEqual(result['counters']['inflight'], 0)
        self.assertEqual([x['status'] for x in lines], [500, 500, 500])
        self.assertEqual(lines[0]['body_head'], self.BODY[:300])
        drained = 'vllm:request_success_total 3\nvllm:generation_tokens_total 150\n' + self.BODY
        result, lines = self.run_checked([(200, drained), (500, drained), (200, drained)], final=True)
        self.assertEqual(result['counters']['requests'], 3)  # drain-to-stable rule unchanged
        self.assertEqual(len(lines), 1)

    def test_gather_error_body_fails_at_once(self):
        body = 'An error has occurred while serving metrics:\n\ncollected metric ...'
        result, lines = self.run_checked([(500, body), (200, self.BODY)])
        self.assertIsInstance(result, a.EvidenceError)
        self.assertRegex(str(result), 'gather error')
        self.assertEqual(len(lines), 1)

    def test_exhausted_retries_fail_and_never_use_a_500_body(self):
        result, lines = self.run_checked([(500, self.BODY)] * runner.SCRAPE_ATTEMPTS)
        self.assertIsInstance(result, a.EvidenceError)
        self.assertRegex(str(result), 'did not return 200')
        self.assertEqual(len(lines), runner.SCRAPE_ATTEMPTS)
        self.assertGreater(sum(runner.SCRAPE_BACKOFF_S) + .3 * (runner.SCRAPE_ATTEMPTS - 5), 5)


if __name__ == '__main__':
    unittest.main()
