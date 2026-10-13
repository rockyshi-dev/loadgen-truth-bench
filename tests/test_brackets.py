"""Delivery rule from sampled counters, the online clock and attempt orchestration; synthetic data, no traffic."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from ltbench import analysis as a
from ltbench import runner
from ltbench import verify_install
from ltbench.spec import PINS, RATES, TOOLS, digest
from tests.helpers import PATHS
from tests.test_core import probe_row

ZERO = {'u_offset_ns': 0, 'l_offset_ns': 0}


def sample(seconds, count, width=.001):
    """A scrape that starts at `seconds` and ends `width` later."""
    start = round(seconds * 1e9)
    return {'start_ns': start, 'end_ns': start + round(width * 1e9), 'counters': {'requests': count}}


def probe(seconds):
    return {'end_ns': round(seconds * 1e9), 'done': True, 'error': None}


def timeline(events, probes=(), calibration=ZERO, baseline=0):
    result = a.CompletionTimeline(baseline, calibration)
    ends = sorted(p['end_ns'] for p in probes)
    for row in events:
        result.observe(row, ends)
    return result


class SampledCompletion(unittest.TestCase):
    def test_baseline_probe_subtraction_and_last_tool_increase(self):
        probes = [probe(101.5), probe(162.5), probe(164.5)]
        events = [sample(100, 7), sample(100.5, 7), sample(101, 7)]
        events += [sample(101.5 + .5 * i, 9) for i in range(121)]       # 101.5 .. 161.5
        events += [sample(162, 10), sample(162.5, 11), sample(163, 11), sample(163.5, 11),
                   sample(164, 11), sample(164.5, 12), sample(165, 12)]
        result = timeline(events, probes, baseline=7)
        bounds = result.span(2)
        # Latest U<1 is the 101.0 sample; L>=1 first at the 101.5 sample's end.
        self.assertEqual((bounds['f_lo_ns'], bounds['f_hi_ns']), (101_000_000_000, 101_501_000_000))
        self.assertEqual((bounds['l_lo_ns'], bounds['l_hi_ns']), (161_500_000_000, 162_001_000_000))
        self.assertAlmostEqual(bounds['lower_s'], 59.999)
        self.assertAlmostEqual(bounds['upper_s'], 61.001)

    def test_probe_only_upper_can_start_clock_but_cannot_supply_span(self):
        result = timeline([sample(1, 0), sample(1.5, 1)], [probe(1.5005)], {'u_offset_ns': 0, 'l_offset_ns': 1_000_000})
        self.assertEqual(result.f_lo_ns, 1_000_000_000)  # possible, not confirmed
        self.assertIsNone(result.span(0)['lower_s'])
        self.assertFalse(a.delivery('vllm', 10, 0, result.span(0)))

    def test_one_increase_has_one_sample_bounds(self):
        result = timeline([sample(0, 0), sample(.5, 600)])
        span = result.span(600)
        self.assertAlmostEqual(span['lower_s'], -.501)
        self.assertAlmostEqual(span['upper_s'], .501)

    def test_threshold_three_way_and_timeout_override(self):
        for tool in ('vllm', 'inference-perf'):
            for lower, upper, expected in ((61.16, 63.16, True), (63.16, 65.16, 'unstable'),
                                            (63.160001, 65.160001, False), (62, 64, 'unstable')):
                self.assertEqual(a.delivery(tool, 10, 600, {'lower_s': lower, 'upper_s': upper}), expected)
            self.assertFalse(a.delivery(tool, 10, 600, None, did_not_complete=True))
        self.assertTrue(a.delivery('aiperf', 10, 559))
        self.assertFalse(a.delivery('aiperf', 10, 558))

    def test_broken_counter_evidence_rejects(self):
        for invalid in (sample(2, -1), sample(2, float('nan')), sample(2, .5)):
            with self.assertRaises(a.EvidenceError):
                a.CompletionTimeline(0, ZERO).observe(invalid, [])
        result = timeline([sample(0, 0), sample(.5, 2)])
        for invalid in (sample(1, 1), sample(.5, 3), sample(.4, 3)):
            with self.assertRaises(a.EvidenceError):
                result.observe(invalid, [])
        with self.assertRaisesRegex(a.EvidenceError, 'calibration'):
            a.CompletionTimeline(0, None)
        with self.assertRaisesRegex(a.EvidenceError, 'calibration'):
            a.CompletionTimeline(0, {'u_offset_ns': 2, 'l_offset_ns': 1})

    def test_errored_sample_is_counted_not_used(self):
        result = timeline([sample(0, 0)])
        self.assertIsNone(result.observe({'start_ns': 1, 'end_ns': 2, 'error': 'timeout'}, []))
        self.assertEqual(result.dropped_samples, 1)
        self.assertEqual(len(result.rows), 1)

    def test_missing_baseline_cannot_supply_first_completion_interval(self):
        with self.assertRaisesRegex(a.EvidenceError, 'initial zero-tool sample'):
            a.CompletionTimeline(0, ZERO).observe(sample(1, 600), [])

    def test_unstable_delivery_survives_round_aggregation_and_blocks_clean(self):
        names = ['b.completed', 'b.rate', 'b.ttft.p50', 'b.ttft.p99', 'b.e2e.p50',
                 'b.e2e.p99', 'c.ttft.p50', 'c.e2e.p50', 'c.gap', 'c.tokens']
        records = [{'tool': tool, 'rate': rate, 'round': n, 'valid': True, 'delivered': True,
                    'cells': {name: {'value': 'agrees'} for name in names}}
                   for tool in TOOLS for rate in RATES for n in (1, 2, 3)]
        self.assertEqual(a.aggregate(records, True)['decision'], 'CLEAN')
        for r in records:
            if r['tool'] == 'vllm' and r['rate'] == 50:
                r['delivered'] = 'unstable'
        candidate = a.aggregate(records, True)
        self.assertEqual(candidate['delivery']['vllm/50'], 'unstable')
        self.assertEqual(candidate['knees']['vllm'], 10)
        self.assertEqual(candidate['decision'], 'FINDINGS')
        records[4]['delivered'] = True
        self.assertEqual(a.aggregate(records, True)['delivery']['vllm/50'], 'unstable')


def attempt_patches(test, directory, before, after, telemetry, probe_rows, outcome, accounting=None):
    """Patch every process/HTTP boundary of run_one; the probe log is a real file."""
    owned = Mock()
    owned.pid = 4242
    owned.known = {}
    owned.process.poll.return_value = 0
    session = AsyncMock()
    fake_aiohttp = types.SimpleNamespace(ClientSession=Mock(return_value=session),
                                        ClientTimeout=Mock(), TCPConnector=Mock())

    async def fake_counter_reads(url, stop, output, probe_log, on_sample=None, **kwargs):
        for event in telemetry:
            on_sample(event)
        await stop.wait()
        return telemetry

    async def fake_cpu_scans(groups, stop, output, **kwargs):
        await stop.wait()
        return []

    def fake_probe(url, prompt, directory, affinity, name='probe'):
        with (directory / f'{name}.jsonl').open('w') as stream:
            stream.write(json.dumps({'grid_origin_ns': 0, 'period_ns': 100_000_000, 'written_ns': 0}) + '\n')
            for row in probe_rows:
                stream.write(json.dumps(row) + '\n')
        return owned
    accounting = accounting or {'pass': True, 'busiest_tool_thread_window_fraction': .99}
    return [patch.dict('sys.modules', {'aiohttp': fake_aiohttp}),
            patch.object(runner, 'check_port'), patch.object(runner, 'nofile_limit', return_value={}),
            patch.object(runner, 'clock_state', return_value={}),
            patch.object(runner, 'OwnedProcess', return_value=owned),
            patch.object(runner, 'ready', new=AsyncMock()),
            patch.object(runner, 'scrape_checked', new=AsyncMock(return_value=before)),
            patch.object(runner, 'final_counters', new=AsyncMock(return_value=after)),
            patch.object(runner, 'probe_process', side_effect=fake_probe),
            patch.object(runner, 'wait_probe_warm', new=AsyncMock()),
            patch.object(runner, 'stop_probe', new=AsyncMock()),
            patch.object(runner, 'counter_reads', side_effect=fake_counter_reads),
            patch.object(runner, 'cpu_scans', side_effect=fake_cpu_scans),
            patch.object(runner, 'supervise', new=AsyncMock(return_value=outcome)),
            patch.object(runner, 'cpu_accounting', return_value=accounting),
            patch.object(runner, 'normalize', return_value={'corrected': {}}),
            patch.object(runner, 'comparisons', return_value={'synthetic': {'value': 'agrees'}})]


class RuntimeClock(unittest.TestCase):
    def test_attempt_wires_brackets_and_persists_spans(self):
        async def exercise(directory, zero_tool=False):
            p = probe_row(start_ns=500_000_000, end_ns=1_500_000_000)
            events = [sample(.5 * k, 2 if k >= 3 else 0) for k in range(1, 130)]
            events += [sample(65, 3), sample(65.5, 3)]
            before, after = sample(0, 0), sample(66, 3)
            before['counters'].update(tokens=0, inflight=0)
            after['counters'].update(tokens=150, inflight=0)
            if zero_tool:
                p['end_ns'] = 1_400_000_000
                events = [sample(.5 * k, 1 if k >= 3 else 0) for k in range(1, 132)]
                after['counters'].update(requests=1, tokens=50)
            outcome = {'timed_out': False, 'tool_exit_ns': 66 * 10**9, 'deadline_start_ns': 1}
            with tempfile.TemporaryDirectory():
                patches = attempt_patches(self, directory, before, after, events, [p], outcome)
                for item in patches:
                    item.start()
                try:
                    # launch_ns is read from the real clock: make it precede the samples.
                    with patch.object(runner.time, 'monotonic_ns', return_value=0):
                        result = await runner.run_one({'tool': 'vllm', 'rate': 10, 'round': 1}, directory,
                                                      {'tokenizer': '/synthetic', 'probe_prompt': 'text', **PATHS},
                                                      {'server': [1], 'tool': [0], 'probe': [2]}, False, ZERO)
                finally:
                    for item in reversed(patches):
                        item.stop()
            if zero_tool:
                self.assertEqual(result['delivered'], False)
                self.assertEqual(result['cells'], {})
                self.assertIsNone(result['completion_span']['upper_s'])
            else:
                self.assertEqual(result['delivered'], 'unstable')
                self.assertEqual(result['completion_span']['f_lo_ns'], 1_000_000_000)
                self.assertAlmostEqual(result['completion_span']['lower_s'], 62.999)
                self.assertAlmostEqual(result['completion_span']['upper_s'], 64.001)
                self.assertTrue(result['normalized'])
            self.assertEqual(result['calibration'], ZERO)
            self.assertEqual(json.loads((directory / 'completion-span.json').read_text()), result['completion_span'])
        with tempfile.TemporaryDirectory() as tmp:
            for zero_tool in (False, True):
                with self.subTest(zero_tool=zero_tool):
                    asyncio.run(exercise(Path(tmp) / str(zero_tool), zero_tool))

    def supervise(self, f_lo, times, probe_alive=True, telemetry_done=False, exits=False, exit_after=None):
        import threading
        released = threading.Event()
        if exits:
            released.set()
        if exit_after is not None:
            threading.Timer(exit_after, released.set).start()
        tool = Mock()
        tool.process.wait = lambda: released.wait(5)
        tool.stop = Mock(side_effect=released.set)
        clock = Mock()
        clock.timeline.f_lo_ns = f_lo
        clock.refresh.return_value = f_lo
        clock.timeline.possible_completion = f_lo is not None
        clock.timeline.certain_completion = f_lo is not None
        probe = Mock()
        probe.process.poll.return_value = None if probe_alive else 1
        task = Mock()
        task.done.return_value = telemetry_done
        task.result.side_effect = a.EvidenceError('bad counter')
        values, last = iter(times), [times[-1]]

        def now():
            last[0] = next(values, last[0])
            return last[0]

        async def exercise():
            try:
                return await runner.supervise(tool, clock, probe, task, 0)
            except a.EvidenceError as exc:
                return exc
            finally:
                released.set()
        with patch.object(runner, 'SUPERVISION_TICK_S', .01), \
             patch.object(runner.time, 'monotonic_ns', side_effect=now):
            return asyncio.run(exercise()), tool

    def test_deadline_uses_f_lo_and_inclusive_180_second_boundary(self):
        result, tool = self.supervise(100 * 10**9, [279_999_999_999, 280_000_000_000])
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['deadline_basis'], 'f_lo + 180 s')
        self.assertEqual(result['deadline_start_ns'], 100 * 10**9)
        tool.stop.assert_called_once()

    def test_pre_completion_startup_is_not_the_deadline(self):
        result, tool = self.supervise(None, [179 * 10**9, 599 * 10**9], exit_after=.2)
        self.assertFalse(result['timed_out'])
        tool.stop.assert_not_called()
        result, tool = self.supervise(None, [599 * 10**9, 600 * 10**9])
        self.assertTrue(result['timed_out'])
        self.assertIn('no certain tool completion (L >= 1) within 600 s', result['deadline_basis'])
        self.assertEqual(result['deadline_start_ns'], 0)          # the cap runs from launch
        self.assertEqual(result['deadline_rule'], 'launch + 600 s without a certain completion')

    def test_normal_exit_is_reported_by_the_waitpid_thread(self):
        result, tool = self.supervise(None, [5], exits=True)
        self.assertFalse(result['timed_out'])
        tool.stop.assert_not_called()

    def test_collector_failure_cannot_leave_tool_waiting_forever(self):
        result, _ = self.supervise(None, [1], telemetry_done=True)
        self.assertRegex(str(result), 'bad counter')
        result, _ = self.supervise(None, [1], probe_alive=False)
        self.assertRegex(str(result), 'Probe exited')


class Preflight(unittest.TestCase):
    def test_vllm_local_build_tag_keeps_upstream_source_verification(self):
        distribution = types.SimpleNamespace(version='0.30.0+cu129', locate_file=lambda name: '/synthetic/' + name)
        with patch.object(verify_install.importlib.metadata, 'distribution', return_value=distribution), \
             patch.object(verify_install, 'digest', return_value='source-hash') as checked, \
             patch.object(verify_install.Path, 'read_text', return_value='{"vllm":{"client.py":"source-hash"}}'), \
             patch.object(verify_install.sys, 'argv', ['verify_install.py', 'vllm']), \
             patch('builtins.print'):
            verify_install.main()
            checked.assert_called_once_with(Path('/synthetic/client.py'))
            distribution.version = '0.30.0'
            verify_install.main()
            distribution.version = '0.31.0+cu129'
            with self.assertRaisesRegex(ValueError, 'wrong installed version'):
                verify_install.main()
            with patch.object(verify_install.sys, 'argv', ['verify_install.py', 'guidellm']):
                distribution.version = '0.8.0+local'
                with self.assertRaisesRegex(ValueError, 'wrong installed version'):
                    verify_install.main()
        with patch.object(verify_install.importlib.metadata, 'distribution', return_value=distribution), \
             patch.object(verify_install, 'digest', return_value='other-hash'), \
             patch.object(verify_install.Path, 'read_text', return_value='{"vllm":{"client.py":"source-hash"}}'), \
             patch.object(verify_install.sys, 'argv', ['verify_install.py', 'vllm']):
            distribution.version = '0.30.0+cu129'
            with self.assertRaisesRegex(ValueError, 'does not match pinned commit'):
                verify_install.main()

    def manifest(self, tmp, **updates):
        from ltbench.construct import implementation_hashes
        tokenizer = Path(tmp) / 'tok'
        tokenizer.mkdir()
        (tokenizer / 'tokenizer.json').write_text('{}')
        manifest = {'implementation_sha256': implementation_hashes(),
                    'pins': {k: list(v) for k, v in PINS.items()},
                    'tokenizer': str(tokenizer),
                    'tokenizer_sha256': {'tokenizer.json': digest(tokenizer / 'tokenizer.json')},
                    'root': str(Path(tmp) / 'work'), 'vllm_env': str(Path(tmp) / 'vllm'), 'hf_home': str(tmp)}
        manifest.update(updates)
        path = Path(tmp) / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return manifest, path

    def test_changed_code_pins_or_tokenizer_reject_before_any_process(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'Popen') as spawn, \
                patch.object(runner.subprocess, 'run') as started:
            manifest, path = self.manifest(tmp, implementation_sha256={'runner.py': 'old'})
            with self.assertRaisesRegex(a.EvidenceError, 'Implementation changed'):
                runner.preflight(manifest, path)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'Popen') as spawn, \
                patch.object(runner.subprocess, 'run') as started:
            manifest, path = self.manifest(tmp, pins={})
            with self.assertRaisesRegex(a.EvidenceError, 'pins'):
                runner.preflight(manifest, path)
            spawn.assert_not_called()
            started.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path = self.manifest(tmp)
            (Path(manifest['tokenizer']) / 'tokenizer.json').write_text('{"edited": 1}')
            with self.assertRaisesRegex(a.EvidenceError, 'Tokenizer bytes'):
                runner.preflight(manifest, path)

    def test_installation_must_be_complete_and_aiperf_tokenizer_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path = self.manifest(tmp)
            with self.assertRaisesRegex(a.EvidenceError, 'Installation is incomplete'):
                runner.preflight(manifest, path)
            (Path(manifest['root']) / 'setup').mkdir(parents=True)
            (Path(manifest['root']) / 'setup/INSTALL_READY').touch()
            with self.assertRaisesRegex(a.EvidenceError, 'Run with'):
                runner.preflight(manifest, path)
            with patch.object(runner, 'aiperf_tokenizer_identity') as identity, \
                    patch.object(runner, 'check_environments') as envs:
                runner.preflight(manifest, path)
            envs.assert_called_once_with(Path(manifest['root']), Path(manifest['vllm_env']))
            identity.assert_called_once_with(manifest['tokenizer'], root=Path(manifest['root']),
                                             vllm_env=Path(manifest['vllm_env']), hf_home=manifest['hf_home'])

    def test_environment_check_names_the_missing_piece(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, vllm_env = Path(tmp) / 'work', Path(tmp) / 'shared-vllm'
            (root / 'setup').mkdir(parents=True)
            envs = {t: (vllm_env if t == 'vllm' else root / 'envs' / t) for t in TOOLS}
            for tool, env in envs.items():
                (env / 'bin').mkdir(parents=True)
                (env / 'bin' / tool).touch()
                (root / 'setup' / f'{tool}-verify.json').write_text(json.dumps(
                    {'files': {'x.py': {'path': str(env / 'lib/x.py')}}}))
            prefix = root / 'envs' / 'inference-perf'
            runner.check_environments(root, vllm_env, prefix)
            with self.assertRaisesRegex(a.EvidenceError, 'Run with'):
                runner.check_environments(root, vllm_env, Path(tmp))
            with self.assertRaisesRegex(a.EvidenceError, 'vllm: no executable'):
                runner.check_environments(root, root / 'envs' / 'vllm', prefix)
            (root / 'envs/vllm/bin').mkdir(parents=True)
            (root / 'envs/vllm/bin/vllm').touch()
            with self.assertRaisesRegex(a.EvidenceError, 'vllm: the environment setup.sh verified'):
                runner.check_environments(root, root / 'envs' / 'vllm', prefix)
            (root / 'setup' / 'aiperf-verify.json').unlink()
            with self.assertRaisesRegex(a.EvidenceError, 'aiperf-verify.json missing'):
                runner.check_environments(root, vllm_env, prefix)


if __name__ == '__main__':
    unittest.main()
