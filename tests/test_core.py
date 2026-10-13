"""CPU-only arithmetic, parser, command and failure-path tests; no network traffic."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from ltbench import analysis as a
from ltbench import normalize
from ltbench import runner
from ltbench.probe import SSE
from ltbench.spec import MODEL, RATES, TOOLS, schedule
from tests.helpers import command
from ltbench.telemetry import counters, cpu_accounting


def probe_row(**updates):
    return dict(dict(error=None, done=True, ttft_s=.202, e2e_s=1.190,
                     usage_tokens=50, gaps_s=[.0201] * 49, start_ns=1, end_ns=2), **updates)


class Arithmetic(unittest.TestCase):
    def test_linear_percentiles_and_offset(self):
        self.assertEqual(a.percentile(list(range(1, 21)), 50), 10.5)
        self.assertAlmostEqual(a.percentile(list(range(1, 21)), 99), 19.81)
        self.assertEqual(a.sample_offset(list(range(1, 21)), 11, 50), [.5, .5])
        self.assertEqual(a.cell(11, 10.5, 1e-6)['value'], 'wrong')

    def test_no_nan_or_empty_sample(self):
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(a.EvidenceError):
                a.percentile(values, 50)

    def test_and_tolerances(self):
        self.assertEqual(a.cell(.206, .2, .1, .005)['value'], 'agrees')
        self.assertEqual(a.cell(.221, .2, .1, .005)['value'], 'wrong')
        self.assertEqual(a.cell(None, .2, .1)['value'], 'no entry')

    def test_counter_subtraction_and_one_request_tolerance(self):
        before = {'requests': 5, 'tokens': 250}
        after = {'requests': 16, 'tokens': 850}
        result = a.gate_l(before, after, [probe_row()])
        self.assertEqual(result['requests'], 10)
        self.assertEqual(result['residual_tokens'], 50)
        self.assertTrue(result['pass'])
        after['tokens'] += 1
        self.assertFalse(a.gate_l(before, after, [probe_row()])['pass'])

    def test_missing_probe_cannot_pass(self):
        for rows in ([], [probe_row(done=False)], [probe_row(usage_tokens=None)], [probe_row(gaps_s=[.02])]):
            with self.assertRaises(a.EvidenceError):
                a.gate_s(rows)

    def test_server_tail_gate(self):
        self.assertTrue(a.gate_s([probe_row()]))
        self.assertFalse(a.gate_s([probe_row(ttft_s=.211)]))

    def test_gate_p_median_and_tail_are_separate_checks(self):
        self.assertTrue(a.gate_p([probe_row(ttft_s=t) for t in (.199, .202, .209)]))
        for ttfts in ((.199, .199, .209), (.202, .206, .209), (.202, .202, .212)):
            self.assertFalse(a.gate_p([probe_row(ttft_s=t) for t in ttfts]))
        for median in (.200, .205):
            self.assertTrue(a.gate_p([probe_row(ttft_s=median)] * 98 + [probe_row(ttft_s=.210)] * 2))

    def test_gate_p_gap_and_every_request_token_count(self):
        for gap in (.020, .021):
            self.assertTrue(a.gate_p([probe_row(gaps_s=[gap] * 49)]))
        for gap in (.01999, .02101):
            self.assertFalse(a.gate_p([probe_row(gaps_s=[gap] * 49)]))
        self.assertFalse(a.gate_p([probe_row()] * 99 + [probe_row(usage_tokens=49)]))

    def test_delivery_never_uses_fixed_total_as_evidence(self):
        with self.assertRaisesRegex(a.EvidenceError, 'bounds'):
            a.delivery('vllm', 10, 600)
        self.assertTrue(a.delivery('vllm', 10, 600, {'lower_s': 61.16, 'upper_s': 63.16}))
        self.assertFalse(a.delivery('vllm', 10, 600, {'lower_s': 63.17, 'upper_s': 65.17}))
        self.assertTrue(a.delivery('guidellm', 10, 559))
        self.assertFalse(a.delivery('guidellm', 10, 558))
        self.assertFalse(a.delivery('vllm', 10, 0, did_not_complete=True))

    def test_missing_round_and_duplicate_round(self):
        result = a.aggregate([], True)
        self.assertEqual(result['decision'], 'FINDINGS')
        self.assertTrue(all(x is None for x in result['knees'].values()))
        duplicate = {'tool': 'vllm', 'rate': 10, 'round': 1, 'valid': True}
        with self.assertRaises(a.EvidenceError):
            a.aggregate([duplicate, duplicate], True)

    def test_dropped_rates_precede_findings(self):
        self.assertEqual(a.aggregate([], True, 200)['decision'], 'NOT-MEASURABLE')



class Instrument(unittest.TestCase):
    def test_sse_split_utf8_and_coalescing(self):
        decoder = SSE()
        raw = 'data: {"choices":[{"text":"é"}]}\r\n\r\ndata: [DONE]\n\n'.encode()
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(json.loads(events[0])['choices'][0]['text'], 'é')
        self.assertEqual(events[1], b'[DONE]')

    def test_counter_label_sum_not_histograms(self):
        text = '\n'.join(['vllm:request_success_total{reason="length"} 20',
                          'vllm:request_success_total{reason="stop"} 1',
                          'vllm:generation_tokens_total{model="x"} 1050',
                          'vllm:num_requests_running{model="x"} 2',
                          'vllm:request_success_total_fake 500'])
        self.assertEqual(counters(text), {'requests': 21, 'tokens': 1050, 'inflight': 2})
        with self.assertRaises(a.EvidenceError):
            counters('unrelated_counter 0')

    def test_empty_telemetry_never_quiet(self):
        with self.assertRaises(a.EvidenceError):
            cpu_accounting([], {'tool': [], 'server': [], 'probe': []})

    def test_pinned_fresh_server_has_uninstantiated_counter_vectors(self):
        fresh = f'vllm:num_requests_running{{model_name="{MODEL}"}} 0\n'
        self.assertEqual(counters(fresh, allow_uninstantiated=True), {'requests': 0, 'tokens': 0, 'inflight': 0})
        with self.assertRaises(a.EvidenceError):
            counters(fresh)
        with self.assertRaises(a.EvidenceError):
            counters('unrelated_counter 0', allow_uninstantiated=True)

    def test_no_side_effect_before_preflight_stop(self):
        with patch.object(runner.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(a.EvidenceError, 'loaded from its file'):
                runner.preflight({}, None)
            spawn.assert_not_called()

    def test_owned_process_constructor_failure_cleans_child(self):
        process = Mock(pid=987654321)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, 'Popen', return_value=process), patch.object(runner.os, 'killpg') as kill:
                with patch.object(runner, 'write_json', side_effect=[None, OSError('disk full')]):
                    with self.assertRaises(OSError):
                        runner.OwnedProcess(['fake'], {}, [0], Path(directory), 'tool')
                self.assertEqual(kill.call_args.args[0], process.pid)
                process.wait.assert_called_once()

    def test_exited_leader_does_not_hide_a_live_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=987654321)
            process.poll.return_value = None
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                 patch.object(runner.os, 'killpg') as kill:
                owned = runner.OwnedProcess(['fake'], {}, [0], Path(tmp), 'tool')
                with patch.object(owned, 'group_alive', side_effect=[True, True, False, False]), \
                     patch.object(owned, 'signal_descendants') as descendants:
                    owned.stop()
                self.assertEqual([call.args[1] for call in kill.call_args_list], [runner.signal.SIGTERM])
                self.assertEqual([call.args[0] for call in descendants.call_args_list],
                                 [runner.signal.SIGTERM, runner.signal.SIGKILL])

    def test_preflight_rejects_a_manifest_other_than_its_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_b = Path(tmp) / 'b.json'
            path_b.write_text('{"other": true}')
            with self.assertRaisesRegex(a.EvidenceError, 'loaded from its file'):
                runner.preflight({'other': False}, path_b)

    def test_failed_zero_request_dry_run_cannot_issue_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'dry'
            idle = {'L': {'requests': 0, 'pass': True}, 'probe': {}}
            failed = {'exit_code': 1, 'did_not_complete': False,
                      'L': {'requests': 0, 'pass': True}}
            calibration = {'u_offset_ns': 0, 'l_offset_ns': 0}
            with patch.object(runner, 'calibration_phase', new=AsyncMock(return_value=(calibration, True))), \
                 patch.object(runner, 'clock_state', return_value={}), \
                 patch.object(runner, 'probe_requests', return_value=[probe_row()]), \
                 patch.object(runner, 'gate_p', return_value=True), \
                 patch.object(runner, 'run_one', new=AsyncMock(side_effect=[idle, failed])) as attempt:
                with self.assertRaisesRegex(a.EvidenceError, 'Dry run failed'):
                    asyncio.run(runner.dry_run({'probe_prompt': 'text'}, output, {'server': [0]}, 8188, 'test-sha'))
                self.assertFalse((output / 'dry-run-receipt.json').exists())
                # Gate P is read from a full-harness attempt with an idle dummy tool.
                self.assertEqual(attempt.call_args_list[0].kwargs['command']['expected_workers'], 0)
                self.assertTrue(attempt.call_args_list[1].kwargs['worker_check'])


class Commands(unittest.TestCase):
    def test_schedule_is_three_complete_rounds_and_ascending_rates(self):
        rows = schedule(42)
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows, schedule(42))
        for round_id in (1, 2, 3):
            subset = [r for r in rows if r['round'] == round_id]
            self.assertEqual([r['rate'] for r in subset], sorted(r['rate'] for r in subset))
            for rate in RATES:
                self.assertEqual({r['tool'] for r in subset if r['rate'] == rate}, set(TOOLS))

    def test_completions_and_worker_settings(self):
        commands = {t: command(t) for t in TOOLS}
        self.assertIn('openai', commands['vllm']['argv'])
        self.assertIn('/v1/completions', commands['vllm']['argv'])
        self.assertIn('600', commands['vllm']['argv'])
        self.assertIn('request_format=/v1/completions', ' '.join(commands['guidellm']['argv']))
        self.assertEqual(commands['guidellm']['env']['GUIDELLM__MAX_WORKER_PROCESSES'], '24')
        argv = commands['aiperf']['argv']
        self.assertEqual(argv[argv.index('--benchmark-grace-period') + 1], '5')
        config = json.loads(commands['inference-perf']['files']['inference-perf.yaml'])
        self.assertEqual(config['load']['num_workers'], 24)
        self.assertEqual(config['data']['output_distribution']['mean'], 50)
        self.assertTrue(config['report']['request_lifecycle']['per_stage'])

    def test_run_refuses_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({'root': tmp, 'vllm_env': tmp, 'hf_home': tmp}))
            with patch.object(runner, 'preflight'), patch.object(runner.asyncio, 'run') as started, \
                    patch.dict(runner.os.environ):
                with self.assertRaisesRegex(a.EvidenceError, 'overwrite'):
                    runner.run('dry-run', manifest, tmp)
                started.assert_not_called()
            runner.set_run_prefix(None)


class NativeExports(unittest.TestCase):
    def inference_fixture(self, path):
        # Summary deliberately has no achieved_rate, as in pinned v0.7.0.
        summary = {'load_summary': {'count': 3, 'schedule_delay': {}},
                   'successes': {'count': 2, 'output_tokens': {'total': 100},
                   'latency': {'time_to_first_token': {'median': .202, 'p99': .202},
                               'request_latency': {'median': 1.19, 'p99': 1.19},
                               'inter_token_latency': {'median': .0201}}}}
        raw = [{'error': None, 'start_time': t,
                'computed_metrics': {'time_to_first_token': .202, 'request_latency': 1.19},
                'info': {'response_metrics': {'server_usage': {'completion_tokens': 50}}}}
               for t in (10., 11.)]
        raw.append({'error': 'failed', 'start_time': 12.})
        (path / 'summary_lifecycle_metrics.json').write_text(json.dumps(summary))
        (path / 'per_request_lifecycle_metrics.json').write_text(json.dumps(raw))
        (path / 'stage_0_lifecycle_metrics.json').write_text(json.dumps({'load_summary': {'achieved_rate': 1.5}}))

    def test_inference_perf_reads_stage_rate_and_checks_all_send_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.inference_fixture(path)
            result = normalize.normalize('inference-perf', path)
            self.assertEqual(result['rate'], 1.5)
            self.assertEqual(result['all_start_times'], [10., 11., 12.])
            self.assertEqual(len(result['requests']), 2)
            cells = a.comparisons('inference-perf', result, 2, [probe_row()], False)
            self.assertEqual(cells['b.rate']['reference'], 1.5)
            self.assertEqual(cells['b.rate']['value'], 'agrees')
            result['rate'] = 1.5149
            self.assertEqual(a.comparisons('inference-perf', result, 2, [probe_row()], False)['b.rate']['value'], 'agrees')
            result['rate'] = 1.5151
            self.assertEqual(a.comparisons('inference-perf', result, 2, [probe_row()], False)['b.rate']['value'], 'wrong')

    def test_inference_perf_requires_exactly_one_stage_with_native_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.inference_fixture(path)
            stage = path / 'stage_0_lifecycle_metrics.json'
            stage.unlink()
            with self.assertRaisesRegex(a.EvidenceError, 'single.*stage 0'):
                normalize.normalize('inference-perf', path)
            stage.write_text('{"load_summary": {}}')
            with self.assertRaisesRegex(a.EvidenceError, 'achieved_rate'):
                normalize.normalize('inference-perf', path)
            (path / 'stage_1_lifecycle_metrics.json').write_text('{}')
            with self.assertRaisesRegex(a.EvidenceError, 'single.*stage 0'):
                normalize.normalize('inference-perf', path)

    def test_vllm_failed_request_is_not_a_percentile_sample(self):
        doc = {'completed': 1, 'failed': 1, 'duration': 1.2, 'request_throughput': 1 / 1.2,
               'total_output_tokens': 50, 'start_times': [10, 11], 'ttfts': [.2, 0],
               'latencies': [1.18, 0], 'itls': [[.02] * 49, []], 'output_lens': [50, 0],
               'errors': ['', 'failed'], 'median_itl_ms': 20,
               'p50_ttft_ms': 200, 'p99_ttft_ms': 200, 'p50_e2el_ms': 1180, 'p99_e2el_ms': 1180}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'vllm.json').write_text(json.dumps(doc))
            result = normalize.normalize('vllm', path)
            self.assertEqual(len(result['requests']), 1)
            self.assertEqual(result['ttft_p50_s'], .2)
            doc['errors'].pop()
            (path / 'vllm.json').write_text(json.dumps(doc))
            with self.assertRaises(a.EvidenceError):
                normalize.normalize('vllm', path)


if __name__ == '__main__':
    unittest.main()
