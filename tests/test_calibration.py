"""Completion brackets with measured counter offsets, and sampling jitter: synthetic clocks and
counters only, never HTTP traffic. The perfect-tool cases check that a tool sending for exactly
60 s reads as delivered."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import bisect
import unittest

from ltbench import analysis as a
from tests.test_brackets import ZERO, probe, sample, timeline

S = 1_000_000_000
MS = 1_000_000


def probe_trace(stop_ns, lead_ns, period_ns=100 * MS):
    """Probe from t=0, one request every 100 ms, each ending 1.18 s later;
    the server counts each one lead_ns before the probe logs its end."""
    starts = range(0, stop_ns, period_ns)
    ends = [s + 1_180 * MS for s in starts]
    return ends, [e - lead_ns for e in ends]


def calibration_scrapes(lead_ns, seconds=25, period_ns=3 * MS, read_ns=1 * MS, b2b_stop_ns=None,
                        post_delay_ns=2_430 * MS):
    """Calibration trace: one pre-start read, back-to-back 1 ms reads every
    3 ms while the probe runs (until b2b_stop_ns, default: past the last
    increment), then one post-drain read post_delay_ns later."""
    ends, counted = probe_trace(seconds * S, lead_ns)
    counted.sort()

    def read(t, phase):
        return {'start_ns': t, 'end_ns': t + read_ns, 'phase': phase,
                'counters': {'requests': bisect.bisect_right(counted, t + read_ns // 2)}}
    scrapes = [read(-period_ns, 'pre-start')]
    t = 0
    last = max(counted) + period_ns if b2b_stop_ns is None else b2b_stop_ns
    while t <= last:
        scrapes.append(read(t, 'back-to-back'))
        t += period_ns
    scrapes.append(read(max(t, max(counted)) + post_delay_ns, 'post-drain'))
    rows = [{'end_ns': e, 'done': True, 'error': None} for e in ends]
    return scrapes, rows


def calibration_from_trace(lead_ns, **kwargs):
    return a.calibrate(*calibration_scrapes(lead_ns, **kwargs))


def perfect_tool(tool_start_s, rate, lead_ns=2 * MS, period_s=.5, jitter=()):
    """A perfect tool: 60 R requests, one every 1/R s, each 1.18 s
    long; the probe stops when the tool exits 2 s after its last completion."""
    n = 60 * rate
    tool = [int((tool_start_s + 1.18 + i / rate) * S) for i in range(n)]
    tool_exit = tool[-1] + 2 * S
    ends, counted = probe_trace(tool_exit, lead_ns)
    events = sorted(counted + tool)
    calibration = calibration_from_trace(lead_ns)
    rows = [{'end_ns': e, 'done': True, 'error': None} for e in ends]
    samples = []
    k = 0
    stop = max(events) + 3 * S
    while k * period_s * S < stop:
        start = int(k * period_s * S) + (jitter[k] if k < len(jitter) else 0)
        read = start + MS // 2
        samples.append({'start_ns': start, 'end_ns': start + MS,
                        'counters': {'requests': bisect.bisect_right(events, read)}})
        k += 1
    result = a.build_timeline(0, samples, rows, calibration)
    result.check(int(tool_start_s * S), n)
    span = result.span(n)
    return {'true_span_s': (tool[-1] - tool[0]) / S, 'first_ns': tool[0], 'span': span,
            'calibration': calibration, 'verdict': a.delivery('vllm', rate, n, span)}


class PerfectTool(unittest.TestCase):
    def test_perfect_tool_is_delivered_at_every_start_and_rate(self):
        for start in (.5, 5., 15.):
            for rate in (10, 1000):
                with self.subTest(start=start, rate=rate):
                    out = perfect_tool(start, rate)
                    span = out['span']
                    self.assertIs(out['verdict'], True)
                    self.assertLessEqual(span['lower_s'], 60.0)
                    self.assertGreaterEqual(span['upper_s'], 60.0)
                    self.assertLessEqual(span['lower_s'], out['true_span_s'])
                    self.assertGreaterEqual(span['upper_s'], out['true_span_s'])
                    # The deadline origin f_lo is a bound on the tool's own first
                    # completion, not the probe's start.
                    self.assertGreaterEqual(span['f_lo_ns'], int(start * S))
                    self.assertLessEqual(span['f_lo_ns'], out['first_ns'])
                    self.assertLessEqual(span['f_hi_ns'] - span['f_lo_ns'], 2 * S)

    def test_lagging_counter_is_handled_by_its_own_calibration(self):
        out = perfect_tool(5., 10, lead_ns=-7 * MS)
        self.assertIs(out['verdict'], True)
        self.assertLess(out['calibration']['u_offset_ns'], -7 * MS)
        self.assertGreater(out['calibration']['l_offset_ns'], -7 * MS)

    def test_ordinary_scheduling_jitter_is_not_a_failure(self):
        jitter = [0, 30 * MS, 0, 450 * MS, 0, 499 * MS]
        out = perfect_tool(5., 10, jitter=jitter)
        self.assertIs(out['verdict'], True)


class Calibration(unittest.TestCase):
    def test_measured_offsets_contain_the_true_lead_with_margin(self):
        cal = calibration_from_trace(2 * MS)
        self.assertLessEqual(cal['measured_lead_min_ns'], 2 * MS)
        self.assertGreaterEqual(cal['measured_lead_max_ns'], 2 * MS)
        self.assertEqual(cal['u_offset_ns'], cal['measured_lead_min_ns'] - a.CALIBRATION_MARGIN_NS)
        self.assertEqual(cal['l_offset_ns'], cal['measured_lead_max_ns'] + a.CALIBRATION_MARGIN_NS)
        self.assertLessEqual(cal['max_read_window_ns'], 4 * MS)
        self.assertEqual(cal['probe_requests'], 250)
        self.assertEqual(cal['probe_requests_used'], 250)
        self.assertTrue(cal['bound_pass'])

    def test_uncounted_or_failed_probe_requests_reject(self):
        def tagged(*rows):
            phases = ['pre-start'] + ['back-to-back'] * (len(rows) - 2) + ['post-drain']
            return [dict(r, phase=ph) for r, ph in zip(rows, phases)]
        rows = [probe(1.0), probe(1.04)]
        scrapes = tagged(sample(0, 0), sample(.5, 0), sample(1.02, 1), sample(2, 1))
        with self.assertRaisesRegex(a.EvidenceError, 'server counted 1'):
            a.calibrate(scrapes, rows)
        with self.assertRaises(a.ProbeFailure):
            a.calibrate(scrapes, [probe(1.0), {'end_ns': 1, 'done': False, 'error': 'timeout'}])
        with self.assertRaisesRegex(a.EvidenceError, 'decreased'):
            a.calibrate(tagged(sample(0, 1), sample(.5, 3), sample(1, 2), sample(2, 2)), [probe(.5)])
        with self.assertRaisesRegex(a.EvidenceError, 'tagged'):
            a.calibrate([sample(0, 0), sample(1, 1), sample(2, 1)], [probe(.5)])


class CalibrationReadSelection(unittest.TestCase):
    """Only back-to-back reads while the probe runs, only increments between
    the first and the last of them; the 60 ms bound includes both margins."""

    def test_post_drain_read_does_not_widen_the_offsets(self):
        # A drained read 2.43 s after the last back-to-back read would bracket the
        # requests that completed in the drain and set both ends.
        # Here the back-to-back reads stop 3 s before the last completion.
        scrapes, rows = calibration_scrapes(2 * MS, b2b_stop_ns=23 * S)
        cal = a.calibrate(scrapes, rows)
        self.assertLessEqual(cal['max_read_window_ns'], 4 * MS)
        self.assertLessEqual(cal['measured_lead_max_ns'] - cal['measured_lead_min_ns'], 8 * MS)
        self.assertLess(cal['used_increments'][1], len(rows))
        self.assertEqual(cal['probe_requests'], len(rows))
        self.assertEqual(cal['post_drain_reads'], 1)
        self.assertTrue(cal['bound_pass'])
        # The old global rule over every read would have taken the drain window.
        widest = max(r['end_ns'] for r in scrapes) - scrapes[-2]['start_ns']
        self.assertGreater(widest, 2 * S)

    def test_bound_includes_both_margins(self):
        scrapes, rows = calibration_scrapes(2 * MS)
        cal = a.calibrate(scrapes, rows)
        self.assertEqual(cal['lead_range_with_margins_ns'],
                         cal['measured_lead_max_ns'] - cal['measured_lead_min_ns'] + 2 * a.CALIBRATION_MARGIN_NS)
        # One slow 41 ms read covering an increment: 40 ms + 2 x 10 ms > 60 ms.
        slow = [dict(r) for r in scrapes]
        i = next(i for i, r in enumerate(slow) if r['counters']['requests'] == 1)
        slow[i - 1]['end_ns'] = slow[i - 1]['start_ns'] + MS
        slow[i]['start_ns'] = slow[i - 1]['start_ns'] + MS
        slow[i]['end_ns'] = slow[i]['start_ns'] + 41 * MS
        slow = [r for j, r in enumerate(slow) if j <= i or r['start_ns'] > slow[i]['end_ns']]
        failed = a.calibrate(slow, rows)
        self.assertGreater(failed['lead_range_with_margins_ns'], a.CALIBRATION_BOUND_NS)
        self.assertFalse(failed['bound_pass'])


class Brackets(unittest.TestCase):
    def test_upper_uses_scrape_start_and_lower_uses_scrape_end(self):
        # One probe end inside the scrape window: U excludes it, L includes it.
        result = timeline([sample(1, 0, width=.1), sample(2, 1, width=.1)], [probe(2.05)])
        row = result.rows[1]
        self.assertEqual((row['tool_completed_lower'], row['tool_completed_upper']), (0, 1))
        self.assertEqual(result.f_lo_ns, 1 * S)

    def test_offsets_shift_the_probe_subtraction(self):
        cal = {'u_offset_ns': 5 * MS, 'l_offset_ns': 20 * MS}
        result = timeline([sample(1, 0), sample(2, 1)], [probe(2.004)], cal)
        self.assertEqual(result.rows[1]['tool_completed_upper'], 0)   # 2.004 <= 2 + 5 ms
        self.assertEqual(result.rows[1]['tool_completed_lower'], 0)
        result = timeline([sample(1, 0), sample(2, 1)], [probe(2.008)], cal)
        self.assertEqual(result.rows[1]['tool_completed_upper'], 1)
        self.assertEqual(result.rows[1]['tool_completed_lower'], 0)

    def test_gap_bound_is_one_second_between_scrape_starts(self):
        result = timeline([sample(0, 0), sample(.5, 0)])
        result.observe(sample(1.5, 0), [])                      # exactly 1 s: accepted
        with self.assertRaisesRegex(a.EvidenceError, 'gap exceeds'):
            result.observe(sample(2.500000001, 0), [])

    def test_errored_telemetry_row_does_not_break_sorting(self):
        rows = [sample(0, 0), {'start_ns': S // 2, 'end_ns': S // 2 + MS, 'error': 'timeout'},
                {'error': 'no times at all'}, sample(.9, 0), sample(1.4, 1)]
        result = a.build_timeline(0, rows, [], ZERO)
        self.assertEqual(result.dropped_samples, 2)
        self.assertEqual(len(result.rows), 3)

    def test_consistency_check_detects_wrong_offsets(self):
        # Probe completion counted 30 ms after its logged end, but the offsets
        # claim the counter leads: before the tool starts, U goes negative.
        cal = {'u_offset_ns': 0, 'l_offset_ns': 10 * MS}
        result = timeline([sample(1, 0), sample(1.5, 0), sample(2, 1)], [probe(1.49)], cal)
        with self.assertRaisesRegex(a.EvidenceError, 'offsets violated'):
            result.check(tool_launch_ns=3 * S, total=0)
        good = timeline([sample(1, 0), sample(1.5, 1), sample(2, 1)], [probe(1.49)], cal)
        self.assertEqual(good.check(3 * S, 0)['pre_launch_rows'], 3)

    def test_earlier_lower_bound_above_later_upper_bound_rejects(self):
        cal = {'u_offset_ns': 0, 'l_offset_ns': 0}
        result = timeline([sample(0, 0), sample(.5, 2), sample(1, 2)], [probe(.95)], cal)
        with self.assertRaisesRegex(a.EvidenceError, 'earlier L exceeds later U'):
            result.check(0, 2)

    def test_online_rows_have_upper_only_and_span_needs_offline_rebuild(self):
        result = a.CompletionTimeline(0, ZERO)
        result.observe(sample(1, 0), [], online=True)
        row = result.observe(sample(1.5, 1), [], online=True)
        self.assertIsNone(row['tool_completed_lower'])
        self.assertEqual(result.f_lo_ns, 1 * S)
        with self.assertRaisesRegex(a.EvidenceError, 'Full drained probe log'):
            result.span(0)

    def test_zero_requests_and_missing_final_crossing(self):
        result = timeline([sample(1, 0), sample(1.5, 0)])
        self.assertIsNone(result.span(0)['lower_s'])
        self.assertFalse(a.delivery('vllm', 10, 0, result.span(0)))
        with self.assertRaisesRegex(a.EvidenceError, 'do not bracket'):
            timeline([sample(0, 0), sample(.5, 1)]).span(2)


if __name__ == '__main__':
    unittest.main()
