"""The report and compare commands on a synthetic panel directory: tables, decision, W, TSV."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ltbench import __main__ as cli, report as R
from ltbench.analysis import aggregate
from ltbench.spec import TOOLS

CELLS = R.CELLS


def cells(tool, rate, rnd, overrides=None):
    out = {}
    for name in CELLS:
        c = {'value': 'agrees', 'reported': 1.0, 'reference': 1.0, 'difference': 0.0, 'relative_difference': 0.0,
             'known_issue': None}
        if tool == 'guidellm' and name == 'b.rate':
            c = {'value': 'no entry', 'reported': 9.9, 'reference': None, 'difference': None}
        if tool == 'inference-perf' and name == 'c.gap':
            c = {'value': 'no entry', 'reported': .02, 'reference': None, 'difference': None}
        if tool == 'guidellm' and name.startswith('b.') and name[-3:] in ('p50', 'p99'):
            c.update(value='differs by documented definition', value_numpy_reference='wrong',
                     difference=2e-6 if name.endswith('p50') else 3e-4,
                     documented_rank={'rank': 300, 'value_at_rank': 1.0})
        out[name] = c
    for name, value in (overrides or {}).get((tool, rate, rnd), {}).items():
        out[name] = dict(out[name], **value)
    return out


def attempt(tool, rate, rnd, *, delivered=True, cells_override=None, did_not_complete=False, busiest=.5,
            completed=None):
    count = completed if completed is not None else 60 * rate
    span = {'lower_s': 59.8, 'upper_s': 60.1, 'f_lo_ns': 0, 'f_hi_ns': 100_000_000,
            'l_lo_ns': 59_900_000_000, 'l_hi_ns': 60_100_000_000}
    return {'tool': tool, 'rate': rate, 'round': rnd, 'valid': True, 'delivered': delivered,
            'did_not_complete': did_not_complete, 'exit_code': -15 if did_not_complete else 0,
            'deadline_basis': 'f_lo + 180 s' if did_not_complete else None,
            'shortfall_attribution': None if delivered else 'unexplained',
            'attempt_path': f'/x/run/r{rnd}-{rate}-{tool}/attempt-1',
            'L': {'pass': True, 'requests': float(count), 'tokens': 50.0 * count, 'residual_tokens': 0.0},
            'S': True, 'probe': {'ttft_p99_s': .2026, 'gap_p50_s': .0206},
            'completion_span': span,
            'busiest_tool_thread_send_window': busiest,
            'cells': {} if did_not_complete else cells(tool, rate, rnd, cells_override)}


def panel(tmp, *, short=None, dnf=None, overrides=None, s_fail_rate=1000, store=True, tamper=None):
    run = Path(tmp) / 'run'
    run.mkdir()
    rates = [r for r in (10, 50, 200, 500, 1000) if s_fail_rate is None or r < s_fail_rate]
    accepted = []
    for tool in TOOLS:
        for rate in rates:
            for rnd in (1, 2, 3):
                if (tool, rate) == short:
                    accepted.append(attempt(tool, rate, rnd, delivered=False, completed=int(0.9 * 58.82 * rate),
                                            cells_override=overrides))
                elif (tool, rate) == dnf:
                    accepted.append(attempt(tool, rate, rnd, delivered=False, did_not_complete=True))
                else:
                    accepted.append(attempt(tool, rate, rnd, cells_override=overrides))
    (run / 'accepted-attempts.json').write_text(json.dumps(accepted))
    if s_fail_rate is not None:
        for n, p99 in ((1, .2184), (2, .2163)):
            d = run / f'r1-{s_fail_rate}-aiperf' / f'attempt-{n}'
            d.mkdir(parents=True)
            (d / 'raw-gate-calculations.json').write_text(json.dumps(
                {'tool': 'aiperf', 'rate': s_fail_rate, 'round': 1, 'S': False, 'valid': False,
                 'attempt_path': str(d), 'probe': {'ttft_p99_s': p99, 'gap_p50_s': .0203}}))
    if store:
        decision = aggregate(accepted, True, s_fail_rate)
        if tamper:
            decision.update(tamper)
        (run / 'decision.json').write_text(json.dumps(decision))
    return run


class Report(unittest.TestCase):
    def test_findings_panel_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = panel(tmp, short=('guidellm', 500), dnf=('inference-perf', 500))
            data = R.load_run(run)
            rep = R.report(data)
            self.assertEqual(rep['warnings'], [])
            self.assertEqual(rep['decision'], 'FINDINGS')
            self.assertEqual(rep['valid_rates'], [10, 50, 200, 500])
            self.assertEqual(rep['knees'], {'vllm': 500, 'guidellm': 200, 'aiperf': 500, 'inference-perf': 200})
            self.assertEqual(rep['W'], 0)
            self.assertEqual(rep['W_numpy_reference'], 4)
            self.assertEqual(rep['cell_tally']['vllm']['agrees'], 40)
            self.assertEqual(rep['cell_tally']['guidellm'],
                             {'agrees': 20, 'differs by documented definition': 16, 'wrong': 0, 'unstable': 0,
                              'no entry': 4, 'not measured': 0})
            self.assertEqual(rep['cell_tally']['inference-perf']['no entry'], 13)
            self.assertEqual(rep['cell_tally']['inference-perf']['agrees'], 27)
            self.assertEqual([round(x) for x in rep['guidellm_percentile_abs_difference_us']['p99']], [300, 300])
            self.assertEqual(len(rep['gate_s_failures']), 2)
            text = R.text_report(rep)
            self.assertIn('| guidellm | yes | yes | yes | no | **200** |', text)
            self.assertIn('| inference-perf | yes | yes | yes | no | **200** |', text)
            self.assertIn('r1-1000-aiperf/attempt-1: probe TTFT p99 218.4 ms', text)
            self.assertIn('guidellm at 500: completed 26469 / 26469 / 26469 against a floor of 27939.5.', text)
            self.assertIn('**0** under the documented definitions, **4** under the plain numpy-percentile rule', text)
            self.assertIn('| inference-perf/500 | 1 | 30000.0 | True | -15 | f_lo + 180 s |', text)

    def test_unstable_cell_and_known_issue_listed(self):
        overrides = {('guidellm', 200, 3): {'b.ttft.p50': {'value': 'wrong', 'value_numpy_reference': 'agrees',
                                                           'known_issue': 'vllm-project/guidellm#1193'}}}
        with tempfile.TemporaryDirectory() as tmp:
            rep = R.report(R.load_run(panel(tmp, overrides=overrides)))
            self.assertEqual(rep['cells']['guidellm/200/b.ttft.p50'], 'unstable')
            self.assertNotIn('guidellm/200/b.ttft.p50', rep['cells_numpy_reference'])   # unstable both ways
            self.assertEqual(rep['known_issue_cells'][0]['known_issue'], 'vllm-project/guidellm#1193')
            self.assertIn('guidellm r3-200 b.ttft.p50: wrong (vllm-project/guidellm#1193)', R.text_report(rep))

    def test_wrong_cell_at_ten_counts_in_w(self):
        overrides = {(t, 10, n): {'c.ttft.p50': {'value': 'wrong'}} for t in ('aiperf',) for n in (1, 2, 3)}
        with tempfile.TemporaryDirectory() as tmp:
            rep = R.report(R.load_run(panel(tmp, overrides=overrides)))
            self.assertEqual(rep['W'], 1)
            self.assertEqual(rep['decision'], 'FINDINGS')

    def test_stored_decision_that_disagrees_is_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = panel(tmp, tamper={'decision': 'CLEAN', 'knees': {'vllm': 1000}})
            rep = R.report(R.load_run(run))
            self.assertTrue(any('decision' in w for w in rep['warnings']))
            self.assertTrue(any('knees' in w for w in rep['warnings']))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['report', str(run)]), 1)

    def test_too_few_valid_rates_is_not_measurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = R.report(R.load_run(panel(tmp, s_fail_rate=200)))
            self.assertEqual(rep['decision'], 'NOT-MEASURABLE')

    def test_tsv_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = panel(tmp)
            tsv, js = Path(tmp) / 'cells.tsv', Path(tmp) / 'r.json'
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(['report', str(run), '--tsv', str(tsv), '--json', str(js)]), 0)
            lines = tsv.read_text().splitlines()
            self.assertEqual(lines[0].split('\t'), R.TSV_COLUMNS)
            self.assertEqual(len(lines) - 1, 4 * 4 * 3 * len(CELLS))
            row = next(l.split('\t') for l in lines if l.startswith('guidellm\t10\t1\tb.ttft.p50\t'))
            self.assertEqual(row[4:6], ['differs by documented definition', 'wrong'])
            self.assertEqual(row[11:13], ['300', '1.0'])
            # Documented-definition differences and the two no-reference cells do not block CLEAN.
            self.assertEqual(json.loads(js.read_text())['decision'], 'CLEAN')
            self.assertIn('Decision row: **CLEAN**', out.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(['report', tmp]), 2)
            self.assertIn('not a finished panel directory', err.getvalue())


class Compare(unittest.TestCase):
    def test_identical_and_differing_panels(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            run_a = panel(a, short=('guidellm', 500))
            run_b = panel(b, short=('guidellm', 500))
            run_c = panel(c, overrides={('aiperf', 50, 2): {'c.gap': {'value': 'wrong'}}})
            same = R.compare(R.load_run(run_a), R.load_run(run_b))
            self.assertEqual(same['differences'], [])
            self.assertEqual(same['round_cells_compared'], 4 * 4 * 3 * len(CELLS))
            diff = R.compare(R.load_run(run_a), R.load_run(run_c))
            self.assertIn('knees', ' '.join(diff['differences']))
            self.assertIn('delivery guidellm/500: False | True', diff['differences'])
            self.assertIn('cell aiperf/50/c.gap: agrees | unstable', diff['differences'])
            self.assertEqual(diff['round_cells_differing'], 1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['compare', str(run_a), str(run_b)]), 0)
                self.assertEqual(cli.main(['compare', str(run_a), str(run_c)]), 1)


if __name__ == '__main__':
    unittest.main()
