"""Result tables of a finished panel, recomputed from its raw output; and a cell-by-cell comparison of two panels.

Reads `accepted-attempts.json` (every accepted attempt with its gates, completion span, CPU
accounting and cells), `decision.json` (the decision the run stored) and the attempt
directories' `raw-gate-calculations.json` / `attempt-error.json` (for attempts that were not
accepted, e.g. the gate S failures that drop a rate). The decision rule is applied again to the
accepted attempts (`analysis.aggregate`), and a difference from the stored decision is a warning.
No measurement code runs.
"""
import json
import math
from pathlib import Path
from statistics import mean

from ltbench.analysis import aggregate
from ltbench.spec import RATES, TOOLS

CELLS = ('b.completed', 'b.rate', 'b.ttft.p50', 'b.ttft.p99', 'b.e2e.p50', 'b.e2e.p99',
         'c.ttft.p50', 'c.e2e.p50', 'c.gap', 'c.tokens')
VALUES = ('agrees', 'differs by documented definition', 'wrong', 'unstable', 'no entry', 'not measured')
TSV_COLUMNS = ['tool', 'rate', 'round', 'cell', 'value', 'value_numpy_reference', 'reported', 'reference',
               'difference', 'relative_difference', 'known_issue', 'documented_rank', 'documented_value_at_rank']
SPAN_THRESHOLD_S = 63.16
SEND_S = 60 - 1.18


class ReportError(ValueError):
    pass


def load_run(run):
    run = Path(run)
    path = run / 'accepted-attempts.json'
    if not path.is_file():
        raise ReportError(f'{path} not found: not a finished panel directory')
    accepted = json.loads(path.read_text())
    stored = json.loads((run / 'decision.json').read_text()) if (run / 'decision.json').is_file() else None
    others = []
    for raw in sorted(run.glob('r*-*-*/attempt-*/raw-gate-calculations.json')):
        others.append(json.loads(raw.read_text()))
    for err in sorted(run.glob('r*-*-*/attempt-*/attempt-error.json')):
        row = json.loads(err.read_text())
        row['attempt_error'] = True
        others.append(row)
    return {'run': run, 'accepted': accepted, 'stored': stored, 'attempts': others}


def dropped_rate(data):
    """The rate from which gate S dropped every higher rate, or None.

    Taken from the stored decision's valid rates; checked against the accepted attempts
    (a rate with no accepted attempt and all higher rates likewise)."""
    present = {r['rate'] for r in data['accepted']}
    inferred = next((rate for rate in RATES if rate not in present
                     and all(r not in present for r in RATES if r >= rate)), None)
    stored = data['stored']
    if stored and 'valid_rates' in stored:
        missing = [r for r in RATES if r not in stored['valid_rates']]
        return (min(missing) if missing else None), inferred
    return inferred, inferred


def recompute(data):
    dropped, inferred = dropped_rate(data)
    # A panel starts only with a dry-run receipt whose gate P passed.
    candidate = aggregate(data['accepted'], True, dropped)
    warnings = []
    if dropped != inferred:
        warnings.append(f'stored valid rates imply dropping from {dropped}; accepted attempts imply {inferred}')
    if data['stored'] is not None:
        for key in ('decision', 'valid_rates', 'knees', 'delivery', 'cells', 'W', 'W_numpy_reference'):
            if key in data['stored'] and _canon(data['stored'][key]) != _canon(candidate.get(key)):
                warnings.append(f'recomputed {key} differs from the stored decision.json')
    return candidate, warnings


def _canon(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _fmt(x, digits=2):
    return '-' if x is None else f'{x:.{digits}f}'


def _range(values, digits=2, scale=1.0):
    values = [v * scale for v in values if v is not None]
    if not values:
        return '-'
    lo, hi = min(values), max(values)
    return _fmt(lo, digits) if round(lo, digits) == round(hi, digits) else f'{_fmt(lo, digits)}-{_fmt(hi, digits)}'


def runs_of(data, tool, rate):
    return sorted((r for r in data['accepted'] if r['tool'] == tool and r['rate'] == rate), key=lambda r: r['round'])


def delivery_word(value):
    return {True: 'yes', False: 'no'}.get(value, str(value))


def cell_rows(data):
    """One row per tool, rate, round and cell, with every raw difference."""
    rows = []
    for r in sorted(data['accepted'], key=lambda r: (r['tool'], r['rate'], r['round'])):
        for name, c in sorted(r.get('cells', {}).items()):
            rank = c.get('documented_rank') or {}
            rows.append([r['tool'], r['rate'], r['round'], name, c.get('value'),
                         c.get('value_numpy_reference', c.get('value')), c.get('reported'), c.get('reference'),
                         c.get('difference'), c.get('relative_difference'), c.get('known_issue'),
                         rank.get('rank'), rank.get('value_at_rank')])
    return rows


def write_tsv(data, path):
    lines = ['\t'.join(TSV_COLUMNS)]
    for row in cell_rows(data):
        lines.append('\t'.join('' if v is None else str(v) for v in row))
    Path(path).write_text('\n'.join(lines) + '\n')


def report(data):
    candidate, warnings = recompute(data)
    rates = candidate['valid_rates']
    out = {'run': str(data['run']), 'decision': candidate['decision'], 'valid_rates': rates,
           'knees': candidate.get('knees'), 'W': candidate.get('W'),
           'W_numpy_reference': candidate.get('W_numpy_reference'), 'warnings': warnings}
    # Gate S failures (attempts not accepted).
    out['gate_s_failures'] = [
        {'attempt': str(Path(a['attempt_path']).relative_to(Path(a['attempt_path']).parents[1]))
         if 'attempt_path' in a else None,
         'tool': a.get('tool'), 'rate': a.get('rate'), 'round': a.get('round'),
         'probe_ttft_p99_ms': (a.get('probe') or {}).get('ttft_p99_s', math.nan) * 1000 if a.get('probe') else None,
         'probe_gap_p50_ms': (a.get('probe') or {}).get('gap_p50_s', math.nan) * 1000 if a.get('probe') else None,
         'probe_failure': a.get('probe_failure')}
        for a in data['attempts'] if a.get('S') is False]
    out['delivery'] = {f'{t}/{r}': candidate['delivery'].get(f'{t}/{r}') for t in TOOLS for r in rates}
    # Fixed request count: completion spans of every accepted attempt.
    spans = {}
    for tool in ('vllm', 'inference-perf'):
        rows = [r for rate in rates for r in runs_of(data, tool, rate)
                if r.get('delivered') is True and r.get('completion_span', {}).get('lower_s') is not None]
        sp = [r['completion_span'] for r in rows]
        spans[tool] = {'attempts': len(sp),
                       'lower_s': [min((s['lower_s'] for s in sp), default=None), max((s['lower_s'] for s in sp), default=None)],
                       'upper_s': [min((s['upper_s'] for s in sp), default=None), max((s['upper_s'] for s in sp), default=None)],
                       'first_end_bracket_s': [(s['f_hi_ns'] - s['f_lo_ns']) / 1e9 for s in sp],
                       'last_end_bracket_s': [(s['l_hi_ns'] - s['l_lo_ns']) / 1e9 for s in sp]}
    out['spans'] = spans
    counts = {}
    for tool in TOOLS:
        for rate in rates:
            counts[f'{tool}/{rate}'] = [
                {'round': r['round'], 'completed': (r.get('L') or {}).get('requests'),
                 'delivered': r.get('delivered'), 'did_not_complete': r.get('did_not_complete'),
                 'exit_code': r.get('exit_code'), 'deadline_basis': r.get('deadline_basis'),
                 'shortfall_attribution': r.get('shortfall_attribution'),
                 'span_s': [(r.get('completion_span') or {}).get('lower_s'), (r.get('completion_span') or {}).get('upper_s')],
                 'probe_ttft_p99_ms': (r.get('probe') or {}).get('ttft_p99_s', math.nan) * 1000,
                 'busiest_thread': r.get('busiest_tool_thread_send_window')}
                for r in runs_of(data, tool, rate)]
    out['attempts'] = counts
    # Reporting cells per tool.
    tally = {}
    for tool in TOOLS:
        t = {v: 0 for v in VALUES}
        for rate in rates:
            for name in CELLS:
                t[candidate['cells'].get(f'{tool}/{rate}/{name}', 'not measured')] += 1
        tally[tool] = t
    out['cell_tally'] = tally
    out['cells'] = {k: v for k, v in candidate['cells'].items()}
    out['cells_numpy_reference'] = candidate.get('cells_numpy_reference', {})
    # GuideLLM percentile differences against numpy, and round-cells tagged with a known issue.
    diffs = {'p50': [], 'p99': []}
    tagged = []
    for r in data['accepted']:
        if r['rate'] not in rates:
            continue
        for name, c in r.get('cells', {}).items():
            if c.get('known_issue'):
                tagged.append({'tool': r['tool'], 'rate': r['rate'], 'round': r['round'], 'cell': name,
                               'value': c.get('value'), 'known_issue': c['known_issue'],
                               'difference': c.get('difference')})
            if r['tool'] == 'guidellm' and name.startswith('b.') and name[-3:] in ('p50', 'p99') \
                    and c.get('difference') is not None:
                diffs[name[-3:]].append(abs(c['difference']))
    out['guidellm_percentile_abs_difference_us'] = {
        q: [min(v) * 1e6, max(v) * 1e6] if v else None for q, v in diffs.items()}
    out['known_issue_cells'] = sorted(tagged, key=lambda x: (x['tool'], x['rate'], x['round'], x['cell']))
    return out


def text_report(rep):
    rates = rep['valid_rates']
    lines = [f"# Load generators against a server with known timing: {Path(rep['run']).name}", '',
             f"Decision row: **{rep['decision']}**", '']
    for w in rep['warnings']:
        lines.append(f'WARNING: {w}')
    if rep['warnings']:
        lines.append('')
    lines += ['## Valid rates', '', ', '.join(map(str, rates)) + ' requests per second.']
    if rep['gate_s_failures']:
        lines.append('')
        lines.append('Gate S failures (probe TTFT p99 bound 210 ms, median gap bound 21 ms):')
        for f in rep['gate_s_failures']:
            if f['probe_failure']:
                lines.append(f"- {f['attempt']}: probe request failed ({f['probe_failure']})")
            else:
                lines.append(f"- {f['attempt']}: probe TTFT p99 {_fmt(f['probe_ttft_p99_ms'], 1)} ms, "
                             f"median gap {_fmt(f['probe_gap_p50_ms'], 2)} ms")
    lines += ['', '## Delivery', '', '| Tool | ' + ' | '.join(map(str, rates)) + ' | Knee |',
              '|---|' + '---|' * (len(rates) + 1)]
    for tool in TOOLS:
        lines.append(f'| {tool} | ' + ' | '.join(delivery_word(rep['delivery'][f'{tool}/{r}']) for r in rates)
                     + f" | **{rep['knees'].get(tool)}** |")
    lines.append('')
    for tool, s in rep['spans'].items():
        if s['attempts']:
            lines.append(f"- {tool}, delivered attempts ({s['attempts']}): first-to-last completion span lower bound "
                         f"{_range(s['lower_s'])} s, upper bound {_range(s['upper_s'])} s (threshold {SPAN_THRESHOLD_S} s); "
                         f"first completion bracketed to {_range(s['first_end_bracket_s'])} s, "
                         f"last to {_range(s['last_end_bracket_s'])} s.")
    for tool in ('guidellm', 'aiperf'):
        for rate in rates:
            rows = rep['attempts'][f'{tool}/{rate}']
            lines.append(f"- {tool} at {rate}: completed {' / '.join(str(int(r['completed'])) if r['completed'] is not None else '-' for r in rows)} "
                         f"against a floor of {0.95 * rate * SEND_S:.1f}.")
    nd = [(k, r) for k, rows in rep['attempts'].items() for r in rows if r['delivered'] is not True]
    if nd:
        lines += ['', 'Attempts not delivered:', '']
        lines += ['| Tool/rate | Round | Completed | Did not complete | Exit | Deadline | Span (s) | Probe TTFT p99 (ms) | Busiest thread | Attribution |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for k, r in nd:
            span = r['span_s']
            span_txt = '-' if span[0] is None else f'[{span[0]:.2f}, {span[1]:.2f}]'
            lines.append(f"| {k} | {r['round']} | {r['completed']} | {r['did_not_complete']} | {r['exit_code']} | "
                         f"{r['deadline_basis'] or '-'} | {span_txt} | {_fmt(r['probe_ttft_p99_ms'], 1)} | "
                         f"{_fmt(r['busiest_thread'])} | {r['shortfall_attribution'] or '-'} |")
    lines += ['', 'Busiest tool thread over the send window, fraction of one core: mean over rounds (range).', '',
              '| Tool | ' + ' | '.join(map(str, rates)) + ' |', '|---|' + '---|' * len(rates)]
    for tool in TOOLS:
        cells = []
        for rate in rates:
            vals = [r['busiest_thread'] for r in rep['attempts'][f'{tool}/{rate}'] if r['busiest_thread'] is not None]
            cells.append(f'{mean(vals):.2f} ({_range(vals)})' if vals else '-')
        lines.append(f'| {tool} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Reporting cells (b: the tool against its own raw data; c: the tool against the server)', '',
              '| Tool | ' + ' | '.join(VALUES) + ' |', '|---|' + '---|' * len(VALUES)]
    for tool in TOOLS:
        lines.append(f'| {tool} | ' + ' | '.join(str(rep['cell_tally'][tool][v]) for v in VALUES) + ' |')
    lines += ['', f"W (wrong cells at R = 10): **{rep['W']}** under the documented definitions, "
                  f"**{rep['W_numpy_reference']}** under the plain numpy-percentile rule.", '']
    d = rep['guidellm_percentile_abs_difference_us']
    if d.get('p50') or d.get('p99'):
        lines.append('GuideLLM reported percentile minus numpy percentile of its own values, absolute: '
                     + ', '.join(f'{q} {lo:.0f}-{hi:.0f} us' for q, v in d.items() if v for lo, hi in [v]) + '.')
        lines.append('')
    other = [(k, v) for k, v in rep['cells'].items() if v != 'agrees' and int(k.split('/')[1]) in rates]
    if other:
        lines += ['Cells other than "agrees" (value under the numpy rule in brackets where it differs):', '']
        for k, v in other:
            numpy_value = rep['cells_numpy_reference'].get(k)
            lines.append(f'- {k}: {v}' + (f' [{numpy_value}]' if numpy_value else ''))
        lines.append('')
    if rep['known_issue_cells']:
        lines += ['Round-cells tagged with a known upstream issue:', '']
        for c in rep['known_issue_cells']:
            lines.append(f"- {c['tool']} r{c['round']}-{c['rate']} {c['cell']}: {c['value']} ({c['known_issue']}), "
                         f"difference {c['difference']}")
    return '\n'.join(lines).rstrip() + '\n'


def compare(a, b):
    """Differences between two panels: decision, knees, delivery, aggregated and per-round cells."""
    ra, rb = report(a), report(b)
    diffs = []
    for key in ('decision', 'valid_rates', 'knees', 'W', 'W_numpy_reference'):
        if _canon(ra[key]) != _canon(rb[key]):
            diffs.append(f'{key}: {ra[key]} | {rb[key]}')
    for key in sorted(set(ra['delivery']) | set(rb['delivery'])):
        if ra['delivery'].get(key) != rb['delivery'].get(key):
            diffs.append(f"delivery {key}: {ra['delivery'].get(key)} | {rb['delivery'].get(key)}")
    for key in sorted(set(ra['cells']) | set(rb['cells'])):
        if ra['cells'].get(key) != rb['cells'].get(key):
            diffs.append(f"cell {key}: {ra['cells'].get(key)} | {rb['cells'].get(key)}")
    rows_a = {tuple(r[:4]): r for r in cell_rows(a)}
    rows_b = {tuple(r[:4]): r for r in cell_rows(b)}
    per_round = 0
    for key in sorted(set(rows_a) | set(rows_b), key=str):
        va = rows_a.get(key, [None] * 6)[4:6]
        vb = rows_b.get(key, [None] * 6)[4:6]
        if va != vb:
            per_round += 1
            diffs.append(f"round-cell {'/'.join(map(str, key))}: {va} | {vb}")
    return {'differences': diffs, 'round_cells_compared': len(set(rows_a) | set(rows_b)),
            'round_cells_differing': per_round}
