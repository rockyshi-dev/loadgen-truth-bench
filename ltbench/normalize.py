"""Read pinned native exports. Fail on schema drift rather than invent values."""
from datetime import datetime
import json
from pathlib import Path
from ltbench.analysis import EvidenceError


def get(obj, dotted):
    for key in dotted.split('.'):
        obj = obj[key]
    return obj


def load(path):
    return json.loads(Path(path).read_text())


def normalize(tool, directory):
    path = Path(directory)
    try:
        return globals()[tool.replace('-', '_')](path)
    except EvidenceError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, FileNotFoundError) as exc:
        raise EvidenceError(f'{tool} export schema/required field mismatch: {exc}') from exc


def vllm(path):
    s = load(path / 'vllm.json')
    arrays = ['start_times', 'ttfts', 'latencies', 'output_lens', 'errors', 'itls']
    if len({len(s[k]) for k in arrays}) != 1:
        raise EvidenceError('vLLM per-request arrays are not aligned')
    rows = []
    for i, error in enumerate(s['errors']):
        if error or s['output_lens'][i] == 0:
            continue
        rows.append({'ttft_s': s['ttfts'][i], 'e2e_s': s['latencies'][i],
                     'start_time': s['start_times'][i], 'tokens': s['output_lens'][i]})
    return {'completed': s['completed'], 'incomplete': 0, 'rate': s['request_throughput'],
            'duration_s': s['duration'], 'all_start_times': s['start_times'], 'requests': rows,
            'output_tokens': [r['tokens'] for r in rows] + [s['total_output_tokens'] / s['completed']],
            **{f'{m}_p{q}_s': s[f'p{q}_{native}_ms'] / 1000
               for m, native in [('ttft', 'ttft'), ('e2e', 'e2el')] for q in (50, 99)},
            'gap_p50_s': s['median_itl_ms'] / 1000,
            'corrected': {k: v for k, v in s.items() if 'queue' in k}}


def guidellm(path):
    doc = load(path / 'guidellm.json')
    if len(doc['benchmarks']) != 1:
        raise EvidenceError('Expected exactly one GuideLLM benchmark')
    s = doc['benchmarks'][0]
    metrics = s['metrics']
    rows = []
    for r in s['requests']['successful']:
        t = r['info']['timings']
        rows.append({'ttft_s': t['first_token_iteration'] - t['request_start'],
                     'e2e_s': t['request_end'] - t['request_start'],
                     'start_time': t['request_start'], 'tokens': r['output_tokens']})
    return {'completed': metrics['request_totals']['successful'],
            'incomplete': metrics['request_totals']['incomplete'],
            'rate': get(metrics, 'requests_per_second.successful.mean'),
            'duration_s': None, 'all_start_times': [r['start_time'] for r in rows],
            'requests': rows,
            'output_tokens': [r['tokens'] for r in rows] + [get(metrics, 'output_token_count.successful.mean')],
            **{f'{m}_p{q}_s': get(metrics, f'{native}.successful.percentiles.p{q}') * scale
               for m, native, scale in [('ttft', 'time_to_first_token_ms', .001),
                                        ('e2e', 'request_latency', 1)] for q in (50, 99)},
            'gap_p50_s': get(metrics, 'inter_token_latency_ms.successful.percentiles.p50') / 1000,
            # GuideLLM's own units and counts for the nearest-rank classification:
            # the per-request values it exports and the count its summary gives.
            'percentile_native': {
                metric: {'unit': unit, 'summary_count': get(metrics, f'{native}.successful.count'),
                         'values': [r[native] for r in s['requests']['successful']],
                         'reported': {q: get(metrics, f'{native}.successful.percentiles.p{q}') for q in (50, 99)}}
                for metric, native, unit in [('ttft', 'time_to_first_token_ms', 'ms'),
                                             ('e2e', 'request_latency', 's')]},
            'counts': {'export_successful': len(s['requests']['successful']),
                       'summary_successful': metrics['request_totals']['successful'],
                       'export_incomplete': len(s['requests'].get('incomplete') or []),
                       'summary_incomplete': metrics['request_totals']['incomplete']},
            'corrected': {k: metrics[k] for k in ('request_dispatch_delay', 'request_scheduled_latency')}}


def aiperf(path):
    s = load(path / 'profile_export_aiperf.json')
    rows = []
    starts = []
    for line in (path / 'profile_export.jsonl').read_text().splitlines():
        r = json.loads(line)
        starts.append(r['metadata']['request_start_ns'] / 1e9)
        if r.get('error') or r['metadata'].get('error'):
            continue
        m = r['metrics']
        rows.append({'ttft_s': m['time_to_first_token']['value'] / 1000,
                     'e2e_s': m['request_latency']['value'] / 1000,
                     'start_time': starts[-1], 'tokens': m['output_token_count']['value']})
    # start_time/end_time are the phase window (records_manager.py:2429-2430,
    # metrics_json_exporter.py:60-69); absent fields give no window and no tag.
    window = None
    if s.get('start_time') and s.get('end_time'):
        window = (datetime.fromisoformat(s['end_time'])
                  - datetime.fromisoformat(s['start_time'])).total_seconds()
    return {'completed': s['request_count']['avg'], 'incomplete': 0,
            'rate': s['request_throughput']['avg'], 'duration_s': s['benchmark_duration']['avg'],
            'phase_window_s': window,
            'all_start_times': starts, 'requests': rows,
            'output_tokens': [r['tokens'] for r in rows] + [s['output_token_count']['avg']],
            **{f'{m}_p{q}_s': s[native][f'p{q}'] / 1000
               for m, native in [('ttft', 'time_to_first_token'), ('e2e', 'request_latency')] for q in (50, 99)},
            'gap_p50_s': s['inter_token_latency']['p50'] / 1000,
            'corrected': {k: s[k] for k in ('effective_latency', 'credit_to_start_latency') if k in s}}


def inference_perf(path):
    s = load(path / 'summary_lifecycle_metrics.json')
    stages = sorted(path.glob('stage_*_lifecycle_metrics.json'))
    if stages != [path / 'stage_0_lifecycle_metrics.json']:
        raise EvidenceError('Expected exactly the single inference-perf stage 0 report')
    stage = load(stages[0])
    raw = load(path / 'per_request_lifecycle_metrics.json')
    rows = []
    for r in raw:
        if r['error'] is not None:
            continue
        m = r['computed_metrics']
        response = r['info']['response_metrics']
        # Summary token totals prefer server usage; computed_metrics.output_tokens
        # is the client count even when use_server_output_tokens is enabled.
        tokens = response['server_usage']['completion_tokens']
        rows.append({'ttft_s': m['time_to_first_token'], 'e2e_s': m['request_latency'],
                     'start_time': r['start_time'], 'tokens': tokens})
    return {'completed': s['successes']['count'], 'incomplete': 0,
            'rate': stage['load_summary']['achieved_rate'], 'duration_s': None,
            'rate_source': 'stage_0_lifecycle_metrics.json:load_summary.achieved_rate',
            'all_start_times': [r['start_time'] for r in raw], 'requests': rows,
            'output_tokens': [r['tokens'] for r in rows] + [s['successes']['output_tokens']['total'] / s['successes']['count']],
            **{f'{m}_p{q}_s': get(s, f'successes.latency.{native}.{key}')
               for m, native in [('ttft', 'time_to_first_token'), ('e2e', 'request_latency')]
               for q, key in [(50, 'median'), (99, 'p99')]},
            'gap_p50_s': get(s, 'successes.latency.inter_token_latency.median'),
            'corrected': {'schedule_delay': s['load_summary']['schedule_delay']}}
