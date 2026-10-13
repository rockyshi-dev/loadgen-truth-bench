"""Pure arithmetic: gates, counter calibration, read placement, completion brackets, reference
calculations, cell classification and the decision rule. No benchmark is run by this module.

Missing evidence never becomes an agreeing cell or a passing gate.
"""
import bisect
from fractions import Fraction
import math
from statistics import median

from ltbench.spec import RATES, TOOLS


class EvidenceError(ValueError):
    pass


def percentile(values, q):
    values = sorted(float(x) for x in values)
    if not values or not all(math.isfinite(x) for x in values):
        raise EvidenceError('Missing or nonfinite percentile sample')
    index = (len(values) - 1) * q / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def sample_offset(values, reported, q):
    """Index offset relative to numpy's fractional index; ties give an interval."""
    xs = sorted(values)
    expected = (len(xs) - 1) * q / 100
    lo, hi = bisect.bisect_left(xs, reported), bisect.bisect_right(xs, reported)
    if lo != hi:
        return [lo - expected, hi - 1 - expected]
    if lo == 0 or lo == len(xs):
        return {'outside_sample': True, 'insertion_offset': lo - expected}
    rank = lo - 1 + (reported - xs[lo - 1]) / (xs[lo] - xs[lo - 1])
    return [rank - expected, rank - expected]


def cell(reported, reference, relative, absolute=0.0, known=None):
    if reported is None or reference is None:
        return {'value': 'no entry', 'reported': reported, 'reference': reference,
                'difference': None}
    if not all(math.isfinite(float(x)) for x in (reported, reference)):
        raise EvidenceError('Nonfinite metric')
    delta = reported - reference
    wrong = abs(delta) > relative * abs(reference) and abs(delta) > absolute
    return {'value': 'wrong' if wrong else 'agrees', 'reported': reported,
            'reference': reference, 'difference': delta,
            'relative_difference': delta / reference if reference else None,
            'known_issue': known if wrong else None}


def probe_statistics(rows):
    if not rows:
        raise EvidenceError('Probe logged no requests')
    if any(r.get('error') or not r.get('done') for r in rows):
        raise ProbeFailure('Probe requests failed or the probe did not drain')
    gaps = [g for r in rows for g in r['gaps_s']]
    if not gaps or any(r['usage_tokens'] is None or len(r['gaps_s']) != 49 for r in rows):
        raise EvidenceError('Probe missing usage or expected content chunks')
    return {'ttft_p50_s': percentile([r['ttft_s'] for r in rows], 50),
            'ttft_p99_s': percentile([r['ttft_s'] for r in rows], 99),
            'e2e_p50_s': percentile([r['e2e_s'] for r in rows], 50),
            'gap_p50_s': percentile(gaps, 50),
            'tokens': [r['usage_tokens'] for r in rows], 'requests': len(rows)}


def gate_p(rows):
    """Gate P: the median range and the tail ceiling must both hold."""
    stats = probe_statistics(rows)
    return (.200 <= stats['ttft_p50_s'] <= .205 and stats['ttft_p99_s'] <= .210
            and .020 <= stats['gap_p50_s'] <= .021
            and all(t == 50 for t in stats['tokens']))


def gate_l(before, after, probe):
    stats = probe_statistics(probe)
    count = after['requests'] - before['requests'] - stats['requests']
    tokens = after['tokens'] - before['tokens'] - sum(stats['tokens'])
    if count < 0 or tokens < 0:
        raise EvidenceError('Counters reset or probe accounting does not reconcile')
    return {'pass': abs(tokens - 50 * count) <= 50, 'requests': count,
            'tokens': tokens, 'residual_tokens': tokens - 50 * count}


def gate_s(probe):
    stats = probe_statistics(probe)
    return stats['ttft_p99_s'] <= .210 and stats['gap_p50_s'] <= .021


# Counter brackets use two offsets measured in the dry run. Model: the server counts probe request k (k-th by logged end time
# e_k) at c_k, and a_k = e_k - c_k is how far the counter leads the probe's log.
# If every a_k lies in [U_OFFSET, L_OFFSET], then for any read instant r
#     P(r + U_OFFSET) <= probe requests inside C(r) <= P(r + L_OFFSET),
# so U = C - P(start + U_OFFSET) bounds the tool count at the scrape's start from
# above and L = C - P(end + L_OFFSET) bounds it at the scrape's end from below
# (the scrape start is the read instant used for every lower time bound).
# Equivalently U(t) = C(t) - P(t + lead), L(t) = C(t) - P(t + lag).
MAX_SAMPLE_GAP_NS = 1_000_000_000   # one-second sampling bound
CALIBRATION_MARGIN_NS = 10_000_000  # added on each side of the measured range

# The probe sends one request every 100 ms on a fixed
# grid; counter reads are placed on the same grid, after the observed completion
# phase; the per-process CPU scan runs in its own 0.5 s loop.
PROBE_PERIOD_NS = 100_000_000        # 10 requests per second
COUNTER_READ_PERIOD_NS = PROBE_PERIOD_NS
CPU_SCAN_PERIOD_S = .5               # its own loop, every 0.5 s
REQUEST_NS = 1_180_000_000           # 200 ms + 49 x 20 ms
READ_GUARD_NS = 50_000_000           # "+ 50 ms" after the observed completion phase
DEFAULT_DELAY_NS = 30_000_000        # d before any completion
DELAY_WINDOW_NS = 2_000_000_000      # d: completions logged in the 2 s before the due instant
CALIBRATION_S = 25                   # 25 s with the probe at 10 Hz
CALIBRATION_BOUND_NS = 60_000_000    # lead range including both margins
CALIBRATION_ATTEMPTS = 3             # a failed bound is repeated at most twice


class ProbeFailure(EvidenceError):
    """A probe request failed or timed out: gate S failure for this attempt."""


def probe_ends(probe):
    """Sorted logged end times of the probe's requests that the server answered."""
    bad = [r for r in probe if r.get('error') or not r.get('done')]
    if bad:
        raise ProbeFailure(f'{len(bad)} probe request(s) failed or timed out: '
                           f"{bad[0].get('error') or 'not done'}")
    return sorted(int(r['end_ns']) for r in probe)


def calibrate(scrapes, probe, margin_ns=CALIBRATION_MARGIN_NS, bound_ns=CALIBRATION_BOUND_NS):
    """Measure the counter-vs-probe offsets from probe-only traffic.

    Every read carries `phase`: 'pre-start' (before the probe process starts),
    'back-to-back' (reads taken one after another while the probe runs) or
    'post-drain' (after the probe has drained). Order statistics: the k-th
    counter increment lies in (start of the last read below k, end of the first
    read at or above k]. Only back-to-back reads are bracket edges, and only
    the probe requests whose increment k falls between the first and the last of
    those reads (first count < k <= last count) enter the offsets. The pre-start
    read supplies only the counter baseline and the post-drain read only the
    total check (server count = probe requests); neither bounds an increment.
    """
    ends = probe_ends(probe)
    rows = [s for s in scrapes if not s.get('error') and 'counters' in s]
    phases = [s.get('phase') for s in rows]
    if any(p not in ('pre-start', 'back-to-back', 'post-drain') for p in phases):
        raise EvidenceError('Calibration reads must be tagged pre-start, back-to-back or post-drain')
    pre = [s for s in rows if s['phase'] == 'pre-start']
    b2b = [s for s in rows if s['phase'] == 'back-to-back']
    post = [s for s in rows if s['phase'] == 'post-drain']
    if not pre or not post or len(b2b) < 2 or not ends:
        raise EvidenceError('Calibration needs a pre-start read, back-to-back reads, '
                            'a post-drain read and probe completions')
    order = [s['start_ns'] for s in rows]
    if order != sorted(order) or phases != sorted(phases, key=['pre-start', 'back-to-back', 'post-drain'].index):
        raise EvidenceError('Calibration reads are not in clock and phase order')
    def count(row):
        # telemetry.counters parses Prometheus values as floats (e.g. 0.0); the
        # increment index must be an exact integer (as in CompletionTimeline.observe).
        value = row['counters']['requests']
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise EvidenceError('Invalid calibration completion counter')
        return int(value)
    base = count(pre[-1])
    counts = [count(r) - base for r in rows]
    if any(b < a for a, b in zip(counts, counts[1:])):
        raise EvidenceError('Calibration counter decreased')
    total = count(post[-1]) - base
    if total != len(ends):
        raise EvidenceError(f'Calibration: server counted {total} probe requests after the drain, '
                            f'probe logged {len(ends)}')
    b2b_counts = [count(r) - base for r in b2b]
    first_k, last_k = b2b_counts[0] + 1, b2b_counts[-1]
    if last_k < first_k:
        raise EvidenceError('No probe request completed between the first and last back-to-back reads')
    low = high = None
    widths = []
    j = 0
    for k in range(first_k, last_k + 1):
        end = ends[k - 1]
        while b2b_counts[j] < k:
            j += 1
        first, previous = b2b[j], b2b[j - 1]
        a_min, a_max = end - first['end_ns'], end - previous['start_ns']
        widths.append(a_max - a_min)
        low = a_min if low is None else min(low, a_min)
        high = a_max if high is None else max(high, a_max)
    widths.sort()
    u_offset, l_offset = int(low - margin_ns), int(high + margin_ns)
    return {'u_offset_ns': u_offset, 'l_offset_ns': l_offset,
            'measured_lead_min_ns': int(low), 'measured_lead_max_ns': int(high),
            'margin_ns': margin_ns, 'probe_requests': len(ends),
            'probe_requests_used': last_k - first_k + 1, 'used_increments': [first_k, last_k],
            'counter_reads': len(rows), 'back_to_back_reads': len(b2b),
            'pre_start_reads': len(pre), 'post_drain_reads': len(post),
            'median_read_window_ns': widths[len(widths) // 2],
            'max_read_window_ns': widths[-1],
            'lead_range_with_margins_ns': l_offset - u_offset, 'bound_ns': bound_ns,
            'bound_pass': l_offset - u_offset <= bound_ns,
            'meaning': 'counter leads the probe log by a in [lead_min, lead_max]; '
                       'U uses u_offset (= lead_min - margin), L uses l_offset (= lead_max + margin)',
            'rule': 'back-to-back reads while the probe runs; bounded lead range'}


class ReadPlacement:
    """Counter-read placement on the probe's 100 ms grid.

    Read k is due at origin + k x 100 ms + 1.18 s + d + 50 ms, where d is the
    median of (logged completion - planned send - 1.18 s) over the probe
    requests whose logged completion lies in [due - 2 s, due); with no such
    completion d = 30 ms. The due instant depends on d and d on the due instant,
    so `due` iterates to the fixed point (a median over ~20 completions moves by
    at most one element per iteration) and reports whether it converged.
    """
    def __init__(self, origin_ns, period_ns=PROBE_PERIOD_NS):
        self.origin_ns, self.period_ns = int(origin_ns), int(period_ns)
        self.ends, self.delays = [], []      # sorted by logged end

    def add(self, row):
        if row.get('done') and not row.get('error') and 'planned_ns' in row and 'end_ns' in row:
            end = int(row['end_ns'])
            i = bisect.bisect_right(self.ends, end)
            self.ends.insert(i, end)
            self.delays.insert(i, end - int(row['planned_ns']) - REQUEST_NS)

    def delay(self, due_ns):
        lo = bisect.bisect_left(self.ends, due_ns - DELAY_WINDOW_NS)
        hi = bisect.bisect_left(self.ends, due_ns)
        if hi <= lo:
            return DEFAULT_DELAY_NS, 0
        return int(median(self.delays[lo:hi])), hi - lo

    def base(self, k):
        return self.origin_ns + k * self.period_ns + REQUEST_NS + READ_GUARD_NS

    def due(self, k, iterations=6):
        base = self.base(k)
        d, n = self.delay(base + DEFAULT_DELAY_NS)
        for _ in range(iterations):
            d_next, n_next = self.delay(base + d)
            if d_next == d:
                return {'k': k, 'due_ns': base + d, 'd_ns': d, 'd_samples': n_next, 'converged': True}
            d, n = d_next, n_next
        return {'k': k, 'due_ns': base + d, 'd_ns': d, 'd_samples': n, 'converged': False}

    def next_due(self, now_ns, last_k=None):
        """The next grid instant strictly after now: no catch-up after a late read."""
        k = math.ceil((now_ns - self.base(0)) / self.period_ns) - 1
        if last_k is not None:
            k = max(k, last_k + 1)
        planned = self.due(k)
        while planned['due_ns'] <= now_ns:
            k += 1
            planned = self.due(k)
        return planned


def read_placement_report(reads, probe, calibration):
    """Offline, after the run: per in-run read, the distance to the nearest logged
    probe completion and whether a probe completion fell in its uncertain window
    (start + u_offset, end + l_offset]; d is also recomputed from the full log
    against the recorded due instant (`d_offline_ns`)."""
    ends = probe_ends(probe)
    placement = None
    rows = []
    for read in reads:
        if 'start_ns' not in read:
            continue
        origin = read.get('grid_origin_ns')
        if origin is not None and (placement is None or placement.origin_ns != origin):
            placement = ReadPlacement(origin)
            for row in probe:
                placement.add(row)
        start, end = int(read['start_ns']), int(read['end_ns'])
        i = bisect.bisect_left(ends, start)
        nearest = None
        for j in (i - 1, i):
            if 0 <= j < len(ends):
                e = ends[j]
                distance = 0 if start <= e <= end else min(abs(e - start), abs(e - end))
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, e - start)
        inside = (bisect.bisect_right(ends, end + calibration['l_offset_ns'])
                  - bisect.bisect_right(ends, start + calibration['u_offset_ns']))
        # Where the ambiguous completions lie: before the read starts (U side: the
        # counter may already hold them, U overcounts), during the read, or after it
        # ends (L side: the counter may not hold them yet, L undercounts).
        u_side = max(0, bisect.bisect_left(ends, start) - bisect.bisect_right(ends, start + calibration['u_offset_ns']))
        l_side = max(0, bisect.bisect_right(ends, end + calibration['l_offset_ns']) - bisect.bisect_right(ends, end))
        during = bisect.bisect_right(ends, end) - bisect.bisect_left(ends, start)
        due = read.get('due_ns')
        rows.append({'phase': read.get('phase'), 'k': read.get('k'), 'due_ns': due,
                     'start_ns': start, 'end_ns': end, 'error': read.get('error'),
                     'start_late_ns': None if due is None else start - due,
                     'd_ns': read.get('d_ns'),
                     'd_offline_ns': (placement.delay(due)[0]
                                      if placement is not None and due is not None and read.get('phase') == 'grid'
                                      else None),
                     'nearest_probe_completion_distance_ns': None if nearest is None else nearest[0],
                     'nearest_probe_completion_minus_start_ns': None if nearest is None else nearest[1],
                     'probe_completions_in_uncertain_window': inside,
                     'u_side_completions': u_side, 'during_read_completions': during,
                     'l_side_completions': l_side})
    grid = [r for r in rows if r['phase'] == 'grid' and not r['error']]
    late = [r['start_late_ns'] for r in grid if r['start_late_ns'] is not None]
    longest = run = 0
    for r in grid:
        run = run + 1 if r['probe_completions_in_uncertain_window'] > 0 else 0
        longest = max(longest, run)

    def fraction(key):
        return sum(r[key] > 0 for r in grid) / len(grid) if grid else None
    return {'reads': rows, 'distance_computed': 'offline, after the run, from the complete probe log',
            'summary': {'reads': len(rows), 'grid_reads': len(grid),
                        'pre_origin_reads': sum(r['phase'] == 'pre-origin' for r in rows),
                        'errored_reads': sum(bool(r['error']) for r in rows),
                        'ambiguous_reads': sum(r['probe_completions_in_uncertain_window'] > 0 for r in rows),
                        'skipped_grid_instants': sum(int(r.get('skipped', 0) or 0) for r in reads),
                        'max_start_late_ns': max(late, default=None),
                        'd_online_offline_mismatches': sum(r['d_ns'] != r['d_offline_ns'] for r in grid),
                        # Reported for reading before a panel; not a gate (fractions over successful grid reads).
                        'grid_ambiguous_fraction': fraction('probe_completions_in_uncertain_window'),
                        'grid_u_side_fraction': fraction('u_side_completions'),
                        'grid_during_read_fraction': fraction('during_read_completions'),
                        'grid_l_side_fraction': fraction('l_side_completions'),
                        'grid_longest_ambiguous_run_reads': longest}}


class CompletionTimeline:
    """Brackets of the tool's completed requests from sampled server counters.

    Each row holds U (upper bound of the tool count at the scrape start) and L
    (lower bound at the scrape end). First and last completions:
      f_lo = latest scrape start with U < 1   (no tool completion yet then)
      f_hi = earliest scrape end with L >= 1
      l_lo = latest scrape start with U < T
      l_hi = earliest scrape end with L >= T
    A probe completion that lands inside one scrape's window can lift a single
    U by one; taking the latest U < 1 (not the first U >= 1) keeps such a row
    from moving f_lo back to the probe's start. Bounds use actual sample times,
    so sampling jitter widens the bracket instead of invalidating it.
    """
    def __init__(self, baseline_requests, calibration):
        if calibration is None or calibration['u_offset_ns'] > calibration['l_offset_ns']:
            raise EvidenceError('Counter-vs-probe calibration is required')
        self.baseline_requests = baseline_requests
        self.u_offset = int(calibration['u_offset_ns'])
        self.l_offset = int(calibration['l_offset_ns'])
        self.rows = []
        self.f_lo_ns = None             # latest start with U < 1 so far
        self.possible_completion = False  # some row had U >= 1
        self.certain_completion = False   # some row had L >= 1 (online: computed separately)
        self.first_certain_end_ns = None
        self.dropped_samples = 0

    def observe(self, sample, ends, *, online=False):
        """Add one sample; `ends` is the sorted list of probe end times.

        Online, the row keeps U only (the span is rebuilt offline from the
        complete probe log), and the caller must pass every probe end at or
        before end + l_offset; L is still evaluated to record whether a certain
        tool completion (L >= 1) has been seen, which the 600 s no-completion cap
        needs. Errored samples are counted, not used.
        """
        if sample.get('error') or 'counters' not in sample:
            self.dropped_samples += 1
            return None
        start, end = int(sample['start_ns']), int(sample['end_ns'])
        count = sample['counters']['requests'] - self.baseline_requests
        if not math.isfinite(count) or count < 0 or count != int(count):
            raise EvidenceError('Invalid or reset completion counter')
        if end < start:
            raise EvidenceError('Scrape ends before it starts')
        if self.rows:
            previous = self.rows[-1]
            if start <= previous['start_ns']:
                raise EvidenceError('Completion samples are not in clock order')
            if start - previous['start_ns'] > MAX_SAMPLE_GAP_NS:
                raise EvidenceError('Completion sampling gap exceeds the one-second bound')
            if count < previous['server_completed']:
                raise EvidenceError('Server completion counter decreased')
        elif count != 0:
            raise EvidenceError('Completion timeline requires an initial zero-tool sample')
        upper = int(count) - bisect.bisect_right(ends, start + self.u_offset)
        certain = int(count) - bisect.bisect_right(ends, end + self.l_offset)
        lower = None if online else certain
        if certain >= 1 and not self.certain_completion:
            self.certain_completion, self.first_certain_end_ns = True, end
        row = {'start_ns': start, 'end_ns': end, 'server_completed': int(count),
               'tool_completed_upper': upper, 'tool_completed_lower': lower}
        if upper < 1:
            self.f_lo_ns = start
        else:
            self.possible_completion = True
        self.rows.append(row)
        return row

    def check(self, tool_launch_ns, total):
        """Consistency of the brackets; a violation means the offsets are wrong."""
        problems = []
        best_lower = None
        lowers = sorted((r['end_ns'], r['tool_completed_lower']) for r in self.rows)
        index = 0
        for row in self.rows:
            if row['tool_completed_upper'] < 0:
                problems.append(f"U<0 at {row['start_ns']}")
            if row['tool_completed_lower'] > total:
                problems.append(f"L>T at {row['end_ns']}")
            if row['end_ns'] <= tool_launch_ns and not (
                    row['tool_completed_lower'] <= 0 <= row['tool_completed_upper']):
                problems.append(f"nonzero bracket before tool launch at {row['start_ns']}")
            while index < len(lowers) and lowers[index][0] <= row['start_ns']:
                value = lowers[index][1]
                best_lower = value if best_lower is None else max(best_lower, value)
                index += 1
            if best_lower is not None and best_lower > row['tool_completed_upper']:
                problems.append(f"earlier L exceeds later U at {row['start_ns']}")
        if problems:
            raise EvidenceError('Counter-vs-probe offsets violated: ' + '; '.join(problems[:5]))
        return {'checked_rows': len(self.rows), 'pre_launch_rows':
                sum(r['end_ns'] <= tool_launch_ns for r in self.rows)}

    def span(self, total):
        if total < 0 or not math.isfinite(total) or total != int(total):
            raise EvidenceError('Invalid drained tool completion total')
        if not self.rows or any(r['tool_completed_lower'] is None for r in self.rows):
            raise EvidenceError('Full drained probe log required for the span')
        width = max(r['tool_completed_upper'] - r['tool_completed_lower'] for r in self.rows)
        result = {'total_tool_completions': int(total), 'u_offset_ns': self.u_offset,
                  'l_offset_ns': self.l_offset, 'max_bracket_width_requests': width,
                  'dropped_samples': self.dropped_samples, 'threshold_s': 63.16,
                  'f_lo_ns': None, 'f_hi_ns': None, 'l_lo_ns': None, 'l_hi_ns': None,
                  'lower_s': None, 'upper_s': None}
        if total == 0:
            return result

        def last_start_below(threshold):
            found = [r['start_ns'] for r in self.rows if r['tool_completed_upper'] < threshold]
            if not found:
                raise EvidenceError('No sample shows fewer than the bracketed completions')
            return max(found)

        def first_end(bound, threshold):
            found = next((r['end_ns'] for r in self.rows if r[bound] >= threshold), None)
            if found is None:
                raise EvidenceError('Drained samples do not bracket the tool completions')
            return found
        f_hi = first_end('tool_completed_lower', 1)
        f_lo = last_start_below(1)
        l_lo = last_start_below(total)
        l_hi = first_end('tool_completed_lower', total)
        result.update(f_lo_ns=f_lo, f_hi_ns=f_hi, l_lo_ns=l_lo, l_hi_ns=l_hi,
                      lower_s=(l_lo - f_hi) / 1e9, upper_s=(l_hi - f_lo) / 1e9)
        return result


def build_timeline(baseline, samples, probe, calibration):
    """Offline timeline over every sample, sorted by scrape start (errors skipped)."""
    ends = probe_ends(probe)
    timeline = CompletionTimeline(baseline, calibration)
    usable = [s for s in samples if not s.get('error') and 'counters' in s]
    for sample in sorted(usable, key=lambda s: s['start_ns']):
        timeline.observe(sample, ends)
    timeline.dropped_samples = len(samples) - len(usable)
    return timeline


def delivery(tool, rate, count, span=None, did_not_complete=False):
    if did_not_complete:
        return False
    if count == 0:
        return False
    if tool in ('vllm', 'inference-perf'):
        if span is None:
            raise EvidenceError('Sampled completion-span bounds are required')
        lower, upper = span['lower_s'], span['upper_s']
        if not all(math.isfinite(x) for x in (lower, upper)) or lower > upper:
            raise EvidenceError('Invalid completion-span bounds')
        if upper <= 63.16:
            return True
        if lower > 63.16:
            return False
        return 'unstable'
    return count >= .95 * rate * (60 - 1.18)


GUIDELLM_DOC = ('GuideLLM v0.8.0 docs/en/guides/metrics.md:191 (tag v0.8.0, commit 4545d7c): '
                'the reported percentile is the value at rank ceil(q.n) in the sorted sample')
W_NOTE = ('GuideLLM percentile cells are classified by its documented nearest-rank definition; '
          'W is also given under the plain numpy-percentile rule (W_numpy_reference).')


def guidellm_percentile(reported, exported, summary_count, q):
    """Classification of one GuideLLM percentile cell, in GuideLLM's own units.

    n = GuideLLM's summary count for that distribution; the rank ceil(q.n) is
    computed exactly (fractions) and read from the sorted exported values.
    Equal -> 'differs by documented definition'. When export and summary
    counts differ by m > 0, a value within m ranks of ceil(q.n) also reads
    'differs by documented definition' (count mismatch). Otherwise exactly one
    rank above -> 'wrong', known issue vllm-project/guidellm#1193. Anything
    else -> 'wrong'. Comparisons are exact equality to sorted elements.
    """
    xs = sorted(float(x) for x in exported)
    n = int(summary_count)
    if not xs or n <= 0 or not all(math.isfinite(x) for x in xs):
        raise EvidenceError('GuideLLM percentile needs exported values and a positive summary count')
    m = abs(len(xs) - n)
    rank = math.ceil(Fraction(q, 100) * n)

    def at(r):
        return xs[r - 1] if 1 <= r <= len(xs) else None
    matches = [r for r in range(max(1, rank - m - 1), min(len(xs), rank + m + 1) + 1) if at(r) == reported]
    result = {'rank': rank, 'summary_count': n, 'export_count': len(xs), 'count_mismatch': m,
              'value_at_rank': at(rank), 'value_one_rank_above': at(rank + 1),
              'matching_ranks_near': matches, 'definition': GUIDELLM_DOC, 'known_issue': None}
    if at(rank) == reported:
        result['value'] = 'differs by documented definition'
    elif m > 0 and any(abs(r - rank) <= m for r in matches):
        result['value'] = 'differs by documented definition'
        result['note'] = f'count mismatch of {m}; see b.guidellm.count entry'
    elif m == 0 and at(rank + 1) == reported:
        result['value'] = 'wrong'
        result['label'] = 'wrong (known)'
        result['known_issue'] = 'vllm-project/guidellm#1193'
    else:
        result['value'] = 'wrong'
    return result


def aiperf_1492(result, normalized, count, relative=1e-4):
    """AIPerf #1492 divides by the phase window; tag only when the reported rate
    equals count / phase window and is below the documented reference."""
    window = normalized.get('phase_window_s')
    if result['value'] != 'wrong' or not window or window <= 0:
        return None
    by_window = count / window
    if (abs(result['reported'] - by_window) <= relative * by_window
            and result['reported'] < result['reference']):
        return 'ai-dynamo/aiperf#1492'
    return None


def comparisons(tool, normalized, count, probe, disconnect_counted):
    n = normalized
    stats = probe_statistics(probe)
    rows = n['requests']
    cells = {}
    reported_count = n['completed']
    if tool == 'guidellm':
        if disconnect_counted is None:
            raise EvidenceError('Disconnect dry-run receipt required')
        if disconnect_counted:
            reported_count += n['incomplete']
    cells['b.completed'] = cell(reported_count, count, .01)
    if tool == 'guidellm':
        cells['b.rate'] = cell(n['rate'], None, .01)
    else:
        duration = n['duration_s']
        if tool == 'inference-perf':
            starts = n['all_start_times']
            duration = max(starts) - min(starts)
            rate_count = len(starts)  # pinned code uses successes + failures
        else:
            rate_count = n['completed']
        if duration is None or duration <= 0:
            raise EvidenceError('Missing/nonpositive rate denominator')
        cells['b.rate'] = cell(n['rate'], rate_count / duration, .01)
        if tool == 'aiperf':
            cells['b.rate']['known_issue'] = aiperf_1492(cells['b.rate'], n, rate_count)
    for metric in ('ttft', 'e2e'):
        values = [r[f'{metric}_s'] for r in rows]
        for q in (50, 99):
            reported = n[f'{metric}_p{q}_s']
            key = f'b.{metric}.p{q}'
            cells[key] = cell(reported, percentile(values, q), 1e-6)
            if reported is not None:
                cells[key]['offset_samples'] = sample_offset(values, reported, q)
                if tool == 'guidellm':
                    # Documented nearest-rank classification. The numpy difference stays in
                    # 'difference'; the numpy-rule value is kept so W can be given both ways.
                    native = n['percentile_native'][metric]
                    rank_cell = guidellm_percentile(native['reported'][q], native['values'],
                                             native['summary_count'], q)
                    cells[key]['value_numpy_reference'] = cells[key]['value']
                    cells[key]['known_issue'] = rank_cell.pop('known_issue')
                    cells[key]['value'] = rank_cell.pop('value')
                    cells[key]['documented_rank'] = rank_cell
    if tool == 'guidellm':
        # The export-vs-summary count is its own consistency entry (recorded, no verdict).
        counts = n['counts']
        cells['b.guidellm.count'] = {
            'value': 'recorded', 'export_successful': counts['export_successful'],
            'summary_successful': counts['summary_successful'],
            'summary_count_ttft': n['percentile_native']['ttft']['summary_count'],
            'summary_count_e2e': n['percentile_native']['e2e']['summary_count'],
            'export_incomplete': counts.get('export_incomplete'),
            'summary_incomplete': counts.get('summary_incomplete'),
            'difference': counts['export_successful'] - counts['summary_successful']}
    for metric, absolute in (('ttft', .005), ('e2e', .020)):
        cells[f'c.{metric}.p50'] = cell(n[f'{metric}_p50_s'], stats[f'{metric}_p50_s'], .10, absolute)
    cells['c.gap'] = cell(n['gap_p50_s'], None if tool == 'inference-perf' else stats['gap_p50_s'], .10, .002)
    tokens = n['output_tokens']
    if not tokens:
        cells['c.tokens'] = cell(None, 50, 0)
    else:
        differences = [x - 50 for x in tokens]
        cells['c.tokens'] = {'value': 'wrong' if any(differences) else 'agrees',
                             'reported': tokens, 'reference': 50, 'difference': differences,
                             'wrong_requests': sum(x != 50 for x in tokens)}
    return cells


def aggregate(records, p_passed, dropped_from=None):
    """The decision rule applied mechanically to the accepted attempts."""
    valid_rates = [r for r in RATES if dropped_from is None or r < dropped_from]
    candidate = {'valid_rates': valid_rates,
                 'decision': None, 'knees': {}, 'delivery': {}, 'cells': {}, 'W': None}
    if not p_passed or len(valid_rates) < 3:
        candidate['decision'] = 'NOT-MEASURABLE'
        return candidate
    wrong_count = 0
    wrong_numpy = 0
    clean = True
    candidate['busiest_thread'] = {}
    candidate['guidellm_count_entries'] = {}
    candidate['W_note'] = W_NOTE
    known_cells = ['b.completed', 'b.rate', 'b.ttft.p50', 'b.ttft.p99', 'b.e2e.p50',
                   'b.e2e.p99', 'c.ttft.p50', 'c.e2e.p50', 'c.gap', 'c.tokens']
    for tool in TOOLS:
        knee = None
        continuous = True
        for rate in valid_rates:
            runs = [r for r in records if r['tool'] == tool and r['rate'] == rate]
            round_ids = [r['round'] for r in runs]
            if len(round_ids) != len(set(round_ids)):
                raise EvidenceError('Multiple accepted attempts for the same round')
            complete = set(round_ids) == {1, 2, 3} and all(r.get('valid') for r in runs)
            deliveries = [r.get('delivered') for r in runs]
            delivery_value = ('not measured' if not complete or None in deliveries
                              else deliveries[0] if len(set(deliveries)) == 1 else 'unstable')
            candidate['delivery'][f'{tool}/{rate}'] = delivery_value
            delivered = complete and all(r.get('delivered') is True for r in runs)
            continuous &= delivered
            if continuous:
                knee = rate
            if not complete:
                clean = False
            # The busiest-thread value beside the attribution, every tool and rate.
            candidate['busiest_thread'][f'{tool}/{rate}'] = [
                {'round': r['round'], 'busiest_thread_core_fraction': r.get('busiest_tool_thread_send_window'),
                 'send_window_ns': r.get('send_window_ns'), 'delivered': r.get('delivered'),
                 'shortfall_attribution': r.get('shortfall_attribution'),
                 'did_not_complete': r.get('did_not_complete'), 'exit_code': r.get('exit_code'),
                 'deadline_basis': r.get('deadline_basis')}
                for r in sorted(runs, key=lambda r: r['round'])]
            if tool == 'guidellm':
                candidate['guidellm_count_entries'][str(rate)] = [
                    dict(r.get('cells', {}).get('b.guidellm.count', {}), round=r['round'])
                    for r in sorted(runs, key=lambda r: r['round'])]
            for name in known_cells:
                values = [r.get('cells', {}).get(name, {}).get('value', 'no entry') for r in runs]
                numpy_values = [r.get('cells', {}).get(name, {}).get('value_numpy_reference',
                          r.get('cells', {}).get(name, {}).get('value', 'no entry')) for r in runs]
                value = 'not measured' if not complete else values[0] if len(set(values)) == 1 else 'unstable'
                value_numpy = ('not measured' if not complete else numpy_values[0] if len(set(numpy_values)) == 1
                                else 'unstable')
                candidate['cells'][f'{tool}/{rate}/{name}'] = value
                if value_numpy != value:
                    candidate.setdefault('cells_numpy_reference', {})[f'{tool}/{rate}/{name}'] = value_numpy
                if rate == 10 and value_numpy == 'wrong':
                    wrong_numpy += 1
                exempt = (tool == 'guidellm' and name == 'b.rate') or (tool == 'inference-perf' and name == 'c.gap')
                if not exempt and value in ('no entry', 'not measured', 'unstable'):
                    clean = False
                if rate == 10 and value == 'wrong':
                    wrong_count += 1
                if continuous and value == 'wrong':
                    clean = False
        candidate['knees'][tool] = knee
        clean &= knee == max(valid_rates)
    candidate['W'] = wrong_count                 # documented definitions (governs the decision row)
    candidate['W_numpy_reference'] = wrong_numpy  # plain numpy-percentile rule
    candidate['decision'] = 'CLEAN' if clean and wrong_count == 0 else 'FINDINGS'
    return candidate
