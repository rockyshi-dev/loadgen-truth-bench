"""How late GuideLLM sent each request against its own plan, from the per-request timings in its JSON
export (benchmarks[0].requests.{successful,incomplete,errored}[].info.timings), for every GuideLLM run
of a panel.

Usage: python scripts/guidellm_timing.py <panel-dir> > guidellm_timing.json
"""
import glob, json, statistics, sys
panel = sys.argv[1]
out = {"runs": []}
for rate in (10, 50, 200, 500):
    for rnd in (1, 2, 3):
        f = sorted(glob.glob(f"{panel}/r{rnd}-{rate}-guidellm/attempt-*/tool/guidellm.json"))[-1]
        b = json.load(open(f))["benchmarks"][0]
        T = []
        for kind in ("successful", "incomplete", "errored"):
            for r in b["requests"].get(kind) or []:
                t = r["info"]["timings"]
                T.append((t["targeted_start"], t["queued"], t["dequeued"], t["request_start"], kind))
        T.sort()
        t0 = T[0][0]
        rel = [(a - t0, q - t0, d - t0, s - t0, k) for a, q, d, s, k in T]
        # queue-insert rate: requests whose queued time falls in [20, 58) s, per second
        qs = sorted(q for _, q, _, _, _ in rel)
        n_win = sum(1 for q in qs if 20 <= q < 58)
        late = [s - a for a, _, _, s, _ in rel]
        send_after_dequeue = sorted(s - d for _, _, d, s, _ in rel)
        def near(sec):
            v = [s - a for a, _, _, s, _ in rel if sec <= a < sec + 1]
            return statistics.median(v) if v else None
        sent = [s for _, _, _, s, _ in rel]
        run = {"rate": rate, "round": rnd, "requests_planned_in_export": len(rel),
               "kinds": {k: sum(1 for x in rel if x[4] == k) for k in ("successful", "incomplete", "errored")},
               "planned_last_s": rel[-1][0],
               "queued_per_s_20_58": n_win / 38.0,
               "sent_per_s_20_58": sum(1 for s in sent if 20 <= s < 58) / 38.0,
               "median_lateness_at_5s": near(5), "median_lateness_at_30s": near(30), "median_lateness_at_55s": near(55),
               "max_lateness_s": max(late), "min_lateness_s": min(late),
               "send_after_dequeue_p99_ms": send_after_dequeue[int(0.99 * (len(send_after_dequeue) - 1))] * 1e3}
        if rnd == 1 and rate in (200, 500):
            step = max(1, len(rel) // 600)
            run["trace"] = [[round(a, 3), round(q - a, 4), round(s - a, 4)] for a, q, _, s, _ in rel[::step]]
        out["runs"].append(run)
print(json.dumps(out))
