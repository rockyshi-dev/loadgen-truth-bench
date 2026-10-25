# loadgen-truth-bench

Checks four LLM load generators against a server whose timing is known in advance: up to what
request rate does each tool deliver the load it was asked for, and does each number it reports
agree with what happened, under the tool's own documented definition?

Tools, at pinned releases:

| Tool | Release | Commit |
|---|---|---|
| `vllm bench serve` | vLLM v0.30.0 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| GuideLLM | v0.8.0 | `4545d7cdbab8753d780fa90083259e5a840b6782` |
| AIPerf | v0.13.0 | `794f8bb75f8582f22e412d7e650fc71ca2a3d21a` |
| inference-perf | v0.7.0 | `5804ea6b7ebd2bfceff29cf113311ed3af95146e` |
| server: llm-d-inference-sim | v0.11.2 | `7639395743c76f1cb37222981b33e8ea56cce69b` |

Everything runs on CPUs. No GPU and no model weights are used; only a tokenizer.

## What it measures

**The server.** llm-d-inference-sim, built from its pinned source, with time to first token
200 ms, 20 ms between tokens, both standard deviations 0, load factor 1.0 and `max-num-seqs`
8192, so it never queues. It sends one token per chunk. Its delays are sleeps, so they are
floors. Every request carries `max_tokens: 50` and `ignore_eos: true`, which fixes the output at
50 tokens; a request then takes 200 + 49 x 20 = 1,180 ms.

**The requests.** All four tools use the completions API (`/v1/completions`), streaming with
usage reporting, a synthetic prompt of about 100 tokens from the local Qwen2.5-7B-Instruct
tokenizer, 50 output tokens. The exact command lines are built in `ltbench/spec.py` and stored
in the manifest.

**The one variable.** Target rate R in {10, 50, 200, 500, 1000} requests per second, each tool's
constant-rate mode, a nominal 60 s:

- vLLM: `--request-rate R --burstiness inf`, 60 R prompts; it runs until all are sent; one process.
- GuideLLM: constant profile at R for 60 s (it cancels requests in flight at 60 s);
  `GUIDELLM__MAX_WORKER_PROCESSES=24`.
- AIPerf: `--request-rate R --arrival-pattern constant`, 60 s, grace period 5 s,
  `--use-server-token-count`, `--workers-max 24`.
- inference-perf: `constant` load at R for 60 s, waits for all requests, `num_workers: 24`,
  `use_server_output_tokens: true`.

**The probe.** A minimal client of this repository, in its own process, sends one request every
100 ms on a fixed grid during every run, in the same form as the tools, and logs per request the
TTFT, every gap between chunks, the end-to-end time and the usage token count. It is never busy,
so it shows what the server did during the run.

**The server's counters.** `vllm:request_success_total` and `vllm:generation_tokens_total` are
read before the probe starts, ten times per second during the run, and after the probe has
drained. The probe's own requests are subtracted, so what remains is the tool's.

**Placement on the machine.** The simulator on 16 physical cores of socket 1, the probe and the
counter reader on 4 other cores of socket 1, the tool on 26 cores of socket 0 (both SMT siblings of
a core always go to the same role). Loopback, port 8188. Three rounds; in each round the rates
run in ascending order and, per rate, the four tools in a fresh random order (seeded); the server
is restarted before every tool run.

### Three kinds of check

**(a) Delivery: did the tool send what it was asked to?**

- vLLM and inference-perf send a fixed number of requests and wait for all of them, so the count
  says nothing. "Delivered" means the time from the tool's first to its last completion is at most
  60 / 0.95 = 63.16 s. The server's counter is sampled, so both ends are bracketed: from each read
  (start s, end e), U = C - P(s + u) is an upper bound on the tool's completions and
  L = C - P(e + l) a lower bound, where C is the server's count and P(x) the number of probe
  requests logged as finished by x. The offsets u and l are measured in the dry run (how far the
  counter leads the probe's log, plus 10 ms on each side). The span lies between
  l_lo - f_hi and l_hi - f_lo (f = first, l = last completion; lo = latest read start that still
  excludes it, hi = earliest read end that certainly includes it). Upper bound <= 63.16 s:
  delivered; lower bound > 63.16 s: not delivered; otherwise unstable. Counter reads are placed on
  the probe's grid, after the observed completion phase of its requests, so that few reads are
  ambiguous.
- GuideLLM and AIPerf stop at 60 s. "Delivered" means at least 0.95 x R x (60 - 1.18) completed
  requests.
- A run not finished 180 s after the tool's first possible completion is stopped: "did not
  complete", not delivered, no reported numbers. A tool that completes nothing is stopped 600 s
  after launch. A tool that exits with a non-zero status has not delivered.

**(b) Consistency: do the tool's summary numbers follow from its own per-request data?**

| Cell | Reference | Tolerance |
|---|---|---|
| completed requests | the server's count of the tool's completed requests (GuideLLM: successful plus incomplete if the simulator counts a cancelled stream; settled in the dry run) | 1% |
| achieved rate, vLLM | completed / its duration including the drain | 1% |
| achieved rate, inference-perf | all requests / span of its send times (its stage report's `achieved_rate`) | 1% |
| achieved rate, AIPerf | count / benchmark duration | 1% |
| achieved rate, GuideLLM | time-weighted, no single denominator: no reference ("no entry") | - |
| TTFT and end-to-end p50, p99 | linear-interpolation percentile (numpy's default) of the tool's own per-request values | relative 1e-6 |

GuideLLM documents its percentile as the value at rank ceil(q n) (`docs/en/guides/metrics.md:191`
at v0.8.0), where n is its own summary count. A GuideLLM cell equal to that rank's value reads
**differs by documented definition**; exactly one rank above reads **wrong (known)**, tagged
vllm-project/guidellm#1193; anything else reads **wrong**. Each such cell also keeps its value
under the plain numpy rule. An AIPerf rate that equals count / phase window is tagged
ai-dynamo/aiperf#1492.

**(c) Agreement with the server: do the reported latencies match what the server did?** The
truth is the probe during the same run.

| Quantity | Expected | "Wrong" when |
|---|---|---|
| TTFT p50 | the probe's TTFT p50 | off by more than 10% and more than 5 ms |
| end-to-end p50 | the probe's end-to-end p50 | off by more than 10% and more than 20 ms |
| gap between tokens | the probe's median chunk gap (inference-perf re-tokenizes chunks and can report zero gaps: no expected value, "no entry") | off by more than 10% and more than 2 ms |
| output tokens per request | 50 | any other value |

Each cell is **agrees**, **differs by documented definition**, **wrong** or **no entry**. Every
raw difference is kept, not only the verdict.

### Gates

| Gate | Pass condition | If it fails |
|---|---|---|
| P probe sane (dry run) | with an idle dummy tool, the probe's TTFT p50 200-205 ms and p99 <= 210 ms, median gap 20-21 ms, 50 tokens on every request | the dry run fails |
| L length | the tool's generated tokens (counter increase minus the probe's) within 50 of 50 x its completed requests | the panel stops |
| S server not saturated | the probe's TTFT p99 <= 210 ms and median gap <= 21 ms during the run; a failed probe request also fails S | rerun once; a second failure drops that rate and every higher rate for all tools |
| Q quiet | CPU time of processes not started by the run at most 2% on the server, probe and tool cores | rerun at most twice, then "not measured" |
| R repeat | a cell takes a value only if all three rounds give it, otherwise "unstable" | - |

### Decision rule

- **Knee** of a tool: the highest valid rate up to which it is delivered in all three rounds, with
  every lower valid rate also delivered. A shortfall is attributed to the tool only if gate S passed
  and the tool's busiest thread used more than 0.9 of a core over some 1 s interval of the send
  window [f_hi, l_lo]; otherwise "unexplained". The busiest-thread value is published for every
  tool and rate.
- **W**: the number of (b) and (c) cells that are "wrong" at R = 10, where nothing is near capacity.

| Row | Condition |
|---|---|
| NOT-MEASURABLE | gate P fails, or fewer than three rates are valid under gate S |
| CLEAN | W = 0; no cell at or below a tool's knee is wrong; every tool's knee is the highest valid rate; no needed cell is unstable, not measured or missing (GuideLLM's rate and inference-perf's gap have no reference by definition and are exempt) |
| FINDINGS | anything else: publish the table of knees, wrong cells with their errors, and documented-definition differences |

## Requirements

- Linux on x86_64 with two sockets: at least 26 physical cores on socket 0 and 20 on socket 1
  (`lscpu -p` is read), `taskset`, and an `RLIMIT_NOFILE` hard limit of at least 65,536.
- An otherwise idle machine (gate Q) and a free loopback port (8188 by default).
- conda, git, and internet access during setup only (PyPI, download.pytorch.org, go.dev, GitHub, and the
  Go module proxy). Where go.dev's download path or proxy.golang.org is unreachable, `GO_DOWNLOAD_BASE`
  (default `https://go.dev/dl/`; the same files are at `https://dl.google.com/go/`) and Go's own `GOPROXY`
  select mirrors. The toolchain archive is checked against the size and sha256 go.dev publishes for it,
  pinned in `ltbench/spec.py` (`GO_TOOLCHAIN`), so setup makes no request to go.dev's release list. Where
  github.com is unreachable, `SIM_GIT_URL` names any git remote (a bare clone, for example) holding the
  simulator repository; the source is checked out at the pinned commit id, which identifies its contents.
- The Qwen2.5-7B-Instruct tokenizer in a Hugging Face cache snapshot directory,
  `<HF_HOME>/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<commit>` (revision
  `a09a35458c702b33eeacc393d103063234e8bc28` is the one the command lines were written against).
  AIPerf in offline mode only accepts a repository id and revision, so the tokenizer must be in
  this layout; the run checks that the id and revision resolve to the same directory.

## Setup

```bash
# Installs GuideLLM, AIPerf and inference-perf (and vLLM, unless VLLM_ENV names an existing
# environment with vllm 0.30.0) each in its own conda environment under the work directory,
# checks every installed version and 15 load-bearing source files against the pinned releases,
# and builds the simulator with the Go toolchain its go.mod names (checksum verified).
setsid bash setup.sh /path/to/work > setup.log 2>&1 < /dev/null &
```

Every package in each environment is constrained to the version in `constraints/<tool>.txt`
(the `pip freeze` of the environments the published results were measured in: for example aiohttp 3.14.3 and httpx 0.28.1
everywhere, transformers 5.13.0 in inference-perf's). The tools' own HTTP and tokenizer stacks
are part of what is measured, and so is the probe's: the runner and the probe run in the
inference-perf environment, with its aiohttp. An environment given with `VLLM_ENV` is used as it
is and only checked; its `pip freeze` is saved next to the others for comparison.

`/path/to/work` then holds `envs/<tool>/`, `bin/llm-d-inference-sim`, `setup/` (install reports,
`pip freeze` locks, conda explicit lists, `<tool>-verify.json`, `sim-build.json`) and later the
tools' caches. Set `LTBENCH_ROOT=/path/to/work` or pass `--root`.

## Running

Run from the repository root with the inference-perf environment's Python (it has `aiohttp` and
`transformers`; the probe runs under the same interpreter). A dry run or panel refuses to start
under any other interpreter, or if a tool's executable is missing from the environment that
`setup.sh` verified for it (`--vllm-env` must name the environment setup checked).

Every process the run starts (simulator, probe, tools) gets a minimal environment: that
environment activated (`PATH`, `CONDA_PREFIX`), `HOME`, `LANG`, `LC_*`, `TZ`, `TMPDIR`, the
offline Hugging Face settings, and the harness's own variables. That environment, and nothing else
of the launching shell (no credentials, no `SSH_*`), is what the attempt manifests record.

```bash
PY=/path/to/work/envs/inference-perf/bin/python
export LTBENCH_ROOT=/path/to/work

# 1. Manifest: tool pins, tokenizer and its hashes, probe prompt, CPU roles, schedule, every
#    command line, and the sha256 of this package (a run refuses if the code changed since).
$PY -m ltbench construct --output runs/manifest \
    --tokenizer $HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
#    options: --vllm-env (default <root>/envs/vllm), --hf-home (default: from --tokenizer), --seed (default 20261003)

# 2. Dry run (no result): counter-offset calibration (25 s, repeated up to twice if the lead
#    range exceeds 60 ms), whether a cancelled stream is counted, gate P with an idle dummy tool,
#    then each tool at 10 req/s with gate L, exit 0, the expected request count and the expected
#    number of worker processes connected to the server. Under 15 minutes.
setsid $PY -m ltbench dry-run --manifest runs/manifest/manifest.json --output runs/dry \
    > runs/dry.log 2>&1 < /dev/null &

# 3. Panel: 3 rounds x 5 rates x 4 tools. About 1.5 hours.
setsid $PY -m ltbench panel --manifest runs/manifest/manifest.json \
    --dry-receipt runs/dry/dry-run-receipt.json --output runs/panel \
    > runs/panel.log 2>&1 < /dev/null &

# 4. Tables
$PY -m ltbench report runs/panel --tsv runs/panel-cells.tsv
```

Each run takes a lock in the work directory, so two runs cannot overlap. On a signal, every
process the run started is stopped before it exits. Inherited `LD_PRELOAD`, `PYTORCH_*`, `OMP_*`,
`VLLM_*`, `NCCL_*` and `CUDA_*` other than `CUDA_VISIBLE_DEVICES` stop the run, because each would
change what is measured.

## Output

The dry run writes `calibration-attempts.json`, `calibration.json`, `disconnect.json`,
`p-phase/`, one directory per tool, `dry-run-checks.json` (including, for reading before a
panel, how many counter reads had a probe completion inside their uncertain window) and
`dry-run-receipt.json`.

A panel writes:

- `accepted-attempts.json`: every accepted attempt: gates L, S, Q with their numbers, probe
  statistics, the completion span and its bracket, the send window, CPU accounting (foreign CPU per
  role, the busiest tool process and thread), delivery, attribution, and every cell with its
  reported value, reference, difference and classification.
- `decision.json`: valid rates, delivery per tool and rate, knees, aggregated cells, W under both
  rules, busiest-thread values, GuideLLM export-vs-summary counts, and the decision row.
- `instrument-failures.json`, `affinity.json`, `clock.json`, `calibration.json`.
- `r<round>-<rate>-<tool>/attempt-<n>/`: per attempt the command, manifests and logs of server,
  probe and tool, `probe.jsonl`, `counter-reads.jsonl`, `cpu-scans.jsonl`, `before.json`,
  `after.json`, `completion-samples.json`, `completion-span.json`, `counter-read-placement.json`,
  `raw-gate-calculations.json`, and the tool's own exports under `tool/`. Retries get new
  attempt directories; nothing is overwritten.

## Reading the result

`python -m ltbench report <panel>` recomputes the decision from `accepted-attempts.json` and warns
(exit code 1) where it differs from the stored `decision.json`. It prints:

- the valid rates and the gate S failures that dropped a rate (with the probe's TTFT p99);
- the delivery table with each tool's knee; the completion-span bounds of vLLM and
  inference-perf and how tightly each end is bracketed; GuideLLM's and AIPerf's completed counts
  against the floor; every attempt not delivered with its count, exit code, deadline, span, probe
  TTFT p99, busiest thread and attribution;
- the busiest tool thread over the send window per tool and rate (mean and range over rounds);
- the reporting-cell counts per tool (agrees, differs by documented definition, wrong, unstable,
  no entry, not measured), W under the documented definitions and under the plain numpy rule, the
  range of GuideLLM's percentile differences from numpy, every cell other than "agrees", and every
  round-cell tagged with a known upstream issue.

`--tsv` writes one row per tool, rate, round and cell with the raw reported value, the reference,
the difference, the classification under both rules and GuideLLM's documented rank.
`python -m ltbench compare <panel-a> <panel-b>` lists every difference in decision, knees,
delivery, aggregated cells and per-round cell classifications between two panels.

`python scripts/guidellm_timing.py <panel>` reads GuideLLM's own per-request timings from its
exports and prints, per GuideLLM run, how late each request was sent against GuideLLM's plan, its
send rate between seconds 20 and 58, and how soon a request was sent once a worker took it.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests use synthetic records, mocked processes and a recorded calibration trace; they start no
server, send no network traffic and need no GPU.

## License

MIT
