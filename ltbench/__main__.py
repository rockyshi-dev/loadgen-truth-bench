"""Command line: python -m ltbench {construct,dry-run,panel,report,compare} ... (see --help of each)."""
import argparse
import json
from pathlib import Path
import sys

from ltbench.spec import DEFAULT_PORT, DEFAULT_SEED, work_root


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m ltbench', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('construct', help='write the run manifest (no traffic, no server)')
    c.add_argument('--output', type=Path, required=True, help='new directory for manifest.json')
    c.add_argument('--tokenizer', type=Path, required=True,
                   help='HF cache snapshot directory <HF_HOME>/hub/models--<org>--<name>/snapshots/<commit>')
    c.add_argument('--root', help='work directory made by setup.sh (default $LTBENCH_ROOT)')
    c.add_argument('--vllm-env', type=Path, help='Python environment with vllm 0.30.0 (default <root>/envs/vllm)')
    c.add_argument('--hf-home', type=Path, help='HF_HOME given to the tools (default: derived from --tokenizer)')
    c.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'seed of the random tool order (default {DEFAULT_SEED})')
    for name, text in (('dry-run', 'calibration, gate P and one run of each tool at 10 req/s; writes the receipt a panel needs'),
                       ('panel', 'three rounds of every tool at every rate; writes accepted-attempts.json and decision.json')):
        r = sub.add_parser(name, help=text)
        r.add_argument('--manifest', type=Path, required=True)
        r.add_argument('--output', type=Path, required=True, help='new directory (must not exist)')
        r.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'loopback port of the simulator (default {DEFAULT_PORT})')
        if name == 'panel':
            r.add_argument('--dry-receipt', type=Path, required=True, help='dry-run-receipt.json of a passing dry run')
    r = sub.add_parser('report', help='result tables recomputed from a panel directory')
    r.add_argument('run', type=Path, help='panel output directory')
    r.add_argument('--tsv', type=Path, help='also write every cell of every round with its raw difference')
    r.add_argument('--json', type=Path, help='also write the numbers as JSON')
    m = sub.add_parser('compare', help='cell-by-cell differences between two panel directories')
    m.add_argument('a', type=Path)
    m.add_argument('b', type=Path)
    args = p.parse_args(argv)

    if args.cmd == 'construct':
        from ltbench.construct import construct
        root = work_root(args.root)
        construct(args.output, args.tokenizer, args.seed, root, args.vllm_env or root / 'envs' / 'vllm', args.hf_home)
        print(f'wrote {args.output / "manifest.json"}')
        return 0
    if args.cmd in ('dry-run', 'panel'):
        from ltbench.runner import run
        run(args.cmd, args.manifest, args.output, getattr(args, 'dry_receipt', None), args.port)
        return 0
    from ltbench import report
    try:
        if args.cmd == 'report':
            data = report.load_run(args.run)
            rep = report.report(data)
            print(report.text_report(rep), end='')
            if args.tsv:
                report.write_tsv(data, args.tsv)
            if args.json:
                args.json.write_text(json.dumps(rep, indent=1, default=str) + '\n')
            return 1 if rep['warnings'] else 0
        result = report.compare(report.load_run(args.a), report.load_run(args.b))
        for line in result['differences']:
            print(line)
        print(f"{result['round_cells_differing']} of {result['round_cells_compared']} round-cells differ; "
              f"{len(result['differences'])} differences in all")
        return 1 if result['differences'] else 0
    except report.ReportError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
