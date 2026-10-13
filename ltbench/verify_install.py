"""Installed version and load-bearing source files of one tool. Does not import the tool or touch a GPU.

Run inside the tool's own environment: python ltbench/verify_install.py <tool>.
The files listed in upstream_source_hashes.json (argv handling and metric code) must be
byte-identical to the pinned release; for vLLM a local build tag (e.g. +cu129) is accepted and
recorded, because only the client code is used and its files are checked here.
"""
import importlib.metadata
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltbench.spec import PINS, digest  # noqa: E402


def check(name):
    distribution = importlib.metadata.distribution(name)
    expected_version = PINS[name][1].removeprefix('v')
    public = distribution.version.split('+', 1)[0]
    if public != expected_version or (name != 'vllm' and distribution.version != expected_version):
        raise ValueError(f'{name}: wrong installed version {distribution.version}')
    all_hashes = json.loads((Path(__file__).parent / 'upstream_source_hashes.json').read_text())
    result = {'tool': name, 'version': distribution.version, 'pinned_commit': PINS[name][2],
              'files': {}, 'scope': 'Pinned paths used for argv and metric instrumentation'}
    for relative, expected in all_hashes[name].items():
        installed = Path(distribution.locate_file(relative))
        actual = digest(installed)
        if actual != expected:
            raise ValueError(f'Installed source does not match pinned commit: {installed}')
        result['files'][relative] = {'path': str(installed), 'sha256': actual}
    return result


def main():
    print(json.dumps(check(sys.argv[1]), indent=2))


if __name__ == '__main__':
    main()
