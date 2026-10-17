"""Fetch the pinned Go toolchain and build the pinned simulator inside the work directory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltbench.spec import PINS  # noqa: E402


def fetch(url, path):
    with urllib.request.urlopen(url, timeout=90) as response, path.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)


def main(work):
    work = Path(work)
    root = work / 'setup'
    (work / 'bin').mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    go_version = 'go1.26.6'  # pinned source go.mod, not the host's default toolchain
    release_file = root / 'go-release.json'
    if release_file.exists():
        release = json.loads(release_file.read_text())
    else:
        with urllib.request.urlopen('https://go.dev/dl/?mode=json&include=all', timeout=60) as response:
            releases = json.load(response)
        release = next(r for r in releases if r['version'] == go_version)
        release_file.write_text(json.dumps(release, indent=2))
    assert release['version'] == go_version
    artifact = next(f for f in release['files'] if f['filename'] == f'{go_version}.linux-amd64.tar.gz')
    archive = root / artifact['filename']
    if not (root / 'go/bin/go').exists():
        # The archive is checked against the sha256 go.dev publishes, so it may come from a mirror.
        fetch(os.environ.get('GO_DOWNLOAD_BASE', 'https://go.dev/dl/') + artifact['filename'], archive)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == artifact['sha256']
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter='data')
        archive.unlink()
    go_env = dict(os.environ, GOENV='off', GOTELEMETRY='off',
                  GOTELEMETRYDIR=str(root / 'go-telemetry'))
    version = subprocess.check_output([str(root / 'go/bin/go'), 'version'], text=True, env=go_env)
    assert go_version in version
    repo, tag, commit = PINS['sim']
    source = root / 'sim-source'
    if not source.exists():
        subprocess.run(['git', 'init', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'remote', 'add', 'origin', f'https://github.com/{repo}.git'], check=True)
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != commit:
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth=1', 'origin', commit], check=True)
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Refuse to build or overwrite a modified simulator source tree')
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', commit], check=True)
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == commit
    env = dict(go_env, CGO_ENABLED='0', GOTOOLCHAIN='local',
               GOPATH=str(root / 'go-work'), GOCACHE=str(root / 'go-cache'),
               GOENV='off', GOMAXPROCS='8')
    argv = [str(root / 'go/bin/go'), 'build', '-p', '8', '-o', str(work / 'bin/llm-d-inference-sim'),
            'cmd/llm-d-inference-sim/main.go']
    subprocess.run(argv, cwd=source, env=env, check=True)
    receipt = {'repo': repo, 'tag': tag, 'commit': commit, 'go': version, 'argv': argv,
               'binary_sha256': hashlib.sha256((work / 'bin/llm-d-inference-sim').read_bytes()).hexdigest()}
    (root / 'sim-build.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
