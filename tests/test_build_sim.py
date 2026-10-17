"""The Go toolchain archive is checked against the pinned size and sha256 (no release list is requested), and the
simulator source is fetched from any git remote and verified by its commit id."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ltbench import build_sim
from ltbench.spec import GO_TOOLCHAIN


class ToolchainTests(unittest.TestCase):
    def test_pin_is_well_formed(self):
        self.assertRegex(GO_TOOLCHAIN['sha256'], r'^[0-9a-f]{64}$')
        self.assertEqual(GO_TOOLCHAIN['filename'], f"{GO_TOOLCHAIN['version']}.linux-amd64.tar.gz")
        self.assertGreater(GO_TOOLCHAIN['size'], 0)

    def test_archive_fetched_from_mirror_and_checked_without_a_release_list(self):
        fetched = []

        def fake_fetch(url, path):
            fetched.append(url)
            Path(path).write_bytes(b'not a toolchain')

        with tempfile.TemporaryDirectory() as work, \
                mock.patch.object(build_sim, 'fetch', fake_fetch), \
                mock.patch.object(build_sim.urllib.request, 'urlopen', side_effect=AssertionError('network')), \
                mock.patch.dict(build_sim.os.environ, {'GO_DOWNLOAD_BASE': 'https://mirror.example/go/'}):
            with self.assertRaises(ValueError) as caught:
                build_sim.main(work)
        self.assertEqual(fetched, ['https://mirror.example/go/' + GO_TOOLCHAIN['filename']])
        self.assertIn('expected %d' % GO_TOOLCHAIN['size'], str(caught.exception))

    def test_right_size_wrong_hash_is_rejected(self):
        def fake_fetch(url, path):
            with open(path, 'wb') as out:
                out.truncate(GO_TOOLCHAIN['size'])   # a sparse file of the pinned size: all zero bytes

        self.assertNotEqual(hashlib.sha256(b'\0' * GO_TOOLCHAIN['size']).hexdigest(), GO_TOOLCHAIN['sha256'])
        with tempfile.TemporaryDirectory() as work, \
                mock.patch.object(build_sim, 'fetch', fake_fetch), \
                mock.patch.object(build_sim.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            with self.assertRaises(ValueError) as caught:
                build_sim.main(work)
        self.assertIn('sha256 differs', str(caught.exception))


def git(*args, cwd):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class SourceCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        upstream = Path(self.tmp.name) / 'upstream'
        upstream.mkdir()
        git('init', '-q', '-b', 'main', cwd=upstream)
        git('config', 'user.email', 'test@example.invalid', cwd=upstream)
        git('config', 'user.name', 'test', cwd=upstream)
        (upstream / 'main.go').write_text('package main\n')
        git('add', 'main.go', cwd=upstream)
        git('commit', '-q', '-m', 'first', cwd=upstream)
        self.first = git('rev-parse', 'HEAD', cwd=upstream)
        (upstream / 'main.go').write_text('package main\n// second\n')
        git('commit', '-q', '-am', 'second', cwd=upstream)
        self.second = git('rev-parse', 'HEAD', cwd=upstream)
        self.url = upstream.as_uri()

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkout_from_a_mirror_url_lands_on_the_pinned_commit(self):
        source = Path(self.tmp.name) / 'sim-source'
        self.assertEqual(build_sim.checkout_source(source, self.url, self.first), self.url)
        self.assertEqual(git('rev-parse', 'HEAD', cwd=source), self.first)
        self.assertEqual((source / 'main.go').read_text(), 'package main\n')
        # A second call with the tree already in place fetches nothing and stays put.
        with mock.patch.object(build_sim.subprocess, 'run', wraps=subprocess.run) as run:
            build_sim.checkout_source(source, 'https://unreachable.invalid/x.git', self.first)
        self.assertFalse(any('fetch' in c.args[0] for c in run.call_args_list))
        self.assertEqual(git('rev-parse', 'HEAD', cwd=source), self.first)

    def test_a_commit_the_remote_does_not_have_fails(self):
        source = Path(self.tmp.name) / 'sim-source'
        with self.assertRaises(subprocess.CalledProcessError):
            build_sim.checkout_source(source, self.url, '0' * 40)

    def test_a_modified_tree_is_refused(self):
        source = Path(self.tmp.name) / 'sim-source'
        build_sim.checkout_source(source, self.url, self.second)
        (source / 'main.go').write_text('package main\n// edited\n')
        with self.assertRaises(ValueError):
            build_sim.checkout_source(source, self.url, self.second)


if __name__ == '__main__':
    unittest.main()
