"""The Go toolchain archive is checked against the pinned size and sha256; no release list is requested."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ltbench import build_sim
from ltbench.spec import GO_TOOLCHAIN


class BuildSimTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
