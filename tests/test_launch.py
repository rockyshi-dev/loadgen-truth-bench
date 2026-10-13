"""Common launch requirements, tested without HTTP traffic or real children."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import json
from pathlib import Path
import signal
import socket
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ltbench import launch
from ltbench import runner
from ltbench.analysis import EvidenceError
from ltbench.spec import MODEL, TOOLS, hf_home_of, hub_snapshot
from tests.helpers import HF_HOME, ROOT, TOKENIZER, command


class Environment(unittest.TestCase):
    def test_every_forbidden_family_fails_even_if_overridden_or_empty(self):
        for key in ('LD_PRELOAD', 'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS',
                    'VLLM_LOGGING_LEVEL', 'NCCL_DEBUG', 'CUDA_DEVICE_ORDER', 'CUDA_HOME'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(EvidenceError, key):
                    launch.launch_environment({key: ''}, {key: ''})
        launch.set_run_prefix('/work/envs/inference-perf')
        try:
            result = launch.launch_environment({'CUDA_VISIBLE_DEVICES': ''},
                                               {'CUDA_VISIBLE_DEVICES': '7', 'PATH': '/example'})
        finally:
            launch.set_run_prefix(None)
        self.assertEqual(result, {'CUDA_VISIBLE_DEVICES': '',
                                  'PATH': '/work/envs/inference-perf/bin:' + launch.SYSTEM_PATH,
                                  'CONDA_PREFIX': '/work/envs/inference-perf',
                                  'CONDA_DEFAULT_ENV': '/work/envs/inference-perf'})

    def test_bad_environment_is_rejected_before_any_child_or_output(self):
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '4'}), \
             patch.object(runner.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(EvidenceError, 'OMP_NUM_THREADS'):
                runner.OwnedProcess(['fake'], {}, [0], Path('/unused'), 'server')
            spawn.assert_not_called()

    def test_minimal_environment_is_in_manifest_and_popen(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            process = Mock(pid=987654321)
            with patch.dict(os.environ, {'PATH': '/example', 'EXTRA': 'dropped', 'HOME': '/h',
                                         'HF_HUB_OFFLINE': '1'}, clear=True), \
                 patch.object(runner.subprocess, 'Popen', return_value=process) as spawn, \
                 patch.object(runner, 'process_snapshot', return_value={}), \
                 patch.object(runner.os, 'killpg'):
                owned = runner.OwnedProcess(['fake'], {'GOMAXPROCS': '32'}, [0], directory, 'server')
                manifest = json.loads((directory / 'server.manifest.json').read_text())
                self.assertEqual(manifest['environment'], spawn.call_args.kwargs['env'])
                self.assertEqual(set(manifest['environment']),
                                 {'PATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'HOME', 'HF_HUB_OFFLINE',
                                  'GOMAXPROCS', 'CUDA_VISIBLE_DEVICES'})
                self.assertEqual(manifest['environment']['CUDA_VISIBLE_DEVICES'], '')
                owned.stop()

    def test_secret_parent_variables_never_reach_a_child_or_a_file(self):
        secret = 'sk-or-v1-0123456789abcdef0123456789abcdef'
        parent = {'PATH': '/usr/bin', 'HOME': '/h', 'OPENROUTER_API_KEY': secret, 'HF_TOKEN': secret,
                  'GITHUB_TOKEN': secret, 'SSH_CONNECTION': '10.0.0.1 22 10.0.0.2 22',
                  'SSH_AUTH_SOCK': '/tmp/agent', 'AWS_SECRET_ACCESS_KEY': secret}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            process = Mock(pid=987654321)
            with patch.dict(os.environ, parent, clear=True), \
                 patch.object(runner.subprocess, 'Popen', return_value=process) as spawn, \
                 patch.object(runner, 'process_snapshot', return_value={}), \
                 patch.object(runner.os, 'killpg'):
                for name in ('server', 'probe', 'tool'):
                    runner.OwnedProcess(['fake'], {}, [0], directory, name).stop()
                for call in spawn.call_args_list:
                    env = call.kwargs['env']
                    self.assertNotIn(secret, json.dumps(env))
                    self.assertFalse(any(k.startswith('SSH_') for k in env))
                runner.write_json(directory / 'server-env.json',
                                  launch.launch_environment(launch.server_environment([0, 1])))
            written = ''.join(p.read_text() for p in directory.iterdir() if p.is_file())
            self.assertNotIn(secret, written)
            self.assertNotIn('SSH_', written)
            self.assertNotIn('/tmp/agent', written)


class PortIdentity(unittest.TestCase):
    def test_reuseaddr_precedes_bind_and_live_listener_stops_start(self):
        sock = Mock()
        sock.__enter__ = Mock(return_value=sock)
        sock.__exit__ = Mock(return_value=False)
        with patch.object(launch.socket, 'socket', return_value=sock):
            launch.check_port(8188)
            self.assertEqual(sock.method_calls[0].args,
                             (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1))
            self.assertEqual(sock.method_calls[1].args, (('127.0.0.1', 8188),))
            sock.bind.side_effect = OSError('Address already in use')
            with self.assertRaisesRegex(EvidenceError, 'unavailable'):
                launch.check_port(8188)

    def test_listener_inode_must_belong_to_server_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'net').mkdir()
            (root / 'net/tcp').write_text('header\n0: 0100007F:1FFC 00000000:0000 0A 0 0 0 0 0 1234\n')
            (root / 'net/tcp6').write_text('header\n')
            (root / '102/fd').mkdir(parents=True)
            (root / '102/fd/3').symlink_to('socket:[1234]')
            snapshot = {100: dict(state='S', ppid=1, pgid=100, sid=100, start_ticks=10),
                        102: dict(state='S', ppid=100, pgid=102, sid=102, start_ticks=12)}
            with patch.object(launch, 'process_snapshot', return_value=snapshot):
                result = launch.listener_identity(100, 8188, {}, root)
                self.assertEqual(result['owners'], {'102': '1234'})
                with self.assertRaisesRegex(EvidenceError, 'not exclusively owned'):
                    launch.listener_identity(999, 8188, {}, root)

    def test_ready_model_mismatch_or_listener_mismatch_never_passes(self):
        async def exercise(tmp):
            response = AsyncMock(status=200)
            response.raise_for_status = Mock()
            session = Mock()
            session.get.return_value.__aenter__ = AsyncMock(return_value=response)
            session.get.return_value.__aexit__ = AsyncMock(return_value=False)
            context = AsyncMock()
            context.__aenter__.return_value = session
            process = Mock(directory=Path(tmp), known={})
            process.name = 'server'
            process.process.poll.return_value = None
            process.process.pid = 100
            with patch.object(runner, 'http_session', return_value=context), \
                 patch.dict('sys.modules', {'aiohttp': types.SimpleNamespace(ClientError=OSError)}), \
                 patch.object(runner, 'listener_identity') as check:
                response.json.return_value = {'data': [{'id': 'foreign-model'}]}
                with self.assertRaisesRegex(EvidenceError, 'another model'):
                    await runner.ready('http://127.0.0.1:8188', process)
                check.assert_not_called()
                response.json.return_value = {'data': [{'id': MODEL}]}
                check.side_effect = EvidenceError('foreign listener')
                with self.assertRaisesRegex(EvidenceError, 'foreign listener'):
                    await runner.ready('http://127.0.0.1:8188', process)
                check.side_effect = None
                check.return_value = {'owners': {'100': '1234'}}
                await runner.ready('http://127.0.0.1:8188', process)
                self.assertTrue((Path(tmp) / 'server.ready.json').is_file())
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(exercise(tmp))


class Transport(unittest.TestCase):
    def test_probe_and_scraper_pool_checkout_limit(self):
        module = types.SimpleNamespace(ClientSession=Mock(), ClientTimeout=Mock(), TCPConnector=Mock())
        with patch.dict('sys.modules', {'aiohttp': module}):
            launch.http_session(30, limit=0)
        module.TCPConnector.assert_called_once_with(limit=0, keepalive_timeout=1.0)

    def test_tools_run_unmodified_without_overlay_or_pythonpath(self):
        self.assertFalse(Path(__file__).with_name('sitecustomize.py').exists())
        for tool in TOOLS:
            env = command(tool)['env']
            self.assertNotIn('PYTHONPATH', env)

    def test_aiperf_exports_records_not_raw(self):
        argv = command('aiperf')['argv']
        self.assertEqual(argv[argv.index('--export-level') + 1], 'records')


class AiperfOfflineTokenizer(unittest.TestCase):
    """aiperf 0.13.0 offline mode rejects a directory as --tokenizer."""
    COMMIT = 'a09a35458c702b33eeacc393d103063234e8bc28'

    def test_snapshot_path_parsing(self):
        self.assertEqual(hub_snapshot(TOKENIZER), ('Qwen/Qwen2.5-7B-Instruct', self.COMMIT))
        self.assertEqual(hub_snapshot(str(TOKENIZER) + '/'), ('Qwen/Qwen2.5-7B-Instruct', self.COMMIT))
        for bad in ('/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct',
                    '/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/main',
                    '/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a3545',
                    '/models/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'not an HF cache snapshot'):
                    hub_snapshot(bad)

    def test_aiperf_argv_passes_repo_and_revision_not_directory(self):
        built = command('aiperf')
        argv = built['argv']
        self.assertEqual(argv[argv.index('--tokenizer') + 1], 'Qwen/Qwen2.5-7B-Instruct')
        self.assertEqual(argv[argv.index('--tokenizer-revision') + 1], self.COMMIT)
        self.assertNotIn(str(TOKENIZER), argv)
        self.assertEqual(built['env']['HF_HUB_OFFLINE'], '1')
        self.assertEqual(built['env']['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(built['env']['HF_HOME'], HF_HOME)
        self.assertEqual(command('aiperf', hf_home='/other/hf')['env']['HF_HOME'], '/other/hf')
        with self.assertRaises(ValueError):
            command('aiperf', tokenizer=Path('/tmp/tokenizer'))
        with self.assertRaisesRegex(ValueError, 'give --hf-home'):
            hf_home_of('/x/cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/' + self.COMMIT)
        # vLLM keeps the directory.
        argv = command('vllm')['argv']
        self.assertEqual(argv[argv.index('--tokenizer') + 1], str(TOKENIZER))

    def test_identity_check_requires_same_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / 'hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots' / self.COMMIT
            snapshot.mkdir(parents=True)
            other = Path(tmp) / 'elsewhere'
            other.mkdir()
            root = ROOT
            for stdout, code, ok in ((str(snapshot) + '\n', 0, True), (str(other) + '\n', 0, False),
                                     ('', 1, False)):
                done = types.SimpleNamespace(returncode=code, stdout=stdout, stderr='boom')
                with self.subTest(stdout=stdout, code=code), \
                        patch.object(runner.subprocess, 'run', return_value=done) as run:
                    if ok:
                        record = runner.aiperf_tokenizer_identity(snapshot, root)
                        self.assertEqual(record['resolved'], os.path.realpath(snapshot))
                        self.assertEqual(record['repo'], 'Qwen/Qwen2.5-7B-Instruct')
                        self.assertEqual(record['revision'], self.COMMIT)
                    else:
                        with self.assertRaisesRegex(EvidenceError, 'not the manifest tokenizer'):
                            runner.aiperf_tokenizer_identity(snapshot, root)
                    argv = run.call_args.args[0]
                    self.assertEqual(argv[0], str(root / 'envs/aiperf/bin/python'))
                    self.assertEqual(argv[-2:], ['Qwen/Qwen2.5-7B-Instruct', self.COMMIT])
                    env = run.call_args.kwargs['env']
                    self.assertEqual((env['HF_HUB_OFFLINE'], env['TRANSFORMERS_OFFLINE'], env['HF_HOME']),
                                     ('1', '1', str(tmp)))


class ChildSetup(unittest.TestCase):
    def test_fixed_nofile_and_parent_death_signal(self):
        with patch.object(launch.resource, 'getrlimit', return_value=(1024, 1048576)):
            self.assertEqual(launch.nofile_limit(65536)['soft'], 65536)
        with patch.object(launch.resource, 'getrlimit', return_value=(1024, 4096)):
            with self.assertRaisesRegex(EvidenceError, 'below the required'):
                launch.nofile_limit(65536)
        libc = Mock()
        libc.prctl.return_value = 0
        setup = launch.child_setup(65536)
        with patch.object(launch, '_LIBC', libc), \
             patch.object(launch.os, 'getppid', return_value=os.getpid()), \
             patch.object(launch.resource, 'setrlimit') as limit:
            setup()
        libc.prctl.assert_called_once_with(launch.PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        limit.assert_called_once_with(launch.resource.RLIMIT_NOFILE, (65536, 65536))
        with patch.object(launch, '_LIBC', libc), \
             patch.object(launch.os, 'getppid', return_value=1), \
             patch.object(launch.os, '_exit', side_effect=SystemExit) as leave:
            with self.assertRaises(SystemExit):
                setup()
            leave.assert_called_once_with(70)

    def test_tool_process_gets_the_limit_and_manifest_records_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=987654321)
            process.poll.return_value = None
            with patch.object(runner.subprocess, 'Popen', return_value=process) as spawn, \
                 patch.object(runner, 'process_snapshot', return_value={}), \
                 patch.object(runner, 'child_setup', return_value='setup') as setup, \
                 patch.object(runner.os, 'killpg'):
                owned = runner.OwnedProcess(['fake'], {}, [0], Path(tmp), 'tool', nofile=65536)
                setup.assert_called_once_with(65536)
                self.assertEqual(spawn.call_args.kwargs['preexec_fn'], 'setup')
                manifest = json.loads((Path(tmp) / 'tool.manifest.json').read_text())
                self.assertEqual(manifest['rlimit_nofile'], {'soft': 65536, 'hard': 65536})
                owned.stop()


class Signals(unittest.TestCase):
    def test_sigterm_runs_cleanup_before_the_runner_exits(self):
        cleaned = []

        async def work():
            try:
                asyncio.get_running_loop().call_later(.05, os.kill, os.getpid(), signal.SIGTERM)
                await asyncio.sleep(5)
            finally:
                cleaned.append(True)
        with self.assertRaisesRegex(EvidenceError, 'SIGTERM'):
            asyncio.run(runner.guarded(work()))
        self.assertEqual(cleaned, [True])


class Cleanup(unittest.TestCase):
    def test_reparented_descendant_with_new_session_remains_owned(self):
        known = {}
        first = {100: dict(state='S', ppid=1, pgid=100, sid=100, start_ticks=10),
                 102: dict(state='S', ppid=100, pgid=102, sid=102, start_ticks=12)}
        self.assertEqual(launch.owned_tree(100, known, first), {100, 102})
        second = {102: dict(state='S', ppid=1, pgid=102, sid=102, start_ticks=12)}
        self.assertEqual(launch.owned_tree(100, known, second), {102})
        second[102]['start_ticks'] = 99
        self.assertEqual(launch.owned_tree(100, known, second), set())
        second[100] = dict(state='S', ppid=1, pgid=100, sid=100, start_ticks=99)
        self.assertEqual(launch.owned_tree(100, known, second), set())

    def test_surviving_tree_prevents_next_launch_after_sixty_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=987654321)
            process.poll.return_value = None
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                 patch.object(runner, 'process_snapshot', return_value={}), \
                 patch.object(runner.os, 'killpg') as kill:
                owned = runner.OwnedProcess(['fake'], {}, [0], Path(tmp), 'server')
                with patch.object(owned, 'group_alive', return_value=True), \
                     patch.object(runner.time, 'monotonic', side_effect=[0, 60]):
                    with self.assertRaisesRegex(EvidenceError, '60 s cleanup'):
                        owned.stop()
                # After the leader is reaped the group is never signalled again.
                self.assertEqual([c.args[1] for c in kill.call_args_list], [signal.SIGTERM])
                owned.log.close()

    def test_reaped_leader_group_is_not_signalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=987654321)
            process.poll.return_value = 0
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                 patch.object(runner, 'process_snapshot', return_value={}), \
                 patch.object(runner.os, 'killpg') as kill:
                owned = runner.OwnedProcess(['fake'], {}, [0], Path(tmp), 'server')
                owned.stop()
                owned.stop()  # idempotent
                kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
