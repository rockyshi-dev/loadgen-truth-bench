"""Golden command lines: every tool's argv, environment additions and config file, and the
simulator's argv, for one panel configuration (rate 500), against fixed expected values."""
import json
from pathlib import Path
import unittest

from ltbench.spec import TOOLS, server_command, tool_command

GOLDEN = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'commands_rate_500.json').read_text())


class Golden(unittest.TestCase):
    def test_tool_commands_at_rate_500(self):
        p = GOLDEN['placeholders']
        self.assertEqual(set(GOLDEN['tools']), set(TOOLS))
        for tool in TOOLS:
            with self.subTest(tool=tool):
                built = tool_command(tool, p['rate'], p['directory'], p['tokenizer'], p['port'],
                                     root=p['root'], vllm_env=p['vllm_env'], hf_home=p['hf_home'])
                expected = GOLDEN['tools'][tool]
                self.assertEqual(built['argv'], expected['argv'])
                self.assertEqual(built['env'], expected['env'])
                self.assertEqual(built['files'], expected['files'])   # inference-perf YAML, byte for byte
                self.assertEqual(built['expected_workers'], expected['expected_workers'])

    def test_server_argv(self):
        p = GOLDEN['placeholders']
        self.assertEqual(server_command(p['port'], root=p['root']), GOLDEN['server_argv'])

    def test_request_rate_and_count_scale_with_rate(self):
        p = GOLDEN['placeholders']
        for rate in (10, 50, 200, 1000):
            argv = tool_command('vllm', rate, p['directory'], p['tokenizer'], root=p['root'],
                                vllm_env=p['vllm_env'], hf_home=p['hf_home'])['argv']
            expected = [str(rate) if a == '500' else str(60 * rate) if a == '30000' else a
                        for a in GOLDEN['tools']['vllm']['argv']]
            self.assertEqual(argv, expected)


if __name__ == '__main__':
    unittest.main()
