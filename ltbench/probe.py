"""Independent 10 Hz completions probe. Importing this module sends no traffic.

One request every 100 ms on a fixed grid. The first line of the log is the
grid record {grid_origin_ns, period_ns, written_ns}: grid_origin_ns is the planned
send of the first request, and the line is written and flushed before that send.

The runner starts it as its own process (`python probe.py ...`), so its
timestamps never share an event loop with /proc scans, JSON dumps or the
supervision of the tool. SIGTERM stops new requests; in-flight ones drain.
"""
import argparse
import asyncio
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):   # started as a script by the runner, in its own process
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltbench.spec import MODEL
from ltbench.analysis import PROBE_PERIOD_NS
from ltbench.launch import http_session, launch_environment


class SSE:
    """Decode complete SSE events independently of TCP/aiohttp read boundaries."""
    def __init__(self):
        self.pending = b''

    def feed(self, chunk):
        self.pending += chunk
        output = []
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            line = line.rstrip(b'\r')
            # The simulator emits one data line per event. Ignore comments and
            # event metadata. Incomplete UTF-8 bytes are held until newline.
            if line.startswith(b'data:'):
                payload = line[5:].strip()
                if payload:
                    output.append(payload)
        return output


async def request(session, url, prompt, index, planned_ns):
    record = {'index': index, 'planned_ns': planned_ns, 'start_ns': time.monotonic_ns(),
              'content_ns': [], 'usage_tokens': None, 'done': False, 'error': None}
    body = {'model': MODEL, 'prompt': prompt, 'max_tokens': 50, 'ignore_eos': True,
            'stream': True, 'stream_options': {'include_usage': True}}
    try:
        async with session.post(url + '/v1/completions', json=body,
                                headers={'X-Probe-Id': str(index)}) as response:
            response.raise_for_status()
            record['http_status'] = response.status
            decoder = SSE()
            async for chunk in response.content.iter_any():
                received = time.monotonic_ns()
                for payload in decoder.feed(chunk):
                    if payload == b'[DONE]':
                        record['done'] = True
                        continue
                    event = json.loads(payload)
                    if event.get('error'):
                        raise ValueError(str(event['error']))
                    if event.get('usage') is not None:
                        record['usage_tokens'] = event['usage']['completion_tokens']
                    choices = event.get('choices', [])
                    # Empty finishing choices/usage are not content tokens.
                    if any(choice.get('text') for choice in choices):
                        record['content_ns'].append(received)
            if decoder.pending.strip():
                raise ValueError('Truncated SSE line')
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['end_ns'] = time.monotonic_ns()
    times = record['content_ns']
    record['ttft_s'] = (times[0] - record['start_ns']) / 1e9 if times else None
    record['e2e_s'] = (record['end_ns'] - record['start_ns']) / 1e9
    record['gaps_s'] = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
    return record


# Lead between writing the grid record and the first planned send.
ORIGIN_LEAD_NS = 20_000_000


async def run_probe(url, prompt, stop, output, rows=None, period_ns=PROBE_PERIOD_NS):
    launch_environment({})
    tasks = set()
    rows = [] if rows is None else rows
    with output.open('x') as stream:
        written = time.monotonic_ns()
        start = written + ORIGIN_LEAD_NS
        stream.write(json.dumps({'grid_origin_ns': start, 'period_ns': period_ns,
                                 'written_ns': written}) + '\n')
        stream.flush()
        async def one(session, index, planned):
            row = await request(session, url, prompt, index, planned)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            rows.append(row)
        async with http_session(30, limit=0) as session:
            index = 0
            while not stop.is_set():
                planned = start + index * period_ns
                delay = max(0, (planned - time.monotonic_ns()) / 1e9)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    break
                except asyncio.TimeoutError:
                    pass
                task = asyncio.create_task(one(session, index, planned))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                index += 1
            if tasks:
                await asyncio.gather(*tasks)
    return rows


async def main_async(url, prompt, output):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(sig, stop.set)
    await run_probe(url, prompt, stop, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args.url, args.prompt_file.read_text(), args.output))


if __name__ == '__main__':
    main()
