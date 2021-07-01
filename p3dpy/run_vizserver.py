import os
from typing import Any, Optional
import json
import asyncio
from asyncio.subprocess import PIPE, STDOUT
import signal
import webbrowser


class ServerProcess:
    _process: Optional[Any] = None

    def __del__(self):
        if self._process is not None:
            os.kill(self._process.pid, signal.SIGTERM)


_sp = ServerProcess()


async def _spawn_vizserver(host: str = "127.0.0.1", port: int = 8000, timeout: int = 3, params: dict = {}):
    global _sp
    if _sp._process is not None:
        raise RuntimeError("Already vizserver has been spawned.")
    for k, v in params.items():
        if len(v) < 3:
            raise ValueError("Parameter value must be 3 or 4 elements tuple.")
        elif len(v) == 3:
            params[k] = tuple(list(v) + [(v[2] - v[1]) / 100])

    _sp._process = await asyncio.create_subprocess_exec(
        *["vizserver", "--host", host, "--port", str(port), "--params", json.dumps(params)], stdout=PIPE, stderr=STDOUT
    )
    while True:
        try:
            if _sp._process.stdout is not None:
                line = await asyncio.wait_for(_sp._process.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            break
        else:
            if len(line) == 0:
                break
            else:
                print(line.decode().rstrip())
                continue


async def _loop(timeout: int = 3):
    global _sp
    if _sp._process is None:
        return
    while True:
        try:
            line = await asyncio.wait_for(_sp._process.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            pass
        else:
            if len(line) > 0:
                print(line.decode().rstrip())
            continue
        await asyncio.sleep(1)


def vizspawn(host: str = "127.0.0.1", port: int = 8000, params: dict = {}) -> None:
    asyncio.get_event_loop().run_until_complete(_spawn_vizserver(host=host, port=port, params=params))


def vizloop(browser: bool = False, url: str = "http://127.0.0.1:8000") -> None:
    if browser:
        webbrowser.open(url)
    asyncio.get_event_loop().run_until_complete(_loop())
