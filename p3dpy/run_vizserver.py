import asyncio
from asyncio.subprocess import PIPE, STDOUT
import webbrowser


_process = None


async def _spawn_vizserver(host: str = "127.0.0.1", port: int = 8000, timeout: int = 3):
    global _process
    if _process is not None:
        raise RuntimeError("Already vizserver has been spawned.")
    _process = await asyncio.create_subprocess_exec(*["vizserver", "--host", host, "--port", str(port)], stdout=PIPE, stderr=STDOUT)
    while True:
        try:
            line = await asyncio.wait_for(_process.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            break
        else:
            if len(line) == 0:
                break
            else:
                print(line.decode().rstrip())
                continue


async def _loop(timeout: int = 3):
    global _process
    if _process is None:
        return
    while True:
        try:
            line = await asyncio.wait_for(_process.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            pass
        else:
            print(line.decode().rstrip())
            continue
        await asyncio.sleep(1)


def vizspawn(host: str = "127.0.0.1", port: int = 8000):
    asyncio.get_event_loop().run_until_complete(_spawn_vizserver(host=host, port=port))


def vizloop(browser: bool = False, url: str = "http://127.0.0.1:8000"):
    if browser:
        webbrowser.open(url)
    asyncio.get_event_loop().run_until_complete(_loop())
