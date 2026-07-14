"""DVR simulado: sobe um MediaMTX local e publica vídeos em loop como canais
RTSP. Permite testar reconexão derrubando um canal no meio do teste —
exatamente o que acontece quando o DVR do cliente reinicia."""
from __future__ import annotations

import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg

MEDIAMTX_VERSION = "v1.9.3"
BIN_DIR = Path("dev/bin")


def _mediamtx_binary() -> Path:
    exe = BIN_DIR / ("mediamtx.exe" if platform.system() == "Windows" else "mediamtx")
    if exe.exists():
        return exe
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    asset = (
        f"mediamtx_{MEDIAMTX_VERSION}_windows_amd64.zip"
        if platform.system() == "Windows"
        else f"mediamtx_{MEDIAMTX_VERSION}_linux_amd64.tar.gz"
    )
    url = (
        f"https://github.com/bluenviron/mediamtx/releases/download/"
        f"{MEDIAMTX_VERSION}/{asset}"
    )
    dest = BIN_DIR / asset
    print(f"[dvr_sim] baixando MediaMTX de {url}")
    urllib.request.urlretrieve(url, dest)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(dest) as z:
            z.extractall(BIN_DIR)
    else:
        shutil.unpack_archive(str(dest), str(BIN_DIR))
    dest.unlink(missing_ok=True)
    return exe


class DvrSim:
    """Uso:
    with DvrSim({"ch1": Path("a.mp4"), "ch2": Path("b.mp4")}) as sim:
        cv2.VideoCapture(sim.url("ch1"))
    """

    def __init__(self, videos: dict[str, Path], port: int = 8554) -> None:
        self.videos = {k: Path(v) for k, v in videos.items()}
        self.port = port
        self._server: subprocess.Popen | None = None
        self._publishers: dict[str, subprocess.Popen] = {}

    def url(self, channel: str) -> str:
        return f"rtsp://127.0.0.1:{self.port}/{channel}"

    def start(self) -> "DvrSim":
        exe = _mediamtx_binary()
        cfg = BIN_DIR / f"mediamtx-{self.port}.yml"
        cfg.write_text(
            f"rtspAddress: :{self.port}\nhls: no\nwebrtc: no\nrtmp: no\napi: no\n"
            "paths:\n  all_others:\n",
            encoding="utf-8",
        )
        self._server = subprocess.Popen(
            [str(exe), str(cfg)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)  # o servidor sobe em <1s
        for ch in self.videos:
            self._publish(ch)
        time.sleep(1.5)  # dá tempo do ffmpeg começar a publicar
        return self

    def _publish(self, channel: str) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self._publishers[channel] = subprocess.Popen(
            [
                ffmpeg, "-re", "-stream_loop", "-1",
                "-i", str(self.videos[channel]),
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-an", "-f", "rtsp", self.url(channel),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def kill_stream(self, channel: str) -> None:
        """Simula queda de um canal (cabo solto, DVR reiniciando)."""
        p = self._publishers.pop(channel, None)
        if p:
            p.terminate()
            p.wait(timeout=5)

    def restore_stream(self, channel: str) -> None:
        self._publish(channel)
        time.sleep(1.5)

    def stop(self) -> None:
        for ch in list(self._publishers):
            self.kill_stream(ch)
        if self._server:
            self._server.terminate()
            self._server.wait(timeout=5)
            self._server = None

    def __enter__(self) -> "DvrSim":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
