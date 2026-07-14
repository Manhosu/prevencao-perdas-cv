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


def _safe_unlink(path: Path, max_retries: int = 3) -> None:
    """Tenta apagar um arquivo com retentativas curtas em caso de
    PermissionError (arquivo travado, antivírus escaneando, mapeamento
    de imagem do processo em liberação). Levanta erro em português se
    todas as tentativas falharem."""
    for attempt in range(max_retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # backoff: 100ms, 200ms, 300ms
            else:
                raise RuntimeError(
                    f"Não foi possível apagar {path} após {max_retries} tentativas. "
                    f"O arquivo pode estar travado por um processo ou antivírus. "
                    f"Feche todos os aplicativos que possam estar usando o arquivo, "
                    f"apague-o manualmente e rode de novo.\n"
                    f"Detalhes técnicos: {e}"
                ) from e


def _binary_is_healthy(exe: Path) -> bool:
    """Roda `mediamtx --version` para confirmar que o binário não está
    corrompido. Sem isso, um download/extração interrompidos (Ctrl+C,
    antivírus, disco cheio) deixam um .exe quebrado em disco e toda
    execução futura levanta `OSError: [WinError 216]` sem retry."""
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _mediamtx_binary() -> Path:
    exe = BIN_DIR / ("mediamtx.exe" if platform.system() == "Windows" else "mediamtx")
    if exe.exists():
        if _binary_is_healthy(exe):
            return exe
        print(
            f"[dvr_sim] binário existente em {exe} está corrompido "
            "(download ou extração anterior deve ter sido interrompida); "
            "apagando e baixando de novo."
        )
        _safe_unlink(exe)

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
    _safe_unlink(dest)

    if not _binary_is_healthy(exe):
        _safe_unlink(exe)
        raise RuntimeError(
            f"[dvr_sim] o binário do MediaMTX baixado em {exe} está "
            "corrompido ('mediamtx --version' falhou). Verifique a conexão "
            "de rede, o antivírus e o espaço em disco, e rode de novo — o "
            "download será refeito automaticamente na próxima execução."
        )
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
        self._log_path: Path | None = None

    def url(self, channel: str) -> str:
        return f"rtsp://127.0.0.1:{self.port}/{channel}"

    def start(self) -> "DvrSim":
        exe = _mediamtx_binary()
        cfg = BIN_DIR / f"mediamtx-{self.port}.yml"
        # rtpAddress/rtcpAddress/srtAddress derivados da porta base: nos
        # defaults do MediaMTX (:8000, :8001, :8890) duas instâncias em
        # portas RTSP diferentes ainda colidem nessas portas auxiliares e a
        # segunda morre com "bind: só uma utilização de cada endereço de
        # soquete é permitida". SRT não é usado por este simulador, então
        # fica desligado.
        # RTP exige porta par; se a porta RTSP base for ímpar, derivamos
        # uma RTP base que seja par, e RTCP é RTP+1 (ímpar, como esperado).
        rtp_base = self.port + 100
        if rtp_base % 2 != 0:  # se ímpar, torna par
            rtp_base += 1
        cfg.write_text(
            f"rtspAddress: :{self.port}\n"
            f"rtpAddress: :{rtp_base}\n"
            f"rtcpAddress: :{rtp_base + 1}\n"
            f"srtAddress: :{rtp_base + 2}\n"
            "hls: no\nwebrtc: no\nrtmp: no\napi: no\nsrt: no\n"
            "paths:\n  all_others:\n",
            encoding="utf-8",
        )
        self._log_path = BIN_DIR / f"mediamtx-{self.port}.log"
        log_file = self._log_path.open("wb")
        try:
            self._server = subprocess.Popen(
                [str(exe), str(cfg)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_file.close()
        time.sleep(1.0)  # o servidor sobe em <1s
        self._ensure_server_alive()
        for ch in self.videos:
            self._publish(ch)
        time.sleep(1.5)  # dá tempo do ffmpeg começar a publicar
        return self

    def _ensure_server_alive(self) -> None:
        """`Popen` só levanta erro se o executável não puder ser iniciado;
        se o MediaMTX subir e morrer logo em seguida (ex.: porta em uso),
        `start()` retornaria "sucesso" com o servidor morto, e o erro só
        apareceria depois como um `isOpened()==False` inexplicável."""
        if self._server is None or self._server.poll() is None:
            return
        log_text = ""
        if self._log_path and self._log_path.exists():
            log_text = self._log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"MediaMTX morreu logo depois de iniciar na porta {self.port} "
            f"(código de saída {self._server.returncode}). "
            f"Log do MediaMTX ({self._log_path}):\n{log_text}"
        )

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

    @staticmethod
    def _terminate(p: subprocess.Popen, timeout: float = 5) -> None:
        """`p.wait(timeout=...)` sem tratamento derruba quem chama no meio
        de um loop de limpeza (ex.: `stop()` sobre vários canais), deixando
        os processos restantes órfãos. Aqui, um `TimeoutExpired` nunca
        escapa: cai para `kill()` e, se ainda assim travar, desiste em
        silêncio — o chamador sempre continua."""
        p.terminate()
        try:
            p.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        p.kill()
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def kill_stream(self, channel: str) -> None:
        """Simula queda de um canal (cabo solto, DVR reiniciando)."""
        p = self._publishers.pop(channel, None)
        if p:
            self._terminate(p)

    def restore_stream(self, channel: str) -> None:
        self._publish(channel)
        time.sleep(1.5)

    def stop(self) -> None:
        for ch in list(self._publishers):
            self.kill_stream(ch)
        if self._server:
            self._terminate(self._server)
            self._server = None

    def __enter__(self) -> "DvrSim":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
