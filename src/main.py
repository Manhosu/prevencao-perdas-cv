"""Entrypoint headless. A UI (Plano 3) reaproveita o mesmo Pipeline.

    python -m src.main --config config/config.json
"""
from __future__ import annotations

import argparse
import logging
import signal
import time

from src.config.settings import AppConfig, ConfigError
from src.inference.engine import InferenceEngine
from src.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Prevenção de Perdas — núcleo")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--status-every", type=float, default=5.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    try:
        cfg = AppConfig.load(args.config)
    except ConfigError as e:
        log.error("%s", e)
        return 2

    pipeline = Pipeline(cfg, InferenceEngine(cfg.inference))

    def _handle_person(result, frame):
        if result.had_person:
            log.info(
                "[%s] %d pessoa(s) na zona — ids=%s",
                result.camera_name,
                len(result.persons),
                [p.person.track_id for p in result.persons],
            )

    pipeline.on_result = _handle_person

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    # SIGTERM: no Windows o handler só é entregue quando o próprio processo
    # se sinaliza (ex.: os.kill(pid, SIGTERM)) — TerminateProcess/taskkill
    # /F não passam pelo interpretador Python e não têm como ser
    # capturados. `signal.signal` para SIGTERM funciona no CPython deste
    # Windows (verificado), mas registramos defensivamente com getattr:
    # em builds/ambientes onde SIGTERM não existir ou não puder ser
    # registrado, o processo continua parável por Ctrl+C (SIGINT) em vez
    # de travar na inicialização.
    _sigterm = getattr(signal, "SIGTERM", None)
    if _sigterm is not None:
        try:
            signal.signal(_sigterm, _stop)
        except (ValueError, OSError):
            log.warning("não foi possível registrar handler de SIGTERM; use Ctrl+C para parar")

    pipeline.start()
    try:
        while not stopping:
            time.sleep(args.status_every)
            for name, st in pipeline.status().items():
                log.info(
                    "câmera '%s': %s · %.1f fps · %d frames descartados",
                    name, st["state"], st["fps"], st["dropped"],
                )
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
