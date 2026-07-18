"""Entrypoint. Headless por padrão; `--ui` abre a janela do Plano 4, que
reaproveita o mesmo Pipeline/Database/AppConfig já construídos aqui.

    python -m src.main --config config/config.json          # headless
    python -m src.main --config config/config.json --ui     # com interface
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from src.alerts.alert_queue import AlertQueue
from src.alerts.telegram_alert import TelegramSender
from src.config.settings import AppConfig, ConfigError
from src.evidence.recorder import EvidenceRecorder
from src.evidence.retention import RetentionJob
from src.inference.engine import InferenceEngine
from src.pipeline import Pipeline
from src.storage.db import Database
from src.watchdog.monitor import Watchdog


def main() -> int:
    ap = argparse.ArgumentParser(description="Prevenção de Perdas — núcleo")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--status-every", type=float, default=5.0)
    ap.add_argument("--ui", action="store_true",
                    help="abre a janela (Plano 4) em vez de rodar headless")
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

    db = Database("data/app.db")
    db.init_schema()
    recorder = EvidenceRecorder(db, cfg.evidence, cfg.store)
    sender = TelegramSender(cfg.telegram)
    alerts = AlertQueue(sender, db, rate_limit_per_min=cfg.telegram.rate_limit_per_min,
                       send_photo=cfg.telegram.send_photo, send_clip=cfg.telegram.send_clip)
    retention = RetentionJob(db, cfg.evidence)
    watchdog = Watchdog(pipeline.threads, db, cfg.watchdog, alert_queue=alerts)

    if not sender.configured:
        log.warning("Telegram sem token/chat_id no config — os alertas ficam so "
                    "registrados no banco, sem envio.")

    def _on_result(result, frame):
        if result.had_person:
            log.info(
                "[%s] %d pessoa(s) na zona — ids=%s",
                result.camera_name,
                len(result.persons),
                [p.person.track_id for p in result.persons],
            )
        for ev in result.events:
            res = recorder.record(ev, result.camera_name, frame.image,
                                  clip_buffer=pipeline.clip_buffers.get(result.camera_name))
            caption = sender.caption_for(cfg.store.name, result.camera_name,
                                         res.ts_local, ev.zone)
            alerts.enqueue(res.event_id, res.image_path, res.clip_path, caption)
            log.info("OCULTACAO em '%s' (zona %s, score %.2f) — evidencia #%s",
                     result.camera_name, ev.zone, ev.score, res.event_id)

    pipeline.on_result = _on_result

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

    try:
        alerts.start()
        retention.start()
        watchdog.start()
        pipeline.start()
        if args.ui:
            # Import tardio: rodar headless (o caso comum, ex.: serviço na
            # loja) não pode passar a exigir PySide6 nem abrir display algum.
            from PySide6.QtWidgets import QApplication

            from src.ui.app import MainWindow

            qapp = QApplication.instance() or QApplication(sys.argv)
            window = MainWindow(pipeline, db, cfg, args.config)
            window.show()
            qapp.exec()
            # Ao fechar a janela, cai para o `finally` abaixo: o shutdown
            # (pipeline -> watchdog -> retention -> alerts -> db) é o mesmo
            # do modo headless, nunca um caminho separado.
        else:
            while not stopping:
                time.sleep(args.status_every)
                for name, st in pipeline.status().items():
                    log.info(
                        "câmera '%s': %s · %.1f fps · %d frames descartados",
                        name, st["state"], st["fps"], st["dropped"],
                    )
    finally:
        # Ordem importa: para o pipeline PRIMEIRO (para de gerar eventos novos);
        # so entao para watchdog/retention; alerts.stop() por ultimo entre os
        # servicos, pra drenar o que ainda estiver na fila; db.close() e o
        # derradeiro. Cada parada no seu proprio try/except: uma falha aqui
        # nao pode impedir as demais de rodar (e o db.close() sempre acontece).
        for nome, parar in (
            ("pipeline", pipeline.stop),
            ("watchdog", watchdog.stop),
            ("retention", retention.stop),
            ("alertas", alerts.stop),
            ("banco de dados", db.close),
        ):
            try:
                parar()
            except Exception:
                log.exception("falha ao parar '%s'", nome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
