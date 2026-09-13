"""Entrypoint. Headless por padrão; `--ui` abre a janela do Plano 4, que
reaproveita o mesmo Pipeline/Database/AppConfig já construídos aqui.

    python -m src.main --config config/config.json          # headless
    python -m src.main --config config/config.json --ui     # com interface
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from src.alerts.alert_queue import AlertQueue
from src.alerts.rate_gate import CameraAlertGate
from src.alerts.telegram_alert import TelegramSender
from src.config.paths import data_dir, default_config_path, ensure_config
from src.config.settings import AppConfig, ConfigError
from src.evidence.recorder import EvidenceRecorder
from src.evidence.retention import RetentionJob
from src.inference.engine import InferenceEngine
from src.licensing.activation import Estado as EstadoLicenca
from src.licensing.activation import avaliar as avaliar_licenca
from src.licensing.fingerprint import codigo_da_maquina
from src.pipeline import Pipeline
from src.storage.db import Database
from src.watchdog.monitor import Watchdog


def carregar_config(path) -> AppConfig:
    """Carrega o config da loja, semeando um na primeira execução.

    Numa instalação nova não existe `config.json` — e não pode existir, porque
    ele guarda o token do Telegram do revendedor e as câmeras daquela loja. Sem
    semear, o app morria aqui com exit 2 e o lojista via a janela abrir e
    fechar. A tela que cadastra câmera e token está DENTRO do programa, então
    ele precisa abrir mesmo sem configuração nenhuma."""
    if ensure_config(path):
        logging.getLogger("main").info(
            "primeira execução: configuração criada em %s", path)
    return AppConfig.load(path)


# --- log e avisos visíveis ---------------------------------------------------
#
# O app instalado roda SEM janela de console (build `--windowed`). Relato de
# campo (set/2026): na primeira abertura, sem licença, "dá erro e fecha
# sozinho". Com console, o que aparecia primeiro era uma janela preta com
# linhas ERROR — que o lojista lê como defeito —, e qualquer falha de verdade
# sumia junto com o console, sem deixar rastro.
#
# Sem console, o log vai para um arquivo e todo motivo de encerramento com
# janela vira caixa de mensagem em português. O programa não some calado: foi
# exatamente o bug de julho, e ele não pode voltar por outro caminho.

_LOG_PRONTO = False


def arquivo_de_log():
    """Onde o suporte encontra o registro do que aconteceu."""
    return data_dir() / "logs" / "app.log"


def configurar_log() -> None:
    """Console no desenvolvimento; arquivo no app instalado. Idempotente."""
    global _LOG_PRONTO
    if _LOG_PRONTO:
        return
    from logging.handlers import RotatingFileHandler

    from src.config.paths import is_frozen

    formato = logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    raiz = logging.getLogger()
    raiz.setLevel(logging.INFO)

    if is_frozen():
        destino = arquivo_de_log()
        destino.parent.mkdir(parents=True, exist_ok=True)
        arquivo = RotatingFileHandler(destino, maxBytes=2_000_000,
                                      backupCount=3, encoding="utf-8")
        arquivo.setFormatter(formato)
        raiz.addHandler(arquivo)
        # Empacotado sem console, sys.stdout e sys.stderr são None. O
        # Ultralytics e o tqdm escrevem direto neles e derrubariam o processo
        # com AttributeError na primeira linha impressa. Vão para um arquivo à
        # parte, recriado a cada abertura: tamanho limitado, e sem disputar o
        # mesmo arquivo com a rotação do log principal.
        if sys.stdout is None or sys.stderr is None:
            saida = open(destino.parent / "saida.log", "w",
                         encoding="utf-8", buffering=1)
            if sys.stdout is None:
                sys.stdout = saida
            if sys.stderr is None:
                sys.stderr = saida
    else:
        console = logging.StreamHandler()
        console.setFormatter(formato)
        raiz.addHandler(console)
    _LOG_PRONTO = True


def _avisar(titulo: str, texto: str) -> None:
    """Caixa de mensagem na frente de tudo. Só é usada com `--ui`.

    Sem console, é o único jeito de o lojista saber POR QUE o programa não
    abriu, em vez de vê-lo sumir."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841 mantém viva
    caixa = QMessageBox()
    caixa.setIcon(QMessageBox.Icon.Warning)
    caixa.setWindowTitle(titulo)
    caixa.setText(texto)
    caixa.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    caixa.exec()


def executar() -> int:
    """Ponto de entrada do executável. Última linha de defesa: nenhuma falha
    inesperada pode fechar o programa sem dizer por quê."""
    try:
        return main()
    except Exception as e:  # noqa: BLE001 — capturar tudo é o objetivo aqui
        logging.getLogger("main").exception("falha inesperada ao abrir o sistema")
        if "--ui" in sys.argv:
            try:
                _avisar("O sistema encontrou um erro",
                        "O sistema não conseguiu abrir.\n\n"
                        f"Detalhe: {e}\n\n"
                        "Mande para o suporte o arquivo de registro:\n"
                        f"{arquivo_de_log()}")
            except Exception:
                logging.getLogger("main").exception(
                    "não foi possível nem mostrar a mensagem de erro")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Prevenção de Perdas — núcleo")
    ap.add_argument("--config", default=None,
                    help="padrão: config.json na pasta de dados do app")
    ap.add_argument("--status-every", type=float, default=5.0)
    ap.add_argument("--ui", action="store_true",
                    help="abre a janela (Plano 4) em vez de rodar headless")
    args = ap.parse_args()

    configurar_log()
    log = logging.getLogger("main")

    config_path = args.config or default_config_path()
    try:
        cfg = carregar_config(config_path)
    except (ConfigError, OSError) as e:
        log.error("%s", e)
        if args.ui:
            _avisar("Configuração com problema",
                    "O sistema não conseguiu ler a configuração desta loja.\n\n"
                    f"{e}\n\nArquivo: {config_path}")
        return 2

    # --- portão de licença ---------------------------------------------------
    # Antes de qualquer coisa cara (carregar modelo, abrir câmera): uma máquina
    # não liberada não roda. A licença é assinada e amarrada a ESTE computador,
    # então copiar a pasta instalada para outro PC não funciona.
    #
    # Com --ui a tela de liberação resolve na hora (mostra o código da máquina
    # e recebe a licença). Headless o processo sai com código 3 explicando o
    # que fazer. Nos dois casos, nunca em silêncio.
    licenca_dir = data_dir()
    ativacao = avaliar_licenca(licenca_dir)
    if not ativacao.pode_rodar:
        codigo = codigo_da_maquina()
        if not args.ui:
            log.error("%s", ativacao.mensagem)
            log.error("Código desta máquina: %s", codigo)
            return 3
        # Com janela, máquina ainda não liberada é o estado NORMAL da primeira
        # abertura, não um erro — e "ERROR" era justamente o que o lojista lia
        # como defeito. Aqui é INFO; o que ele vê é a tela de liberação.
        log.info("máquina não liberada (código %s): abrindo a tela de liberação",
                 codigo)
        from PySide6.QtWidgets import QApplication

        from src.ui.activation_dialog import pedir_ativacao

        qapp = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
        if not pedir_ativacao(licenca_dir, ativacao.mensagem):
            log.info("tela de liberação fechada sem licença; encerrando")
            _avisar("Sistema ainda não liberado",
                    "O sistema só funciona depois de liberado.\n\n"
                    f"Código deste computador:   {codigo}\n\n"
                    "Envie esse código para o seu fornecedor. Quando receber a "
                    "licença, abra o programa de novo e cole na tela.")
            return 3
        ativacao = avaliar_licenca(licenca_dir)
    if ativacao.estado is EstadoLicenca.TOLERANCIA:
        log.warning("%s", ativacao.mensagem)
        if args.ui:
            # A tolerância só serve se o lojista SOUBER dela: sem console, um
            # aviso só no log significaria descobrir no dia do bloqueio.
            _avisar("Licença precisa ser renovada",
                    f"{ativacao.mensagem}\n\n"
                    f"Código atual deste computador:   {codigo_da_maquina()}")

    pipeline = Pipeline(cfg, InferenceEngine(cfg.inference))

    # O banco vai na área gravável: instalado, a pasta do programa fica em
    # Program Files, onde usuário comum não escreve.
    banco = data_dir() / "data" / "app.db"
    banco.parent.mkdir(parents=True, exist_ok=True)
    db = Database(banco)
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
    elif not cfg.telegram.enabled:
        log.warning("MODO TESTE (telegram.enabled=false): o sistema vigia e "
                    "registra na aba Eventos, mas NAO envia nada no grupo.")

    # Trava anti-enxurrada POR CÂMERA (bug de campo 21/jul: 6 câmeras de teto
    # geraram 300+ mensagens numa tarde). O cooldown do analyzer é por pessoa;
    # este é por câmera. A evidência é gravada SEMPRE (o histórico/aba Eventos
    # fica completo) — só o envio ao Telegram é que respeita o intervalo.
    alert_gate = CameraAlertGate(cfg.telegram.min_seconds_between_alerts)

    # Detector de arma: SÓ é consultado dentro do alerta de pânico (gated), e
    # só existe se ligado no config. Nunca roda no dia a dia → sem alarme falso
    # de celular. Carrega o modelo preguiçosamente na primeira consulta.
    weapon_detector = None
    if cfg.detection.weapon.enabled:
        from src.detection.weapon import WeaponDetector
        from src.config.paths import resource_dir
        wcfg = cfg.detection.weapon
        modelo = wcfg.model
        # caminho relativo → resolve no bundle (mesma regra dos modelos de pose)
        if not os.path.isabs(modelo):
            modelo = str(resource_dir() / modelo)
        weapon_detector = WeaponDetector(modelo, wcfg.threshold)
        log.info("detecção de arma no pânico LIGADA (modelo %s, limiar %.2f)",
                 modelo, wcfg.threshold)

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
            panico = getattr(ev, "kind", "ocultacao") == "panico"
            # Pânico é assalto: NÃO passa pelo anti-enxurrada por câmera (não se
            # segura um alerta de assalto por causa de rate-limit).
            if not panico and not alert_gate.allow(result.camera_name, ev.ts):
                log.info("OCULTACAO em '%s' (evidencia #%s) — alerta SUPRIMIDO "
                         "(anti-enxurrada; %d suprimidos nesta camera)",
                         result.camera_name, res.event_id,
                         alert_gate.suprimidos(result.camera_name))
                continue
            if panico:
                arma = (weapon_detector.has_weapon(frame.image)
                        if weapon_detector is not None else False)
                caption = sender.caption_panico(cfg.store.display_name,
                                                result.camera_name, res.ts_local,
                                                arma=arma)
            else:
                caption = sender.caption_for(cfg.store.display_name, result.camera_name,
                                             res.ts_local, ev.zone)
            alerts.enqueue(res.event_id, res.image_path, res.clip_path, caption)
            log.info("%s em '%s' — evidencia #%s",
                     "PANICO" if panico else f"OCULTACAO (zona {ev.zone}, score {ev.score:.2f})",
                     result.camera_name, res.event_id)

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
            # `config_path`, não `args.config`: este último é None quando o
            # usuário não passou a flag, e a janela grava aqui ao salvar
            # câmeras, zonas e o token do Telegram.
            window = MainWindow(pipeline, db, cfg, config_path)
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
    raise SystemExit(executar())
