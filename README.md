# Prevenção de Perdas com Visão Computacional — MVP

Software **100% local para Windows** que transforma câmeras de segurança existentes (IP e DVR via **RTSP**) em um sistema inteligente de **alerta em tempo real**, detectando comportamentos de **ocultação de produtos** e notificando a equipe via **Telegram**.

> Este README é o blueprint técnico de ponta a ponta do projeto. Serve como base para iniciar o desenvolvimento (inclusive com o Claude Code).

---

## 1. Objetivo & Contexto

- **Cliente:** Adriano (revende a solução para mercados e farmácias).
- **Objetivo:** MVP ágil de prevenção de perdas — detectar em tempo real quando uma pessoa oculta um produto (mão levando item ao **bolso, bolsa, mochila ou sob a roupa**), capturar evidência e alertar a equipe.
- **NÃO é** sistema forense de revisão pós-evento nem gravador de vídeo. É uma **camada de alerta inteligente**.
- **Modelo de negócio do cliente:** instala em vários estabelecimentos → precisa ser **plug-and-play** (sem treino por loja), com **configuração rápida por câmera**.

---

## 2. Requisitos (do escopo acordado)

| # | Requisito | Observação |
|---|---|---|
| R1 | Execução **100% local** no Windows | Sem nuvem. Privacidade total, zero custo recorrente. |
| R2 | Compatível com **câmeras IP e DVR** via **RTSP** | DVR expõe cada câmera como um canal RTSP → seleciona-se quais monitorar. |
| R3 | Até **5 câmeras simultâneas** | Desempenho depende do hardware; ajustável por FPS/resolução. |
| R4 | **Otimizado para hardware modesto** | Processamento pesado só quando há pessoa na zona monitorada. |
| R5 | **Sem treino por loja** | Modelos pré-treinados; funciona em qualquer local. |
| R6 | **Instalação simplificada** | Instalador Windows, sem exigir conhecimento técnico. |
| R7 | **Configuração intuitiva** de câmeras e áreas monitoradas | Assistente visual: apontar RTSP + desenhar a zona. |
| R8 | Detecção com **alto nível de confiança** | Limiar configurável para reduzir falsos alertas. |
| R9 | **Evidência**: imagem + data/hora do evento | Salvos localmente. |
| R10 | **Alerta discreto via Telegram** | Imagem + timestamp + câmera. |

### Expectativas honestas do MVP
- A detecção de **ação/ocultação** é um problema difícil de visão computacional. Este MVP usa uma abordagem **heurística (pose + zonas + regras temporais)**, calibrada em campo — **não** um classificador de ação treinado.
- **Falsos positivos** são reduzidos por calibração, mas não zerados. Meta: nível confortável e útil para a operação.
- **Hardware mínimo recomendado:** CPU quad-core recente + 8 GB RAM. GPU acelera, mas não é obrigatória (usar OpenVINO/ONNX em CPU). Em PCs fracos, ajustar FPS/resolução/nº de câmeras.

---

## 3. Arquitetura & Stack

**Linguagem:** Python 3.11+

| Camada | Tecnologia | Papel |
|---|---|---|
| Captura de vídeo | OpenCV + FFmpeg | Ler streams RTSP (IP/DVR), amostragem de quadros |
| Detecção de pessoa | Ultralytics **YOLO** (yolov8n/yolo11n) | "Portão": só processa quando há pessoa na zona |
| Pose / mãos | **YOLO-pose** (yolov8n-pose) | Keypoints (punhos, ombros, quadril) para rastrear as mãos |
| Rastreamento | **ByteTrack** (nativo no Ultralytics) | Manter identidade da pessoa entre frames |
| Detecção de objeto (opcional) | YOLO (classes COCO: backpack, handbag) | Localizar bolsa/mochila próxima |
| Inferência acelerada | **ONNX Runtime** / **OpenVINO** | Rodar rápido em CPU (comum em PDV Intel) |
| Alertas | Telegram Bot API (`requests` ou `python-telegram-bot`) | Enviar imagem + dados |
| Persistência | SQLite (eventos) + JSON/YAML (config) | Local |
| Interface | **PySide6** ou **CustomTkinter** | Config de câmeras/zonas, status, log de eventos |
| Empacotamento | **PyInstaller** + **Inno Setup** | Instalador Windows |

---

## 4. Pipeline de Processamento (o coração)

Para cada câmera (thread de captura própria), o fluxo é **escalonado** para poupar hardware:

```
RTSP (câmera N)
   → amostragem de quadros (ex.: 4-6 FPS efetivos, configurável)
   → [GATE] YOLO detect: há PESSOA dentro da zona monitorada?
        não → descarta (modo econômico)
        sim ↓
   → YOLO-pose: keypoints da(s) pessoa(s) + ByteTrack (id)
   → CONCEALMENT LOGIC:
        - define zonas de ocultação relativas ao corpo:
            * bolso/cintura  → região dos quadris
            * sob a roupa     → região do tórax/abdômen
            * bolsa/mochila   → objeto detectado próximo à pessoa
        - rastreia trajetória do PUNHO ao longo de uma janela de frames
        - dispara quando: punho entra em zona de ocultação + permanência (dwell)
          + (opcional) padrão "mão fechada/segurando"
        - score de confiança = f(pose_conf, dwell, zona, trajetória)
   → se score ≥ limiar (configurável) E cooldown ok:
        → captura frame anotado + timestamp
        → salva evidência (imagem + registro SQLite)
        → envia alerta Telegram (imagem + hora + câmera)
```

### Concorrência para hardware fraco (crítico)
Não rodar pose em 5 streams ao mesmo tempo. Usar:
- **1 thread de captura por câmera** → alimenta uma **fila limitada** (bounded queue).
- **Pool pequeno de workers de inferência (1–2)** que consome a fila e prioriza câmeras com pessoa presente.
- Downscale do frame para detecção (ex.: 640px), substream do DVR (menor resolução) quando disponível.
- Parâmetros `target_fps` e `resolution` **por câmera** no config, para calibrar no PC alvo.

### Nota de implementação (para o próximo dev/Claude)
Construir a versão **heurística pragmática** primeiro (pose + zonas + dwell). **Não** tentar treinar um modelo de ação nem buscar dataset — o cliente pediu explicitamente "sem treino por loja". O ganho vem da **calibração dos limiares** com vídeo real.

---

## 5. Estrutura do Projeto

```
prevencao-perdas-cv/
├── README.md
├── requirements.txt
├── config/
│   └── config.example.json        # câmeras, zonas, limiares, telegram
├── models/                        # pesos YOLO (.pt / .onnx / openvino_model/)
├── src/
│   ├── main.py                    # entrypoint: carrega config, sobe threads, orquestra
│   ├── capture/
│   │   └── rtsp_capture.py        # captura RTSP + amostragem de quadros (1 thread/câmera)
│   ├── inference/
│   │   └── engine.py              # wrapper YOLO + ONNX/OpenVINO + pool de workers
│   ├── detection/
│   │   ├── person_gate.py         # detecção de pessoa dentro da zona (gate)
│   │   ├── pose_estimator.py      # keypoints (punhos, ombros, quadril)
│   │   ├── tracker.py             # ByteTrack (ids)
│   │   └── concealment.py         # zonas de ocultação + regras temporais + score
│   ├── alerts/
│   │   └── telegram_alert.py      # envio de imagem + caption (hora, câmera)
│   ├── evidence/
│   │   └── recorder.py            # salva frame anotado + registro (SQLite)
│   ├── config/
│   │   └── settings.py            # carregar/validar config (pydantic)
│   └── ui/
│       ├── app.py                 # painel: câmeras, status, eventos
│       └── zone_editor.py         # desenhar a área monitorada sobre um snapshot
├── evidence/                      # imagens dos eventos (gerado)
├── logs/
├── scripts/
│   └── build_installer.py         # PyInstaller
└── installer/
    └── setup.iss                  # Inno Setup (instalador final)
```

---

## 6. Configuração (exemplo)

`config/config.example.json`:
```json
{
  "telegram": {
    "bot_token": "COLOCAR_TOKEN_DO_BOTFATHER",
    "chat_id": "COLOCAR_ID_DO_GRUPO"
  },
  "inference": {
    "device": "openvino",          // "cpu" | "onnx" | "openvino"
    "person_model": "models/yolov8n.pt",
    "pose_model": "models/yolov8n-pose.pt",
    "detect_size": 640
  },
  "detection": {
    "confidence_threshold": 0.6,   // limiar do alerta (calibrável)
    "dwell_seconds": 1.2,          // tempo da mão na zona p/ disparar
    "cooldown_seconds": 30         // evita alertas repetidos da mesma pessoa
  },
  "cameras": [
    {
      "name": "Caixa 01",
      "rtsp_url": "rtsp://user:senha@192.168.0.10:554/cam/realmonitor?channel=3&subtype=1",
      "target_fps": 5,
      "zones": [ [[0.2,0.3],[0.8,0.3],[0.8,0.9],[0.2,0.9]] ]  // polígono normalizado (0-1)
    }
  ]
}
```

### Formatos de URL RTSP por marca (referência)
- **Intelbras / Dahua:** `rtsp://user:senha@IP:554/cam/realmonitor?channel=N&subtype=1`
  (`subtype=1` = substream, menor resolução → melhor desempenho)
- **Hikvision:** `rtsp://user:senha@IP:554/Streaming/Channels/N02`
  (`N` = canal; final `01` = principal, `02` = substream)
- No **DVR**, cada câmera cabeada é um **canal (N)** → escolher os 5 canais desejados.

---

## 7. Integração com Telegram

1. Criar o bot pelo **@BotFather** (`/newbot`) → obter o **token**.
2. Criar um **grupo** e adicionar o bot; obter o **chat_id** (via `getUpdates` da API ou bot auxiliar).
3. Enviar alerta: `POST https://api.telegram.org/bot<token>/sendPhoto` com a imagem + `caption` (data/hora, câmera, tipo de evento).
4. Enviar de forma **assíncrona/em fila** para não travar o pipeline.

---

## 8. Otimização de Desempenho (checklist)

- [ ] Processamento **gated** por presença de pessoa na zona (pose só quando necessário).
- [ ] **Amostragem de quadros** (`target_fps` por câmera).
- [ ] **Substream** do DVR (resolução menor) para detecção.
- [ ] **Downscale** do frame antes da inferência.
- [ ] Exportar modelos para **OpenVINO/ONNX** (aceleração em CPU Intel).
- [ ] **Pool limitado** de workers de inferência (não rodar 5 poses simultâneas).
- [ ] Modelos **nano** (yolov8n / yolov8n-pose).
- [ ] Parâmetros de FPS/resolução ajustáveis **por câmera** para calibrar no PC alvo.

---

## 9. Cronograma (5 semanas — entregas modulares)

| Semana | Foco | Entregável |
|---|---|---|
| 1 | **Captura** | App base + captura RTSP multi-câmera (IP/DVR) + amostragem + shell da UI |
| 2 | **Detecção & zonas** | Gate de pessoa + editor de zonas + ByteTrack + engine ONNX/OpenVINO |
| 3 | **Comportamento** ★ | Pose + heurística de ocultação + score + limiar (★ marco 50%) |
| 4 | **Alertas** | Captura de evidência (imagem+timestamp) + integração Telegram + log de eventos |
| 5 | **Entrega** | Tuning de desempenho + instalador Windows + manual + calibração em campo |

★ **Marco de liberação 50%** ao final da Semana 3 (captura + detecção de comportamento funcionando).

---

## 10. Entregáveis

- Aplicação Windows funcional e testada (+ instalador).
- Código-fonte completo e documentado (**pertence ao cliente** — ele evoluirá o projeto).
- Integração completa com o Telegram (imagem + data/hora).
- Manual de instalação e configuração (câmeras + áreas monitoradas).

---

## 11. Pré-requisitos para iniciar (fornecidos pelo cliente)

- **Acesso RTSP do DVR/câmeras:** marca/modelo, IP, usuário, senha e os **canais** das 5 câmeras.
- **Telegram:** token do bot (BotFather) + grupo/chat de destino.
- **Acesso remoto ao PC alvo** (AnyDesk/TeamViewer) para instalar, testar e calibrar no ambiente real.
- **Vídeos/gravações reais** (ou acesso ao vivo) para calibração da detecção.

---

## 12. Setup de Desenvolvimento

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Baixar modelos pré-treinados (Ultralytics baixa automático no 1º uso)
#    yolov8n.pt e yolov8n-pose.pt

# 3. Configurar
copy config\config.example.json config\config.json
#    editar câmeras (RTSP), telegram (token/chat) e limiares

# 4. Rodar
python src/main.py
```

`requirements.txt` (base):
```
ultralytics
opencv-python
onnxruntime
openvino
numpy
pydantic
python-telegram-bot   # ou requests
PySide6               # ou customtkinter
```

---

## 13. Como começar no Claude Code (ordem sugerida)

1. **Scaffold** da estrutura de pastas (seção 5) + `requirements.txt` + `config.example.json`.
2. **Captura RTSP** (`rtsp_capture.py`): abrir 1 stream, amostrar quadros, exibir — validar com uma câmera real antes de escalar.
3. **Engine de inferência** (`engine.py`): carregar YOLO, gate de pessoa, exportar/rodar em OpenVINO.
4. **Editor de zonas** (`zone_editor.py`) + persistência no config.
5. **Pose + concealment** (`concealment.py`): implementar a heurística (zonas relativas ao corpo + dwell do punho + score). Testar com vídeo real e **calibrar limiares**.
6. **Evidência + Telegram** (`recorder.py`, `telegram_alert.py`).
7. **Multi-câmera + pool de workers** e tuning de desempenho.
8. **Empacotamento** (PyInstaller + Inno Setup) + **manual**.

> **Regra de ouro:** validar cada etapa com **stream/vídeo real** antes de avançar. O risco do projeto não está em escrever o código, e sim na **calibração da detecção** — reserve tempo para isso (Semanas 3 e 5).

---

*Projeto MVP — a acurácia é calibrada em ambiente real. Execução 100% local, sem nuvem. Código-fonte entregue ao cliente.*
