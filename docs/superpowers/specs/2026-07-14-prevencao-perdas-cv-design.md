# Design — Prevenção de Perdas com Visão Computacional (MVP)

**Data:** 2026-07-14
**Cliente:** Adriano (revendedor — mercados e farmácias)
**Executor:** Eduardo
**Prazo:** 5 semanas · **Marco de 50%:** fim da semana 3

---

## 1. Objetivo

Software 100% local para Windows que lê câmeras existentes (IP e DVR, via RTSP), detecta em tempo real o comportamento de **ocultação de produto** (mão levando item ao bolso, bolsa/mochila, sob a roupa ou na cintura), captura evidência e alerta a equipe no Telegram.

Não é gravador nem sistema forense. É uma **camada de alerta**.

## 2. Restrição central que molda todo o design

O material do ambiente real (vídeos, ângulos de câmera, acessos ao DVR) **ainda não chegou** e chegará em paralelo ao desenvolvimento. O sistema, portanto, deve ser construído de modo que a chegada desse material exija **apenas configuração — nunca código**.

Consequências assumidas em todo o documento:

- Todo parâmetro de detecção vive no JSON de configuração, com override por câmera.
- A heurística de ocultação é testável com **keypoints sintéticos**, sem vídeo.
- O caminho RTSP é validado contra um **DVR simulado local** (MediaMTX servindo vídeos em loop), incluindo quedas e reconexão.
- Existe um **modo replay** (rodar o pipeline sobre arquivo de vídeo como se fosse câmera ao vivo) e um **sweep de calibração** que mede acerto e falso positivo sobre uma pasta de vídeos rotulados.

## 3. Escopo

### Dentro

| Item | Origem |
|---|---|
| Captura RTSP multi-câmera com reconexão automática | R2, R3 |
| Gate de pessoa (processamento pesado só quando há gente na zona) | R4 |
| Pose + tracking + heurística de ocultação | R5, R8 |
| Editor visual de zonas por câmera | R7 |
| Evidência: imagem anotada + clipe curto + registro SQLite | R9 |
| Alerta Telegram (fila assíncrona, retry, rate-limit) | R10 |
| Watchdog de câmera offline | acordado com o cliente |
| Purga automática de evidências antigas | acordado com o cliente |
| Teste de capacidade do PC ("aguenta quantas câmeras?") | acordado com o cliente |
| Modo replay + sweep de calibração | necessidade do executor |
| UI PySide6 (câmeras ao vivo, zonas, eventos, feedback) | R7 |
| Instalador Windows (PyInstaller + Inno Setup) | R6 |
| Código-fonte documentado + manual | entregável |

### Fora

- Treino de modelo por loja ou dataset próprio (explicitamente vedado — R5).
- Classificador de ação treinado (o MVP é heurístico).
- Detecção de troca de embalagem (fora do alcance da abordagem — comunicado ao cliente).
- Gravação contínua de vídeo, revisão forense, painel central multi-loja.
- Reconhecimento facial ou identificação de pessoas.

O número de câmeras **não é travado em 5**: o config aceita N câmeras. As 5 são o compromisso de desempenho e calibração; acima disso, o limite é o hardware — medido pelo teste de capacidade.

## 4. Arquitetura

Processo único, multi-thread. Fluxo de dados unidirecional, sem estado compartilhado mutável entre threads (comunicação só por filas e sinais Qt).

```
[CameraThread × N]                    (1 thread por câmera)
   RTSP → decode → amostragem (target_fps) → LatestFrameSlot (tamanho 1, descarta o antigo)
        └─ reconexão automática com backoff exponencial
        └─ publica heartbeat (timestamp do último frame)

[Scheduler]                           (round-robin ponderado)
   escolhe qual câmera o próximo worker atende;
   peso alto para câmeras com pessoa detectada nos últimos K segundos

[InferenceWorker × 1..2]              (pool configurável)
   frame → PersonGate (YOLO detect)
             ├─ nenhuma pessoa dentro da zona → descarta (modo econômico)
             └─ há pessoa ↓
           PoseEstimator (YOLO-pose no RECORTE de cada pessoa)
           Tracker (associação IoU + idade → track_id estável)
           ConcealmentAnalyzer (estado por track_id) → score
             └─ score ≥ limiar & cooldown ok → EventBus.publish(ConcealmentEvent)

[EventBus] (fan-out, não bloqueante)
   ├─ EvidenceRecorder → JPEG anotado + clipe MP4 (buffer circular) + linha no SQLite
   ├─ AlertQueue (thread própria) → Telegram (retry, backoff, rate-limit)
   └─ UI (sinal Qt) → atualiza lista de eventos ao vivo

[Watchdog] (thread própria)
   sem heartbeat de uma câmera por > offline_after_seconds
     → marca offline, alerta no Telegram, alerta na UI; avisa também na recuperação
```

**Por que fila de tamanho 1 (`LatestFrameSlot`):** em vigilância ao vivo, frame velho é lixo. Uma fila com buffer acumularia atraso crescente quando a inferência não acompanha; descartar o frame antigo mantém a latência limitada e degrada suavemente sob carga.

**Por que o gate antes da pose:** pose é ~5–10× mais cara que detect. Corredor vazio custa apenas um detect por amostra. É o que permite 8–10 câmeras num PC modesto de mercado.

**Por que pose no recorte da pessoa:** roda em resolução efetiva muito maior sobre o corpo (crucial para câmera de teto, onde a pessoa ocupa poucos pixels) e é mais barato que rodar pose no frame inteiro.

## 5. Módulos

```
prevencao-perdas-cv/
├── config/config.example.json
├── models/                          # pesos .pt e caches OpenVINO (gerados)
├── src/
│   ├── main.py                      # entrypoint: config → threads → UI
│   ├── config/settings.py           # modelos pydantic + carga/validação/migração
│   ├── capture/
│   │   ├── rtsp_capture.py          # CameraThread: RTSP, amostragem, reconexão, heartbeat
│   │   └── frame_slot.py            # LatestFrameSlot (thread-safe, tamanho 1)
│   ├── inference/
│   │   ├── engine.py                # carga de modelos, export OpenVINO, warmup, predict
│   │   ├── scheduler.py             # round-robin ponderado por atividade
│   │   └── worker_pool.py           # pool de workers de inferência
│   ├── detection/
│   │   ├── person_gate.py           # pessoa dentro do polígono da zona?
│   │   ├── pose_estimator.py        # keypoints COCO-17 no recorte da pessoa
│   │   ├── tracker.py               # associação IoU + idade → track_id
│   │   ├── body_frame.py            # sistema de coordenadas do corpo + zonas de ocultação
│   │   └── concealment.py           # sinais + score + máquina de estados por track
│   ├── evidence/
│   │   ├── clip_buffer.py           # buffer circular de frames por câmera
│   │   ├── recorder.py              # JPEG anotado + MP4 + registro
│   │   └── retention.py             # purga por idade
│   ├── alerts/
│   │   ├── alert_queue.py           # fila + retry + backoff + rate-limit
│   │   └── telegram_alert.py        # sendPhoto / sendVideo
│   ├── storage/db.py                # SQLite (schema, migrações, consultas)
│   ├── watchdog/monitor.py          # câmera offline / recuperada
│   ├── ui/
│   │   ├── app.py                   # janela principal (abas)
│   │   ├── live_view.py             # grade de câmeras + status
│   │   ├── zone_editor.py           # desenhar/arrastar polígono sobre snapshot
│   │   └── event_log.py             # histórico + marcar falso positivo
│   └── tools/
│       ├── replay.py                # pipeline sobre arquivo de vídeo
│       ├── calibrate.py             # sweep de parâmetros sobre vídeos rotulados
│       └── benchmark.py             # teste de capacidade do PC
├── tests/                           # pytest (unit + integração)
├── dev/
│   ├── mediamtx/                    # DVR simulado (config + script)
│   └── fixtures/                    # sequências de keypoints sintéticos
├── scripts/build_installer.py
└── installer/setup.iss
```

Regra de fronteira: `detection/` não conhece RTSP, Telegram nem SQLite. Recebe frame + config, devolve evento. É o núcleo puro e testável.

## 6. Detecção de ocultação (núcleo do sistema)

### 6.1 Sistema de coordenadas do corpo

Keypoints COCO-17. Para cada pessoa rastreada:

- `shoulder_mid = (kp_ombro_esq + kp_ombro_dir) / 2`
- `hip_mid = (kp_quadril_esq + kp_quadril_dir) / 2`
- **Escala do corpo** `S = ||shoulder_mid − hip_mid||` (comprimento do tronco em pixels).
  Fallback quando os quadris têm confiança baixa: `S = 0.55 × altura_da_bbox`.
- **Eixo vertical** `û = (shoulder_mid − hip_mid) / S` — "para cima" no referencial do corpo.
- **Eixo horizontal** `v̂ = perpendicular(û)`.

A posição de um punho `w` é convertida para coordenadas normalizadas do corpo:

```
x_n = ((w − hip_mid) · v̂) / S      # lateral: 0 = eixo do corpo
y_n = ((w − hip_mid) · û) / S      # vertical: 0 = linha do quadril, 1 = linha do ombro
```

Isso resolve, **sem calibração por câmera**, os três problemas que quebram heurísticas ingênuas: pessoa perto vs. longe (a escala `S` normaliza), pessoa inclinada (os eixos acompanham o corpo) e câmeras com alturas diferentes.

### 6.2 Zonas de ocultação (em coordenadas do corpo)

| Zona | Condição | Peso |
|---|---|---|
| `waist` (bolso/cintura) | `−0.45 ≤ y_n ≤ 0.25` e `0.10 ≤ \|x_n\| ≤ 0.85` | 1.00 |
| `torso` (sob a roupa) | `0.15 ≤ y_n ≤ 0.85` e `\|x_n\| ≤ 0.55` | 0.95 |
| `back_waist` (cintura traseira) | mesma caixa de `waist`, **e** pessoa de costas | 1.05 |
| `bag` (bolsa/mochila) | punho dentro da bbox (expandida 10%) de um `backpack`/`handbag` associado à pessoa | 1.00 |

- **Pessoa de costas:** confiança de nariz e dos dois olhos abaixo de `kp_conf_min`.
- **Associação da bolsa à pessoa:** centro do objeto a menos de `1.2 × S` do `shoulder_mid`, escolhendo o mais próximo.
- **Região `reach`** (não é zona de ocultação; alimenta o sinal de trajetória): `y_n > 0.9` ou `|x_n| > 0.95` — braço estendido, longe do corpo, típico de quem pega item na prateleira.

Todos esses limites geométricos são constantes nomeadas no config (bloco `detection.geometry`), calibráveis por câmera.

### 6.3 Sinais (janela deslizante de `window_seconds`, por punho, por track)

| Sinal | Definição | Peso padrão |
|---|---|---|
| `s_dwell` | `min(1, frames_consecutivos_na_zona / (dwell_seconds × fps_efetivo))`, tolerando até `gap_frames` frames fora | 0.40 |
| `s_approach` | 1.0 se o punho esteve na região `reach` dentro da janela **antes** de entrar na zona; decai linearmente até 0 ao longo de `window_seconds` | 0.20 |
| `s_vanish` | 1.0 enquanto o punho tem confiança `< kp_conf_min` **e** sua última posição conhecida (há no máximo `vanish_grace_seconds`) estava dentro de uma zona de ocultação (ou a menos de `0.2 × S` da borda). Expira após `vanish_max_seconds` | 0.30 |
| `s_retract` | 1.0 se, após pelo menos `0.5 × dwell_seconds` dentro da zona, o punho reaparece e sobe (`Δy_n > 0.3` em ≤ 1 s) | 0.10 |

**`s_vanish` é a decisão de design mais importante do projeto.** Quando alguém enfia a mão no bolso, sob a blusa ou dentro da bolsa, o punho **desaparece dos keypoints**. A implementação ingênua descarta esses frames como ruído — e perde exatamente o momento do furto. Aqui, punho que some *tendo desaparecido dentro da zona do corpo* é tratado como **evidência positiva**, não como dado faltante. Sem esse sinal, os casos "sob a roupa" e "dentro da mochila" seriam quase invisíveis.

### 6.4 Score

```
qualidade = média das confianças de (ombros, quadris)
se qualidade < pose_quality_min          → pessoa ignorada (não avalia)
se altura_bbox < min_person_px           → pessoa ignorada (pose não confiável)

bruto  = w_dwell·s_dwell + w_approach·s_approach + w_vanish·s_vanish + w_retract·s_retract
score  = clamp(0, 1, bruto × peso_da_zona × qualidade)
```

Dispara evento quando `score ≥ threshold` **e** o dwell mínimo foi atingido **e** o cooldown do track está livre. Os pesos, o limiar e todos os tempos são configuráveis — globalmente e por câmera.

**Pesos des-normalizados (dois caminhos de disparo).** Os pesos **não somam 1.0** de propósito (o `clamp` cuida do teto): isso dá dois caminhos independentes de disparo, correspondentes às duas "assinaturas" físicas da ocultação. (1) **Caminho dwell** — mão que fica MUITO tempo na zona (`s_dwell` saturado) já dispara sozinha, mesmo com o punho visível o tempo todo (importante para câmera lateral onde a mão não some). (2) **Caminho vanish** — mão que SOME dentro da zona dispara mesmo com dwell parcial. Se os pesos somassem 1.0, nenhum sinal isolado cruzaria o limiar e o sistema ficaria dependente do `vanish` — um furto com a mão visível passaria. Descoberto testando a Task 3 do Plano 2.

### 6.5 Máquina de estados (por `track_id`)

```
IDLE ──punho entra em reach──► APPROACHING
IDLE ──punho entra em zona──► CONCEALING          (sem s_approach; score menor)
APPROACHING ──punho entra em zona / some na zona──► CONCEALING
APPROACHING ──timeout (window_seconds)──► IDLE
CONCEALING ──score ≥ threshold & dwell ok──► ALERT
CONCEALING ──punho sai da zona antes do dwell──► IDLE
ALERT ──emite evento──► COOLDOWN
COOLDOWN ──cooldown_seconds──► IDLE
qualquer estado ──track perdido por track_lost_seconds──► descarta o estado
```

### 6.6 Guardas contra falso positivo

- Pessoa fora do polígono monitorado → nem entra no pipeline (é o gate).
- `pose_quality_min` e `min_person_px` descartam poses não confiáveis.
- `s_approach` evita alertar por mão que **já estava** no bolso quando a pessoa entrou em cena.
- Cooldown por track e intervalo mínimo entre alertas da mesma câmera.
- Colocar produto no carrinho/cesta **não** é alvo: fica geometricamente fora das zonas (mão vai para longe e para baixo do corpo, região `reach`, não para a cintura).

### 6.7 Valores padrão (conservadores até a calibração real)

```
threshold 0.60 · dwell_seconds 1.2 · window_seconds 3.0 · cooldown_seconds 30
kp_conf_min 0.35 · pose_quality_min 0.40 · min_person_px 120
vanish_grace_seconds 0.4 · vanish_max_seconds 3.0 · gap_frames 2 · track_lost_seconds 2.0
pesos (des-normalizados): dwell 0.70 · approach 0.25 · vanish 0.55 · retract 0.15
```

Estes números são o **ponto de partida**, não a entrega. A calibração com o vídeo real do cliente é que define os finais, e o sweep (§8) é a ferramenta que os encontra.

## 7. Configuração

`config/config.json`, validado com pydantic. Erro de validação mostra mensagem clara na UI, não stacktrace.

```json
{
  "store": { "id": "mercado-piloto", "name": "Mercado Piloto" },
  "telegram": {
    "bot_token": "", "chat_id": "",
    "send_photo": true, "send_clip": true,
    "rate_limit_per_min": 15
  },
  "inference": {
    "device": "openvino",
    "person_model": "models/yolo11n.pt",
    "pose_model": "models/yolo11n-pose.pt",
    "detect_size": 640,
    "pose_on_crop": true,
    "workers": 2,
    "detect_bags": true
  },
  "detection": {
    "threshold": 0.60,
    "dwell_seconds": 1.2,
    "window_seconds": 3.0,
    "cooldown_seconds": 30,
    "weights": { "dwell": 0.70, "approach": 0.25, "vanish": 0.55, "retract": 0.15 },
    "zone_weights": { "waist": 1.0, "torso": 0.95, "back_waist": 1.05, "bag": 1.0 },
    "geometry": { "waist_y": [-0.45, 0.25], "waist_x": [0.10, 0.85],
                  "torso_y": [0.15, 0.85], "torso_x_max": 0.55,
                  "reach_y_min": 0.9, "reach_x_min": 0.95 },
    "guards": { "kp_conf_min": 0.35, "pose_quality_min": 0.40, "min_person_px": 120,
                "vanish_grace_seconds": 0.4, "vanish_max_seconds": 3.0,
                "gap_frames": 2, "track_lost_seconds": 2.0 }
  },
  "evidence": {
    "dir": "evidence", "retention_days": 30,
    "clip_pre_seconds": 2.0, "clip_post_seconds": 4.0
  },
  "watchdog": { "offline_after_seconds": 30, "notify": true },
  "cameras": [
    {
      "name": "Corredor Bebidas",
      "rtsp_url": "rtsp://user:senha@192.168.0.10:554/cam/realmonitor?channel=7&subtype=1",
      "enabled": true,
      "target_fps": 5,
      "zones": [ [[0.2,0.3],[0.8,0.3],[0.8,0.9],[0.2,0.9]] ],
      "overrides": { "threshold": 0.65 }
    }
  ]
}
```

`overrides` aceita qualquer chave de `detection` — porque uma câmera de teto e uma câmera lateral **não** compartilham o mesmo limiar, e essa é a diferença entre calibrar e sofrer.

Zonas em coordenadas normalizadas (0–1) sobre o quadro: resistem a mudança de resolução ou substream.

## 8. Modo replay e sweep de calibração

**Replay** (`tools/replay.py`): roda o pipeline completo sobre arquivo de vídeo, tratando-o como câmera. Produz vídeo anotado (esqueleto, zonas, score em tempo real) e CSV com score por frame. É como se validam as detecções sem loja.

**Sweep** (`tools/calibrate.py`): recebe uma pasta com clipes rotulados —

```
videos/
├── ocultacao/     # clipes onde DEVE alertar
└── normal/        # vídeo de movimento comum onde NÃO deve alertar
```

— varre combinações de `threshold`, `dwell_seconds` e pesos, e produz uma tabela:

| threshold | dwell | detectados (de N) | falsos por hora |
|---|---|---|---|

Essa tabela é a resposta objetiva à pergunta que o Adriano vai responder ("quantos alertas falsos por dia a equipe aguenta?"): escolhemos a linha que respeita o teto dele e maximiza a detecção. **Calibração vira medição, não achismo.**

A coluna `feedback` da tabela de eventos (§9) realimenta o sweep: os falsos positivos marcados pela operação viram casos de teste do conjunto `normal`.

## 9. Persistência

SQLite (`data/app.db`), WAL ligado.

```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  store_id TEXT NOT NULL,           -- preparado para painel multi-loja futuro
  camera_name TEXT NOT NULL,
  ts_utc TEXT NOT NULL,
  ts_local TEXT NOT NULL,
  track_id INTEGER,
  score REAL NOT NULL,
  zone TEXT NOT NULL,               -- waist | torso | back_waist | bag
  signals_json TEXT NOT NULL,       -- valores de cada sinal: auditoria e calibração
  image_path TEXT,
  clip_path TEXT,
  sent_telegram INTEGER DEFAULT 0,
  feedback TEXT                     -- NULL | 'true_positive' | 'false_positive'
);
CREATE INDEX idx_events_ts ON events(ts_utc);

CREATE TABLE camera_status (
  camera_name TEXT PRIMARY KEY,
  state TEXT NOT NULL,              -- online | offline | reconnecting
  last_frame_ts TEXT,
  since TEXT NOT NULL
);

CREATE TABLE app_meta (key TEXT PRIMARY KEY, value TEXT);
```

`signals_json` guarda a decomposição do score. Sem isso, um falso positivo em campo é indepurável.

`store_id` custa zero hoje e evita reescrever o schema no dia em que o Adriano quiser ver todas as lojas num painel.

## 10. Alertas

Fila em thread própria — Telegram lento ou fora do ar **nunca** trava a inferência.

- `sendPhoto` com imagem anotada + legenda: loja, câmera, hora local, tipo de zona, score.
- `sendVideo` com o clipe curto (2 s antes + 4 s depois, do buffer circular). Desligável em PC fraco.
- Retry com backoff exponencial (3 tentativas), rate-limit local (`rate_limit_per_min`, padrão 15 — abaixo do teto do Telegram).
- Evento persiste no banco mesmo se o envio falhar; `sent_telegram = 0` permite reenvio.
- Alertas de sistema (câmera offline / recuperada) usam a mesma fila, com prefixo distinto.

## 11. Resiliência

- **Reconexão RTSP:** backoff exponencial (1 s → 2 s → 4 s … teto de 30 s), infinito. Testado derrubando o MediaMTX no meio do teste de integração.
- **Watchdog:** sem frame por mais de `offline_after_seconds` → estado `offline`, alerta no Telegram e badge vermelho na UI; alerta de recuperação quando voltar.
- **Degradação sob carga:** se os workers não acompanham, o `LatestFrameSlot` descarta frames — o FPS efetivo cai, a latência não cresce. A UI mostra o FPS real por câmera, para calibrar `target_fps` em campo.
- **Disco:** purga diária por idade (`retention_days`); se o disco passar de 90% de uso, purga agressiva e avisa.

## 12. Interface (PySide6)

Três abas:

1. **Ao vivo** — grade de câmeras com FPS real, estado (online/offline/reconectando) e overlay das zonas. Botão liga/desliga por câmera.
2. **Câmeras & Zonas** — assistente: colar URL RTSP (com montador por marca: Intelbras/Dahua, Hikvision — preenche o padrão a partir de IP, usuário, senha e canal), testar conexão, tirar snapshot, **desenhar o polígono** (clicar para adicionar vértices, arrastar para ajustar), salvar. Padrão "monitorar o quadro inteiro" em um clique — foi o que o cliente pediu para reduzir trabalho por loja.
3. **Eventos** — tabela com miniatura, hora, câmera, score; abrir a imagem/clipe; **marcar como falso positivo** (alimenta o sweep).

Mais uma aba de **Configuração** (Telegram, limiares, retenção) e o botão do **Teste de Capacidade**.

## 13. Teste de capacidade

`tools/benchmark.py`, exposto por botão na UI. Roda o pipeline real contra streams sintéticos com pessoas em movimento, com carga crescente, e mede FPS sustentado e uso de CPU. Saída: relatório legível —

> *"Este PC sustenta 8 câmeras a 5 FPS com detecção, ou 5 câmeras a 8 FPS. Acima disso, o FPS cai abaixo do recomendado."*

É a ferramenta de venda que o Adriano roda em cada cliente antes de fechar. Também informa honestamente que o limite real depende de **quantas pessoas aparecem ao mesmo tempo** (câmera vazia custa pouco), então mede dois cenários: loja calma e loja movimentada.

## 14. Empacotamento

PyInstaller (modo `onedir`, mais rápido para abrir e para atualizar) + Inno Setup. Modelos `.pt` e o cache OpenVINO embarcados — sem download na primeira execução (loja pode não ter internet liberada). Opção de iniciar com o Windows. Manual de instalação e calibração em PDF/Markdown.

## 15. Testes

| Nível | O quê | Como (sem câmera real) |
|---|---|---|
| Unit | `body_frame`, `concealment`, sinais, score, FSM | **Sequências de keypoints sintéticos** — mão descendo ao quadril e ficando (deve alertar); mão coçando a barriga rapidamente (não deve); punho sumindo dentro da zona (deve); pessoa de costas (deve, com peso maior); pessoa pequena demais (ignorada) |
| Unit | config (pydantic), score, retenção, rate-limit | pytest puro |
| Unit | Telegram | mock do endpoint; testa retry, backoff, rate-limit, falha persistindo no banco |
| Integração | captura RTSP, reconexão, watchdog | **MediaMTX local** servindo vídeos em loop; derruba-se o servidor no meio do teste |
| Integração | pipeline ponta a ponta | replay de vídeo → evento → SQLite → Telegram mockado |
| Sistema | desempenho | benchmark com carga crescente |

A tabela acima é o motivo pelo qual é possível entregar o sistema **completo e testado** antes de o material do cliente chegar: a lógica é validada por fixtures, não por pixels. O vídeo real ajusta *números*, não descobre *bugs*.

## 16. Fases de entrega

| Fase | Conteúdo | Depende do cliente? |
|---|---|---|
| **F0** | Scaffold, config pydantic, SQLite, DVR simulado (MediaMTX), CI de testes | Não |
| **F1** | Captura RTSP multi-câmera, reconexão, heartbeat, UI ao vivo | Não |
| **F2** | Engine (YOLO + OpenVINO), gate de pessoa, pose no recorte, tracking, benchmark | Não |
| **F3** ★ | `body_frame` + `concealment` (TDD com keypoints sintéticos), replay, sweep | Não |
| **F4** | Evidência (imagem + clipe + SQLite), Telegram, watchdog, retenção | Só token/chat_id |
| **F5** | Editor de zonas, abas de eventos/config, instalador, manual | Não |
| **F6** | **Calibração real** + ajuste dos limiares + validação em campo | **Sim** |

★ F3 é o marco de 50% acordado (detecção de comportamento funcionando).

Só a **F6** depende do material do Adriano. Todas as outras podem ser concluídas agora — e é exatamente esse o pedido.

## 17. Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| **Ângulo de câmera ruim** (teto vertical, olho-de-peixe, pessoa pequena) — a pose não sai confiável | Alto: a heurística inteira depende dos keypoints | Fotos das câmeras antes de escolher os 5 canais; `min_person_px` descarta pose ruim em vez de alertar errado; recomendar reposicionamento quando for o caso |
| **Material real chega tarde ou não chega** | Alto: calibração é o que faz o sistema funcionar | Defaults conservadores; sweep pronto para calibrar em horas; replay permite calibrar com qualquer vídeo, inclusive gravado pelo próprio Eduardo |
| **Falso positivo acima do tolerável** | Alto: equipe desliga o sistema | Sweep otimiza contra o teto que o cliente definir; feedback da UI realimenta a calibração; limiar por câmera |
| **DVR limita conexões RTSP simultâneas** | Médio: menos câmeras do que o PC aguentaria | Detectar na conexão e reportar claramente; usar substream; rodízio entre câmeras se necessário |
| **PC do cliente mais fraco que o previsto** | Médio | Benchmark antes de vender; `target_fps` e resolução por câmera; desligar clipe de vídeo |
| **Expectativa de "pegar 100% dos furtos"** | Alto (comercial) | Já comunicado por escrito ao cliente: sistema de regras calibradas, não garantia de acurácia |

## 18. Decisões fechadas

| Decisão | Escolha | Motivo |
|---|---|---|
| UI | **PySide6** | Editor de zonas (polígono arrastável) e grade multi-câmera saem naturais no QGraphicsView; LGPL serve para revenda |
| Validação sem câmera | **DVR simulado (MediaMTX)** | Único jeito de testar reconexão e multi-canal antes da loja |
| Telegram | **`requests` direto** | A API é `sendPhoto`/`sendVideo`; `python-telegram-bot` traz asyncio e peso de dependência sem ganho |
| Detecção | **YOLO11n / YOLO11n-pose** | Sucessor direto do YOLOv8n citado no README, mesmo custo, melhor acurácia |
| Aceleração | **OpenVINO em CPU**, com fallback ONNX/PyTorch | PDV de mercado costuma ser Intel sem GPU |
| Pose | **no recorte da pessoa** | Resolução efetiva muito maior em pessoa pequena (câmera de teto) e mais barato |
| Tracking | **associação IoU + idade**, não o ByteTrack do Ultralytics | O ByteTrack embutido só funciona via `model.track()`, que reprocessa o frame inteiro e não aceita detecções externas — o que quebraria o gate (nosso ganho de desempenho) e acoplaria `detection/` ao modelo. Com câmera fixa, 5 FPS e poucas pessoas, associação por IoU entrega a mesma estabilidade de `track_id` e é testável sem modelo |
| Nº de câmeras | **N configurável**, 5 como compromisso | Pedido do cliente; o limite real é medido pelo benchmark |
