# Modelo de Revisão — configuração por loja

Como configurar cada instalação para o fluxo **"IA aponta → revendedor confere → encaminha pro mercado"**.

## O fluxo

```
PC de cada loja  ──alerta etiquetado──►  GRUPO DE REVISÃO (revendedor + sócio)
   (roda local)      (🏪 Mercado Julie #1)      │
                                                confere o vídeo de 3s
                                                │  é furto?  → encaminha pro grupo do mercado
                                                └  não é?    → ignora
```

Cada loja roda o sistema **local** e manda o alerta pro **mesmo grupo de revisão**. O alerta chega **identificado com o nome e o número da loja**, então o revendedor sabe pra qual mercado encaminhar.

> O encaminhamento com um clique (botão "foi furto" → vai sozinho pro grupo certo) é a **fase 2**, que precisa do servidor central. Por enquanto o encaminhamento é manual — funciona bem para os primeiros mercados.

## Configuração de cada loja (aba Configuração / config.json)

| Campo | Valor no modelo de revisão | Por quê |
|---|---|---|
| `store.name` | Nome do mercado (ex.: `Mercado Julie`) | Aparece no alerta |
| `store.numero` | Número do cliente (ex.: `1`, `2`, `3`) | Vira `Mercado Julie #1` no alerta — pra saber pra qual grupo encaminhar |
| `telegram.chat_id` | **O grupo de REVISÃO** (o seu, não o do mercado) | Todos os alertas caem no seu grupo primeiro |
| `telegram.enabled` | `true` | Precisa enviar pro grupo de revisão |
| `telegram.send_clip` | `true` | O alerta vai como **vídeo** (pra ver o gesto) |
| `telegram.send_photo` | `false` (opcional) | Só o vídeo, mais limpo |

### Enxugar os alertas (crítico — senão inunda o grupo de revisão)

| Campo | Valor sugerido | Efeito |
|---|---|---|
| **Zonas** | Só a **faixa da prateleira mais furtada** — nunca o corredor de passagem | É o que MAIS corta falso |
| `detection.threshold` | `0.85` (mais rigoroso que o padrão 0.75) | Só os gestos mais fortes disparam |
| `telegram.min_seconds_between_alerts` | `90` (ou mais) | Trava anti-enxurrada por câmera |

### Duração do vídeo do alerta

Padrão: `clip_pre_seconds: 2.0` + `clip_post_seconds: 4.0` = **6 segundos** (mostra antes e depois do gesto — melhor pra julgar furto vs devolução).

Se preferir **3 segundos** exatos: `clip_pre_seconds: 1.5`, `clip_post_seconds: 1.5`.

## Passo a passo pra achar o chat_id do grupo de revisão

1. Cria o grupo de revisão no Telegram (você + sócio) e adiciona o bot.
2. Manda `/start` no grupo.
3. Na aba Configuração de qualquer instalação, cola o token e clica **Procurar meu grupo** → seleciona o grupo de revisão → **Salvar**.
4. Usa esse mesmo grupo de revisão em **todas** as lojas (só muda `store.name` e `store.numero` em cada uma).

## Expectativa honesta (pra alinhar com o cliente)

- O sistema **aponta**; **quem confirma é o humano** (você/sócio) vendo o vídeo. É "IA + revisão humana".
- A revisão **diminui** com o tempo (aprende os falsos típicos de cada loja), mas **não some** — a câmera de teto tem o limite de não distinguir furto de movimento normal sozinha.
- Escala bem para **poucos mercados**; para dezenas/centenas, a revisão vira operação com equipe (ou entra o servidor central + faixas de score da fase 2).
