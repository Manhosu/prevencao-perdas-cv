# Modo Caixa — Botão de Pânico

Alerta de **assalto** por gesto: quando alguém levanta **as duas mãos acima da cabeça por 2 segundos**, o sistema dispara um alerta de pânico no Telegram.

É o produto do **caixa** (pivô de 29/jul, depois do detector de furto se mostrar inviável na câmera de teto). Ao contrário do furto, "mãos pra cima" é um gesto **claro e sem ambiguidade** — a pose acerta de qualquer ângulo, com confiança.

## Como funciona

- Num assalto, o caixa levanta as mãos (forçado pelo ladrão, ou de propósito pra sinalizar). Isso é o gatilho.
- **As duas** mãos precisam estar acima da cabeça, por **2 segundos** (não é aceno nem pegar algo na prateleira alta — precisa ser deliberado).
- Dispara alerta com **foto + hora**, legenda de urgência: *"🚨 ALERTA DE PÂNICO — mãos levantadas no caixa"*.
- **Não passa pelo anti-enxurrada** — alerta de assalto nunca é segurado por rate-limit.

## Como ativar (por câmera do caixa)

No `config.json` da instalação (`%LOCALAPPDATA%\PrevencaoPerdas\config.json`), na seção `detection`:

```json
"detection": {
  "mode": "panico",
  "panic": {
    "hold_seconds": 2.0,
    "cooldown_seconds": 30.0
  }
}
```

- `mode: "panico"` liga o modo caixa (o padrão é `"ocultacao"`).
- `hold_seconds`: segundos com as mãos acima para disparar (2 é o recomendado).
- `cooldown_seconds`: silêncio após um disparo, pra não repetir o mesmo alerta.

> O sistema tolera BOM do Bloco de Notas — pode editar o arquivo no Notepad sem quebrar.

Depois de editar, **feche e abra o programa**.

## Câmera e zona

- **Aponte a câmera pro caixa** (não precisa ser de teto; qualquer ângulo bom serve — o gesto de mãos pra cima é claro de qualquer lugar).
- **Não precisa desenhar zona** — sem zona, o sistema vigia o quadro todo, que é o que se quer no caixa. (Se desenhar, ele só olha quem está na área marcada.)
- Telegram: aponte pro grupo da equipe do caixa (ou pro grupo de revisão, se preferir).

## Validação (30/jul) — o que foi medido

O gesto foi testado no pipeline real (pose + `PanicDetector`) frame a frame em **vídeos reais**:

- **Gesto sustentado** (pessoa de mãos erguidas por alguns segundos): **disparou** — correto.
- **Braço erguido passageiro** (polichinelo, mãos sobem e descem rápido): a pose vê o gesto, mas o **hold de 2s suprime** → **0 disparo** — correto, não é pânico.
- **39 fotos de gente normal** (parada, andando, caixa): **0 falso-positivo**.

Ou seja: pega o gesto deliberado, ignora braço erguido passageiro, **não spamma**.

## O que NÃO está incluído — e por quê (medido)

- **Detecção de arma:** testei **6 modelos** de arma prontos contra um conjunto real (armas escancaradas × celulares × cenas de loja). No ponto em que o falso-positivo em celular fica ≤5% (a exigência: *não confundir celular com arma*), o melhor modelo pega só **18%** das armas óbvias; os outros 0–6%. Pra pegar mais, o falso em celular vai a 20–42%. **Não existe ajuste que dispare em arma óbvia sem confundir celular** — enviar isso recriaria a enxurrada de alarme falso, agora gritando "ASSALTO". O **gesto de mãos-pra-cima já é a versão confiável** do mesmo produto (é a reação ao assalto).
- **Invasão do balcão** (cliente tentando abrir o caixa): *plausível, mas não pronto.* "Mão na zona do caixa" dispararia no próprio funcionário (abrir caixa, passar pano) — o falso-alerta que se quer evitar. As versões confiáveis (pessoa fora de horário; 2ª pessoa atrás do balcão) precisam ser **medidas em filmagem real do balcão** antes de prometer — mesma disciplina de go/no-go usada no resto do sistema.
