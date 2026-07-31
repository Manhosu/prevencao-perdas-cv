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

## O que NÃO está incluído (com honestidade)

- **Detecção de arma:** ficou de fora de propósito — arma é objeto pequeno, incerto de longe, e pode falhar justo no momento crítico. O gesto de mãos pra cima **já cobre o assalto** (é a reação ao assalto), de forma muito mais confiável.
- **Invasão do balcão** (mão passando sobre o caixa): planejada como próximo passo, feita com cuidado.
