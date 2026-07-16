# Como validar a entrega de 50% — Prevenção de Perdas

Este documento explica **o que foi entregue no marco de 50%** e, principalmente,
**como você consegue conferir com os próprios olhos** que está funcionando — sem
precisar entender de programação.

---

## O que é o marco de 50%

O combinado para a liberação dos 50% é: **câmeras capturando + detecção de
comportamento funcionando**. Ou seja, provar que o sistema:

1. conecta nas câmeras e enxerga as pessoas;
2. acompanha o corpo e as mãos de cada pessoa;
3. **detecta o gesto de ocultação** (mão indo ao bolso / bolsa / por baixo da
   roupa / cintura) e marca o momento.

Não faz parte deste marco (fica para a segunda metade): o alerta no Telegram, a
tela do programa, o instalador, e a **calibração fina** (deixar o sistema
preciso a ponto de quase não errar) — essa depende de vídeos de furto reais e é
a fase seguinte.

---

## O que você consegue conferir sozinho (3 provas)

### Prova 1 — O sistema enxerga e acompanha as pessoas nas SUAS câmeras

Abra os vídeos anotados que foram gerados a partir das gravações que seu irmão
trouxe da loja:

- `VID-WA0099_anotado.mp4` (corredor de bebidas — canal 11)
- `VID-WA0096_anotado.mp4` (visão ampla — canal 14)

**O que você vai ver:** a imagem da sua própria loja, e por cima dela:
- uma **caixa verde** em volta de cada pessoa;
- um **esqueleto** ligando ombros, quadril e braços;
- dois **pontos vermelhos** marcando as mãos (os punhos).

**O que isso prova:** o sistema conecta na imagem da sua câmera, encontra as
pessoas e acompanha as mãos delas — que é a base de tudo. Repare que o esqueleto
"gruda" no corpo e os pontos vermelhos seguem as mãos enquanto a pessoa anda.

### Prova 2 — O sistema DISPARA no gesto de ocultação

Esta é a prova principal, e a melhor forma de fazer é com um **vídeo de furto**.
Você mesmo pode escolher o vídeo (uma gravação de furto de mercado, sua ou da
internet — de câmera de segurança, não editada) e rodar no sistema:

1. Localize o arquivo **`VALIDAR.bat`** (na pasta do sistema).
2. **Arraste o vídeo de furto para cima do `VALIDAR.bat`** e solte.
3. Espere alguns minutos (ele olha o vídeo quadro a quadro).
4. Ele abre sozinho um vídeo **`_anotado.mp4`** com o resultado.

**O que você vai ver:** quando a pessoa leva a mão ao bolso / bolsa / roupa e
mantém lá, aparece a palavra **"OCULTACAO"** em vermelho no canto, e a caixa da
pessoa fica vermelha. Esse é o momento em que o sistema detectou o comportamento.

**O que isso prova:** o sistema não só enxerga — ele **entende o gesto de
esconder** e marca o instante. É o coração do produto funcionando.

> Você escolhe o vídeo. Não é um vídeo preparado por mim — é você quem testa,
> com o material que quiser. Essa é a validação de verdade.

### Prova 3 — O sistema é medível (a base da calibração)

Junto com cada vídeo anotado, o `VALIDAR.bat` gera um arquivo **`_score.csv`**
(abre no Excel). Cada linha é um instante do vídeo, com uma nota de 0 a 1 de
"quão suspeito" foi aquele momento. É com esse número que, na fase de
calibração, a gente ajusta o sistema para pegar o furto e não encher a loja de
alarme falso.

---

## O ponto honesto (para não haver surpresa depois)

O sistema **detecta o gesto**, mas ainda **não está calibrado** — ou seja, ainda
erra para mais (dispara às vezes em cliente normal que passa a mão perto do
corpo). Isso é **esperado nesta fase** e é exatamente o que a calibração da
segunda metade resolve, usando:

- **vídeos de furto** (encenados ou reais, seus ou da internet) → para ensinar a
  pegar o furto;
- **vídeos de movimento normal da sua loja** → para ensinar a NÃO dar alarme
  falso.

O sistema já vem com as ferramentas de medição prontas para essa calibração. O
que ela precisa é do material de vídeo.

---

## Resumo em uma frase

**Marco de 50% = o sistema enxerga suas câmeras, acompanha as pessoas e detecta o
gesto de ocultação — e você pode conferir isso rodando o `VALIDAR.bat` em
qualquer vídeo de furto que escolher.** A precisão fina (quase não errar) é a
segunda metade, junto com o Telegram, a tela e o instalador.
