# Manual de Instalação e Configuração
## Sistema de Prevenção de Perdas — Alerta de Furto

Este manual é para **instalar o sistema numa loja nova**, do zero. Não precisa
saber programação. Leva cerca de 30 minutos na primeira vez.

---

## Antes de ir à loja — o que levar anotado

| O que | Onde consegue |
|---|---|
| Marca e modelo do DVR | Etiqueta atrás/embaixo do aparelho |
| IP do DVR | No próprio DVR: Menu → Rede → TCP/IP |
| Usuário e senha do DVR | Com o dono da loja |
| Quais canais monitorar | Escolhidos junto com o dono (veja "Escolhendo as câmeras") |

> 💡 **Dica de ouro:** antes de escolher os canais, olhe a imagem ao vivo de cada
> câmera. O sistema precisa **enxergar o corpo e as mãos** das pessoas. Câmera
> muito alta apontada para baixo, muito distante, ou olho-de-peixe funciona mal.
> Prefira câmeras que peguem a pessoa **de corpo inteiro, de lado ou de frente**.

---

## Passo 1 — Instalar o programa

1. Copie o instalador (`PrevencaoPerdas-Setup.exe`) para o computador da loja.
2. Dê **duplo clique** e siga o assistente (Avançar → Avançar → Instalar).
3. Marque a opção **"Iniciar com o Windows"** — assim o sistema volta sozinho
   se o computador reiniciar.
4. Ao terminar, abra pelo atalho na área de trabalho.

**Não precisa instalar mais nada.** Python, os modelos de inteligência artificial
e tudo o mais já vêm dentro do instalador — inclusive funciona sem internet.

### ⚠️ Se for notebook (importante!)

Notebook no modo "equilibrado" reduz a velocidade do processador e o sistema fica
lento sem motivo aparente. Faça isso:

1. Painel de Controle → Opções de Energia → **Alto desempenho**
2. Deixe o notebook **sempre na tomada**

---

## Passo 2 — Cadastrar as câmeras

Na aba **"Câmeras & Zonas"**:

1. Clique em **Adicionar câmera**
2. Escolha a **marca** (Intelbras/Dahua ou Hikvision)
3. Preencha **IP, usuário, senha e o número do canal**
4. Clique em **Testar conexão** — deve aparecer "Conectou! A câmera está respondendo."
5. Repita para cada câmera

> Se o teste falhar: confira o IP, o usuário/senha e se o computador está na
> **mesma rede** do DVR.

**Depois de cadastrar todas, feche e abra o programa** — é o que faz elas
começarem a ser monitoradas.

---

## Passo 3 — Desenhar a área monitorada (o mais importante!)

Esta é a etapa que **mais reduz alarme falso**. Em vez de vigiar a tela inteira,
você marca só onde o furto realmente acontece.

1. Na aba **"Câmeras & Zonas"**, selecione a câmera
2. Na imagem, **clique** para marcar os cantos da área (ex.: a gôndola do produto
   mais furtado)
3. **Duplo clique** para fechar a área
4. Clique em **Salvar zonas** — já vale na hora, sem reiniciar

**Como escolher a área:** marque a frente da prateleira de produto caro (bebida,
perfumaria, carnes). **Evite** incluir o caixa e o corredor de passagem — é onde
mais gente circula e mais alarme falso nasce.

> Se você não marcar nenhuma área, o sistema vigia a tela toda — funciona, mas
> avisa bem mais.

---

## Passo 4 — Ligar o alerta no Telegram

**Você só cria o bot UMA VEZ na vida.** O mesmo bot serve todas as suas lojas.

### Se ainda não tem o bot (só na primeira vez)

1. No Telegram, procure **@BotFather** e mande `/newbot`
2. Escolha um nome (ex.: "Alerta Loja") e um usuário terminando em "bot"
3. Ele te devolve um **token** — guarde bem, é a chave do seu negócio

### Em cada loja nova

1. Crie um **grupo** no Telegram com a equipe da loja
2. Adicione **o seu bot** nesse grupo
3. Mande qualquer mensagem no grupo (um "oi")
4. No programa, aba **"Configuração"**: cole o token e clique em
   **"Procurar meu grupo"** — o sistema acha o grupo sozinho
5. Selecione o grupo na lista e clique em **Salvar configuração do Telegram**

Pronto. Os alertas passam a chegar com **foto, hora e câmera**.

---

## Passo 5 — Ajustar a sensibilidade (calibração)

O sistema tem um "botão de sensibilidade", como um alarme de fumaça:

- **Mais sensível** → pega até gestos sutis, mas avisa mais
- **Mais conservador** → avisa menos, focado no óbvio

**Como calibrar na prática:**

1. Deixe rodando por 1 ou 2 dias
2. Na aba **"Eventos"**, veja os alertas que chegaram
3. Marque cada um: **"Foi furto"** ou **"Foi falso alarme"**
4. Se estiver avisando demais → aperte a área monitorada (Passo 3) e/ou deixe
   mais conservador
5. Se estiver deixando passar → afrouxe um pouco

> Cada vez que você marca "falso alarme", o sistema aprende sobre aquela loja.
> **A calibração melhora com o uso** — não espere estar perfeito no primeiro dia.

---

## O que esperar do sistema (expectativa honesta)

✅ **O que ele faz bem:** avisa a equipe, com foto, quando alguém faz o gesto de
esconder produto (mão ao bolso, à bolsa, sob a roupa) na área monitorada.

⚠️ **O que ele NÃO faz:** decidir sozinho se foi furto. **Quem confirma é a
pessoa.** Ele é um segundo par de olhos que chama a atenção no momento certo.

⚠️ **Alarmes falsos existem** em qualquer sistema desses. O caminho para reduzir
é: área bem marcada + calibração no uso + a equipe descartando os falsos com um
toque.

---

## Problemas comuns

| Problema | Solução |
|---|---|
| "Não consegui conectar" na câmera | Confira IP/usuário/senha e se o PC está na mesma rede do DVR |
| Câmera aparece **offline** | Verifique cabo, energia do DVR e a rede. O sistema avisa no Telegram quando cai e quando volta |
| Alerta não chega no Telegram | Aba Configuração: o bot está no grupo? Alguém mandou mensagem no grupo? Clique em "Procurar meu grupo" de novo |
| Sistema lento | Notebook: plano **Alto desempenho** + na tomada. Ou reduza o número de câmeras |
| Avisando demais | Aperte a área monitorada (Passo 3) e deixe mais conservador |
| Câmera nova não aparece | Feche e abra o programa |

---

## Manutenção

- **Fotos e vídeos** dos alertas ficam guardados **30 dias** e são apagados
  sozinhos (não enche o disco).
- **O computador precisa ficar ligado** para o sistema vigiar.
- Se marcar "Iniciar com o Windows" na instalação, ele volta sozinho após
  reinício ou queda de energia.

---

## ⚖️ Aviso legal (LGPD)

O sistema guarda **fotos de clientes** no computador da loja e envia para o
Telegram. A loja precisa ter **aviso visível de monitoramento por câmera**. A
responsabilidade pelo uso das imagens (consentimento, guarda, finalidade) é da
loja.
