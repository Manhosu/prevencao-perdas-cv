# Perguntas ao Adriano — Sistema de Prevenção de Perdas (CV)

> Documento de levantamento. Itens marcados com 🔴 são **bloqueadores**: sem eles não dá para começar (ou começamos às cegas e retrabalhamos depois).

---

Adriano, li o escopo inteiro e ele está claro. Antes de escrever a primeira linha de código, preciso fechar alguns pontos com você. Separei por blocos — os marcados com 🔴 travam o início, os outros podemos resolver ao longo do caminho.

## 1. 🔴 Acesso às câmeras (DVR / RTSP)

1. Qual a **marca e o modelo** do DVR e das câmeras? (Intelbras, Hikvision, Dahua, outro)
2. **IP do DVR, usuário e senha** — e a porta RTSP, se não for a 554.
3. Quantos canais o DVR tem no total, e **quais 5 canais** você quer monitorar?
4. O DVR oferece **substream** (stream de menor resolução)? É o que usaremos para ganhar desempenho — se tiver, me diga a resolução do principal e do substream.
5. O PC que vai rodar o sistema fica na **mesma rede local** do DVR? (Se for por internet, muda bastante a coisa.)

## 2. 🔴 Enquadramento das câmeras — o ponto mais crítico do projeto

O sistema identifica o corpo da pessoa (ombros, quadril, punhos) para saber quando a mão vai ao bolso/bolsa. **O ângulo da câmera decide se isso funciona ou não.** Câmera no teto apontada direto para baixo, ou muito longe, ou olho-de-peixe: o corpo fica distorcido e o sistema não enxerga a mão direito.

6. Me manda um **print (screenshot) da imagem ao vivo de cada uma das 5 câmeras**. É o item mais importante desta lista — com eles eu digo *antes de começar* quais câmeras têm chance real de funcionar e quais não têm.
7. As câmeras **já estão instaladas** ou ainda dá para reposicionar? (Se der para ajustar altura/ângulo, o resultado melhora muito.)
8. Qual a **distância média** da câmera até onde a pessoa passa? A pessoa aparece de corpo inteiro, da cintura para cima, ou pequena no canto da tela?
9. Como é a **iluminação**? A loja funciona à noite com infravermelho (imagem preto e branco)?
10. É **mercado ou farmácia**? Onde ficam as 5 câmeras (corredor, gôndola, caixa, entrada)?

## 3. 🔴 Vídeos para calibração

O sistema não é treinado por loja — ele usa regras (mão entrou na região do bolso e ficou lá X segundos). Essas regras precisam ser **ajustadas com vídeo do seu ambiente real**, senão ou dispara demais ou não dispara nunca.

11. Você tem **gravações de furtos reais** que já aconteceram (mesmo poucos segundos, mesmo com má qualidade)? Qualquer coisa serve.
12. Se não tiver: dá para **encenar**? Peço um funcionário na frente das câmeras fazendo, devagar e depois em ritmo normal: (a) item no bolso da calça, (b) item na bolsa/mochila, (c) item por baixo da blusa/jaqueta, (d) item na cintura. Uns 10–15 clipes curtos já resolvem.
13. Também preciso de **20–30 minutos de vídeo "normal"** (movimento comum de clientes, sem furto). É com isso que eu meço o **falso alarme** — cliente pegando o celular do bolso, coçando a barriga, mexendo na bolsa. Sem esse material eu não tenho como calibrar o limiar.

## 4. 🔴 PC onde o sistema vai rodar

14. **Configuração da máquina**: processador, memória RAM, tem placa de vídeo dedicada? Qual versão do Windows?
15. É um **PC dedicado** ao sistema ou é o mesmo PC do caixa/PDV que já roda outras coisas?
16. Ele fica **ligado 24 horas**? O sistema precisa iniciar sozinho junto com o Windows?
17. Consigo **acesso remoto** (AnyDesk/TeamViewer) para instalar, testar e calibrar? Quando?
18. Tem **antivírus ou política de TI** que bloqueia instalação de programas nessa máquina?

## 5. O que exatamente conta como "ocultação"

O escopo fala em bolso, bolsa/mochila e por baixo da roupa. Quero confirmar as bordas:

19. **Carrinho/cesta de compras** entra? (Colocar produto no carrinho é comportamento normal — assumo que **não** dispara alerta. Confirma?)
20. **Consumo no local** (abrir e comer/beber dentro da loja) deve alertar?
21. **Troca de embalagem** (colocar produto caro dentro da caixa de um barato) deve alertar? *(Adianto: isso está fora do alcance desta abordagem — se for importante para você, precisamos conversar, porque é outro projeto.)*
22. Cliente que chega com bolsa/mochila e mexe nela por motivo legítimo é comum aí? Como a loja lida com isso hoje?

## 6. Alertas e operação (como o sistema é usado no dia a dia)

23. **Quem recebe** o alerta no Telegram? (Você, o gerente, o segurança, um grupo?)
24. Ao receber o alerta, **o que a pessoa faz**? (Vai até o cliente, observa pela câmera, só registra?) Isso muda o quanto de falso alarme é tolerável.
25. Pergunta mais importante deste bloco: **quantos alertas falsos por dia sua equipe aguenta antes de começar a ignorar o sistema?** 2? 5? 10? Preciso desse número — ele é o alvo da calibração e o critério de "está bom".
26. Quer também **aviso de câmera offline** (DVR caiu, cabo solto, rede fora)? Recomendo fortemente — senão o sistema para de vigiar e ninguém percebe.
27. Precisa de **algum aviso no próprio PC** (som, popup) ou só o Telegram basta?
28. Por **quanto tempo** as imagens de evidência ficam guardadas no PC? (30 dias? Até encher o disco? Sugiro apagar automaticamente as antigas.)

## 7. Telegram

29. O **bot já foi criado** no @BotFather? Se sim, me passe o token e o grupo. Se não, eu te mando o passo a passo (leva 5 minutos) — mas o token precisa ser criado por você, é a sua conta.
30. Um grupo **por loja** ou um grupo único para todas? (Pergunto pensando no seu modelo de revenda.)

## 8. Modelo de negócio / revenda

31. Cada loja é uma **instalação independente** (um PC, um sistema, um grupo de Telegram), certo? É como está no escopo — só confirmando.
32. Você imagina, no futuro, um **painel central** vendo todas as lojas? Não entra no MVP, mas se estiver no seu horizonte eu já deixo o banco de dados preparado agora — custa pouco fazer certo desde o início, e caro depois.
33. O código-fonte é seu (está no escopo). Você quer algum tipo de **trava/licença** para o cliente final não copiar a instalação para outras lojas?

## 9. Privacidade / LGPD

34. O sistema salva **fotos de clientes** (com rosto) no PC da loja e envia para o Telegram. A loja tem aviso de monitoramento por câmera? Essa parte (adequação legal, consentimento, uso das imagens) fica sob sua responsabilidade — só quero deixar isso escrito e alinhado entre nós.

## 10. Aceite e cronograma

35. **Loja piloto**: qual é, e a partir de quando eu tenho acesso remoto a ela?
36. **Marco de 50% (fim da semana 3)**: o combinado é "câmeras capturando + detecção de comportamento funcionando" — como você quer ver isso? Sugiro uma chamada com compartilhamento de tela, eu rodando o sistema no vídeo real da sua loja e mostrando as detecções acontecendo.
37. **Aceite final**: proponho um período de teste rodando na loja, com o Telegram ativo, e nós dois avaliando os alertas de alguns dias reais. Faz sentido?
38. Alguma **data-limite** que você já tenha assumido com o cliente final?

---

## Se você só puder me mandar 3 coisas agora, mande estas:

1. **Os prints das 5 câmeras ao vivo** (item 6) — define se o projeto é viável do jeito que está.
2. **Os acessos do DVR** (IP, usuário, senha, canais — item 1) — sem isso não começo a captura.
3. **Vídeo real**, nem que seja encenado (itens 11–13) — sem isso não calibro nada.

Com isso em mãos eu já começo pela captura das câmeras e te mostro imagem rodando ainda nesta semana.

---

### Nota de expectativa (para deixar registrado entre nós)

Detectar "pessoa escondendo produto" é um dos problemas difíceis de visão computacional. O que vamos entregar é um sistema de **regras calibradas** (mão foi à região do bolso e ficou lá), não uma inteligência que "entende" furto. Isso significa, honestamente:

- Ele vai **pegar boa parte** dos casos claros de ocultação.
- Ele vai **deixar passar** alguns (mão escondida atrás do corpo, pessoa de costas, muita gente na frente).
- Ele vai **errar às vezes** (cliente guardando o próprio celular no bolso).

O objetivo do MVP é chegar num ponto em que a equipe **confia nos alertas o suficiente para agir**. É por isso que insisto tanto nos vídeos reais e no número da pergunta 25 — é o que separa um sistema que a loja usa de um que a loja desliga na segunda semana.
