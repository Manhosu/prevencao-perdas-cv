# Licenciamento — como liberar cada computador

O sistema só roda em computador liberado por você. Cada licença vale para **uma
máquina**: repassar o mesmo código para um segundo PC não funciona.

---

## O que o revendedor recebe

Uma pasta com dois arquivos que **nunca se separam** e **nunca vão para cliente**:

| Arquivo | O que é |
|---|---|
| `GerarLicenca.exe` | O programa que emite licença (dois cliques, sem instalar nada) |
| `chave_privada.pem` | A chave que assina as licenças. Insubstituível |

> **Guarde uma cópia da pasta** (pendrive, cofre de senhas). Perdeu a chave, você
> não emite mais nenhuma licença para as instalações que já existem, e precisa
> gerar chaves novas e reinstalar em todo mundo.
>
> A chave **pública** vai dentro do instalador e pode ser vista por qualquer
> pessoa. Ela só confere assinatura, não cria. É por isso que entregar o
> código-fonte não permite a ninguém emitir licença.

O gerador é compilado com `scripts/build_gerador.py`, **separado** do instalador
da loja. Ele nunca entra no `PrevencaoPerdas-Setup.exe`.

---

## A cada instalação nova

1. O cliente instala e abre o programa. Aparece a tela **"Liberar este computador"**
   com um código no formato `P4T9-MMWN-RH4U-Q8CQ`.
2. Ele te manda esse código (WhatsApp, telefone, o que for).
3. Dois cliques no `GerarLicenca.exe`: digite o código, o nome da loja e Enter na
   validade.
4. Copie o código que aparece (começa com `PP1.`) e mande para ele.
5. Ele cola na tela e clica em **Liberar computador**. Pronto.

O gerador salva um `licenca-NomeDaLoja.txt` a cada emissão. Como cada licença é
de uma máquina, essa pasta é a sua contagem de instalações por parceiro.

Pela linha de comando (desenvolvimento):

```
python scripts/gerar_licenca.py emitir --maquina P4T9-MMWN-RH4U-Q8CQ ^
    --cliente "Mercado Julie" --privada C:/MinhasChaves/chave_privada.pem
```

---

## Criar as chaves (uma única vez, já feito)

```
python scripts/gerar_licenca.py chaves --privada C:/MinhasChaves/chave_privada.pem
```

Grava a privada no caminho escolhido (fora do repositório) e a pública em
`src/licensing/chave_publica.py`. **Gerar de novo invalida todas as licenças já
emitidas** e exige gerar o instalador outra vez. Por isso o comando se recusa a
sobrescrever sem `--forcar`.

---

## Licença com prazo (para mensalidade)

Já vem pronto, mesmo sem uso agora. No gerador, em vez de Enter na validade,
digite a data no formato `2027-03-31`. No dia seguinte ao vencimento o sistema
para e avisa que precisa renovar. Para renovar, emita outra licença com data
nova para a mesma máquina.

---

## Cliente formatou o PC, trocou peça ou trocou de computador

**Formatou ou trocou de computador:** a formatação apaga o sistema junto com a
licença. Reinstale, pegue o código novo que aparece na tela e emita outra
licença. É o mesmo passo de uma instalação nova.

**Apareceu o aviso "este computador mudou":** o sistema identifica a máquina
por duas informações independentes. Algumas trocas de peça mudam só uma delas,
e nesse caso o sistema **continua funcionando por 15 dias**, avisando na tela.
Nesse prazo, peça o código novo e emita outra licença. Ao ativar, o prazo zera.

A tolerância é proposital: quem pagou está usando o sistema como segurança da
loja, e desligar a vigilância de um dia para o outro por causa de uma peça
trocada causaria um estrago maior do que a cópia que a trava evita.

Se as duas informações mudarem com o sistema ainda instalado, é outro
computador, e aí bloqueia na hora.

---

## O que esta trava faz e o que não faz

**Faz:**

- Impede instalar sem código seu.
- Impede copiar a pasta instalada para outra máquina.
- Impede reaproveitar a mesma licença num segundo PC.
- Impede alguém emitir licença, mesmo tendo o código-fonte inteiro.

**Não faz** (precisa de servidor, é outro projeto):

- Painel na internet mostrando onde está instalado.
- Desligar uma licença à distância. Consequência prática: **computador vendido
  leva a licença junto**. Oriente o cliente a desinstalar antes de vender.
- Contar sozinha quantas instalações cada parceiro fez.

---

## Se algo der errado na loja

O programa instalado não tem janela de console. Todo motivo de não abrir aparece
numa caixa de mensagem, e o registro completo fica em
`%LOCALAPPDATA%\PrevencaoPerdas\logs\app.log`.

| O que aparece | O que significa | O que fazer |
|---|---|---|
| Tela "Liberar este computador" | Máquina nova, sem licença | Peça o código e emita |
| "emitida para outro computador" | Licença de outra máquina | Emita para o código correto |
| "computador mudou" | Troca de peça | Emita nova dentro dos 15 dias |
| "a licença venceu" | Passou da validade | Emita com data nova |
| "não foi possível identificar" | PC não expõe os identificadores | Fale com o suporte, é caso raro |
| "O sistema encontrou um erro" | Falha ao abrir | Peça o `app.log` e mande para o suporte |

Rodando sem interface (`--ui` ausente), o programa sai com **código 3** e escreve
no log o motivo e o código da máquina.
