# Licenciamento — como liberar cada computador

O sistema só roda em computador liberado por você. Cada licença vale para **uma
máquina**: repassar o mesmo código para um segundo PC não funciona.

---

## Uma única vez: criar suas chaves

Na sua máquina, dentro da pasta do projeto:

```
python scripts/gerar_licenca.py chaves --privada C:/MinhasChaves/chave_privada.pem
```

Isso cria duas coisas:

| Arquivo | Onde fica | O que é |
|---|---|---|
| `chave_privada.pem` | Só com você, **fora do projeto** | O que emite licenças |
| `src/licensing/chave_publica.py` | Dentro do projeto, vai no instalador | O que confere licenças |

> **Guarde uma cópia da chave privada** (pendrive, cofre de senhas). Perdeu, você
> não emite mais nenhuma licença para as instalações que já existem, e precisa
> gerar chaves novas e reinstalar em todo mundo.
>
> A chave pública pode ser vista por qualquer pessoa. Ela só confere assinatura,
> não cria. É por isso que entregar o código-fonte não permite a ninguém emitir
> licença.

Depois de criar as chaves, **gere o instalador de novo** — ele precisa levar a
chave pública nova.

---

## A cada instalação nova

1. O cliente instala e abre o programa. Aparece a tela **"Liberar este computador"**
   com um código no formato `P4T9-MMWN-RH4U-Q8CQ`.
2. Ele te manda esse código (WhatsApp, telefone, o que for).
3. Você emite a licença:

```
python scripts/gerar_licenca.py emitir ^
    --maquina P4T9-MMWN-RH4U-Q8CQ ^
    --cliente "Mercado Julie" ^
    --privada C:/MinhasChaves/chave_privada.pem
```

4. Copie o código que aparece (começa com `PP1.`) e mande para ele.
5. Ele cola na tela e clica em **Liberar computador**. Pronto.

Guarde sua própria lista de quem recebeu o quê. Como cada licença é de uma
máquina, essa lista é a sua contagem de instalações por parceiro.

---

## Licença com prazo (para mensalidade)

Já vem pronto, mesmo sem você usar agora:

```
python scripts/gerar_licenca.py emitir --maquina ... --cliente "..." ^
    --validade 2027-03-31 --privada C:/MinhasChaves/chave_privada.pem
```

No dia seguinte ao vencimento o sistema para e avisa que precisa renovar. Para
renovar, emita outra licença com data nova para a mesma máquina.

Sem `--validade`, a licença não vence.

---

## Cliente formatou o PC ou trocou o HD

O sistema identifica a máquina por duas informações independentes. Manutenção
normal derruba só uma delas, e nesse caso o sistema **continua funcionando por
15 dias**, avisando na tela e no log.

Isso é proposital: quem pagou está usando o sistema como segurança da loja, e
desligar a vigilância de um dia para o outro por causa de uma troca de disco
causaria um estrago maior do que a cópia que a trava evita.

Nesses 15 dias, peça o código novo da máquina e emita outra licença. Ao ativar,
o prazo zera.

Se as duas informações mudarem, é outro computador, e aí bloqueia na hora.

---

## O que esta trava faz e o que não faz

**Faz:**

- Impede instalar sem código seu.
- Impede copiar a pasta instalada para outra máquina.
- Impede reaproveitar a mesma licença num segundo PC.
- Impede alguém emitir licença, mesmo tendo o código-fonte inteiro.

**Não faz** (precisa de servidor, é outro projeto):

- Painel na internet mostrando onde está instalado.
- Desligar uma licença à distância.
- Contar sozinha quantas instalações cada parceiro fez.

---

## Se algo der errado na loja

| O que aparece | O que significa | O que fazer |
|---|---|---|
| "ainda não foi liberado" | Máquina nova, sem licença | Peça o código e emita |
| "emitida para outro computador" | Licença de outra máquina | Emita para o código correto |
| "computador mudou" | Formatou ou trocou disco | Emita nova dentro dos 15 dias |
| "a licença venceu" | Passou da validade | Emita com data nova |
| "não foi possível identificar" | PC não expõe os identificadores | Fale comigo, é caso raro |

Rodando sem interface (`--ui` ausente), o programa sai com **código 3** e escreve
no log o motivo e o código da máquina.
