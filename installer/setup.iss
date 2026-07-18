; Script do Inno Setup — instalador Windows do "Prevenção de Perdas".
;
; Empacota o que o `scripts/build_installer.py` gerou em `dist/PrevencaoPerdas/`
; (modo onedir do PyInstaller, com os modelos e o modelo de configuração em
; branco já embarcados dentro do próprio executável via --add-data). Este
; .iss NÃO referencia o arquivo de configuração real de uma loja — aquele
; arquivo tem o token do Telegram do revendedor e é gerado/editado na
; própria loja, pela aba "Configuração" da UI, nunca distribuído aqui.
;
; Compilar: ISCC installer\setup.iss  (Inno Setup precisa estar instalado)

#define MyAppName "Prevenção de Perdas — Alerta de Furto"
#define MyAppVersion "1.0"
#define MyAppExeName "PrevencaoPerdas.exe"
#define MyAppPublisher "Prevencao de Perdas"

[Setup]
AppId={{6E9C6F1B-4B8E-9C7B-5D6A-2F0E7A11B1E9}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\PrevencaoPerdas
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist_installer
OutputBaseFilename=PrevencaoPerdas-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
; O desinstalador é gerado automaticamente pelo Inno Setup
; (Uninstallable=yes é o padrão) — não desativar. É o que permite ao
; revendedor remover o sistema de uma loja sem deixar sujeira no Windows.

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na área de trabalho"; GroupDescription: "Atalhos adicionais:"
Name: "startupicon"; Description: "Iniciar com o Windows (recomendado para a loja)"; GroupDescription: "Inicialização:"; Flags: unchecked

[Files]
Source: "..\dist\PrevencaoPerdas\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--ui"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--ui"; Tasks: desktopicon

[Registry]
; Opção "Iniciar com o Windows": grava em HKCU\...\CurrentVersion\Run só se
; a tarefa "startupicon" foi marcada; a flag uninsdeletevalue remove a
; entrada ao desinstalar (nada de lixo no registro depois de removido).
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "PrevencaoPerdas"; ValueData: """{app}\{#MyAppExeName}"" --ui"; Flags: uninsdeletevalue; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--ui"; Description: "Abrir {#MyAppName}"; Flags: nowait postinstall skipifsilent
