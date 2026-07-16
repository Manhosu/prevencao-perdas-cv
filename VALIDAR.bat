@echo off
REM ============================================================
REM  VALIDAR.bat  -  Ferramenta de validacao do detector
REM
REM  COMO USAR: arraste um arquivo de video (.mp4) para cima
REM  deste arquivo e solte. O sistema roda o detector sobre o
REM  video e gera um video ANOTADO ao lado, mostrando:
REM    - caixa verde em volta de cada pessoa
REM    - esqueleto (ombros, quadril, bracos)
REM    - pontos VERMELHOS nos punhos (as maos)
REM    - o score no canto, e "OCULTACAO" quando dispara
REM
REM  Assista o video "_anotado.mp4" que aparece na mesma pasta.
REM ============================================================
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo.
  echo  Arraste um arquivo de video ^(.mp4^) para cima do VALIDAR.bat e solte.
  echo.
  pause
  exit /b 1
)

set "VIDEO=%~1"
set "SAIDA=%~dpn1_anotado.mp4"
set "CSV=%~dpn1_score.csv"

echo.
echo  Analisando: %VIDEO%
echo  Isso pode levar alguns minutos (o sistema olha o video quadro a quadro).
echo.

".venv\Scripts\python.exe" -m src.tools.replay "%VIDEO%" --config config\config.piloto.json --out-video "%SAIDA%" --out-csv "%CSV%" --every 3
if errorlevel 1 (
  echo.
  echo  Ocorreu um erro. Verifique se o ambiente esta instalado ^(pasta .venv^).
  pause
  exit /b 1
)

echo.
echo  Pronto! Abrindo o video anotado...
echo  Arquivo: %SAIDA%
echo.
start "" "%SAIDA%"
endlocal
