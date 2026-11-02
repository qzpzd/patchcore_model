@echo off
setlocal
cd /d "%~dp0"

set PY=D:\files\conda_env\test-gpu\python.exe
if not exist "%PY%" (
  echo [ERROR] 未找到 GPU 环境: %PY%
  exit /b 1
)

echo 使用解释器: %PY%
"%PY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"%PY%" -c "import streamlit" 2>nul
if errorlevel 1 (
  echo [INFO] 正在向 test-gpu 环境安装 streamlit ...
  "%PY%" -m pip install "streamlit>=1.28.0"
  if errorlevel 1 (
    echo [ERROR] streamlit 安装失败。可先执行:
    echo   "%PY%" -m pip install --force-reinstall --no-deps certifi
    echo   "%PY%" -m pip install streamlit
    exit /b 1
  )
)

echo 启动 Streamlit（务必用当前环境的 python -m streamlit）...
"%PY%" -m streamlit run app.py --browser.gatherUsageStats false
endlocal
