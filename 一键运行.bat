@echo off
setlocal EnableExtensions
chcp 936 >nul
title 垃圾图像智能分类系统 - 一键启动
cd /d "%~dp0"

echo ==================================================
echo    垃圾图像智能分类系统  一键启动
echo    PyTorch 迁移学习 + Flask 后端 + 前端页面
echo ==================================================
echo.

rem ---------- 1. 环境检查 ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请安装 Python 3.10-3.12 并勾选 Add Python to PATH
    pause
    exit /b 1
)
if not exist "src\app.py" (
    echo [错误] 未找到 src\app.py，请把本脚本放在项目根目录后再运行
    pause
    exit /b 1
)
if not exist "models\best_model.pth" (
    echo [错误] 未找到 models\best_model.pth，请确认模型文件已就位
    pause
    exit /b 1
)

rem ---------- 2. 依赖检查 ----------
python -c "import torch, torchvision, flask" >nul 2>nul
if errorlevel 1 (
    echo [提示] 检测到依赖缺失，开始安装 requirements.txt ...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络或手动执行：pip install -r requirements.txt
        pause
        exit /b 1
    )
)

rem ---------- 3. 端口检测 ----------
set PORT=5000
netstat -ano | findstr /R /C:":5000 .*LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 5000 端口已被占用，自动改用 8000
    set PORT=8000
)

rem ---------- 4. 启动服务并打开浏览器 ----------
echo [启动] 后端与前端将在 http://127.0.0.1:%PORT% 提供服务
echo [提示] 关闭本窗口或按 Ctrl+C 即可停止服务
echo.
start "" /b cmd /c "timeout /t 5 >nul & start http://127.0.0.1:%PORT%"
python src/app.py --port %PORT%

echo.
echo [已停止] 服务已退出
pause
