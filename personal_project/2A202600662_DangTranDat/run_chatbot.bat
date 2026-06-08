@echo off
echo ========================================
echo   LawBot RAG Chatbot - Day 8 Group Project
echo ========================================
echo.

REM Try venv streamlit first
if exist ".venv\Scripts\streamlit.exe" (
    echo [OK] Found venv streamlit
    .venv\Scripts\streamlit.exe run group_project/member1_ui/app.py
    goto :end
)

REM Try system streamlit
where streamlit >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [OK] Found system streamlit
    streamlit run group_project/member1_ui/app.py
    goto :end
)

REM Try python -m streamlit
echo [WARN] streamlit not found in PATH, trying python -m streamlit...
python -m streamlit run group_project/member1_ui/app.py

:end
