@echo off
echo Running GradCAM Verification... > verification_result.txt
echo. >> verification_result.txt

:: 1. Clear pycache to ensure no stale code
echo [1/2] Clearing __pycache__... >> verification_result.txt
if exist "ml_pipeline\__pycache__" (
    rmdir /s /q "ml_pipeline\__pycache__"
    echo Cache cleared. >> verification_result.txt
) else (
    echo No cache found. >> verification_result.txt
)

:: 2. Run Verification Script
echo [2/2] Running Python Verification Script... >> verification_result.txt
".venv\Scripts\python.exe" verify_gradcam.py >> verification_result.txt 2>&1

echo. >> verification_result.txt
echo Done. >> verification_result.txt
type verification_result.txt
