@echo off
echo @echo off > Script-install.bat
echo pip install -r requirements.txt >> Script-install.bat

echo @echo on > Script-run.bat
echo :: 运行 GUI >> Script-run.bat
echo python main.py >> Script-run.bat
echo pause >> Script-run.bat

echo @echo off > Script-push.bat
echo git pull >> Script-push.bat
echo git add . >> Script-push.bat
echo git commit -m update >> Script-push.bat
echo git push >> Script-push.bat