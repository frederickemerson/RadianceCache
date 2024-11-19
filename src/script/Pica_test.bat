@echo off
echo Running test on PicaX4...

set folders=0
set test_config=../configs/Pica_test.txt

for %%i in (%folders%) do (
    echo Testing folder %%i ...
    python main.py --config %test_config% --test_folder %%i
)

echo All tests executed.
pause
