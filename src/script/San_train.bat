@echo off
echo Running training on San...

set train_config=../configs/San_train.txt

python main.py --config %train_config%

echo Training done.
pause
