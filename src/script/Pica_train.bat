@echo off
echo Running training on Pica...

set train_config=../configs/Pica_train.txt

python main.py --config %train_config%

echo Training done.
pause
