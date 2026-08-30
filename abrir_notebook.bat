@echo off
python -m jupyter notebook modelagem_machine_learning.ipynb
if errorlevel 1 py -m jupyter notebook modelagem_machine_learning.ipynb
