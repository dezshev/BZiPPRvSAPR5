"""Запуск приложения лабораторной работы № 5.

Нейронная сеть Хопфилда: запоминание букв и восстановление зашумлённых образов.
Запуск из папки проекта: python main.py
"""

import sys

from hopfield.ui import run_application

if __name__ == "__main__":
    sys.exit(run_application())
