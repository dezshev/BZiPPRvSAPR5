"""Снимки окна приложения для README.

Запуск: python tools/make_screenshots.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

from hopfield.ui import STYLE, MainWindow  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img")
TABS = ["app_patterns", "app_weights", "app_trial", "app_experiment", "app_help"]


def wait(application: QApplication, seconds: float) -> None:
    """Пауза с обработкой событий окна."""
    end = time.time() + seconds
    while time.time() < end:
        application.processEvents()
        time.sleep(0.02)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    wait(application, 1.5)

    for index, name in enumerate(TABS):
        window.tabs.setCurrentIndex(index)
        wait(application, 0.9)
        window.grab().save(os.path.join(OUT, f"{name}.png"))
        print("сохранено", f"{name}.png")

    # отдельные картинки: восстановление при слабом и сильном шуме
    window.tabs.setCurrentIndex(2)
    for percent, name in ((20, "trial_20.png"), (45, "trial_45.png"), (70, "trial_70.png")):
        window.trial_noise.setValue(percent)
        window.run_trial()
        wait(application, 0.6)
        window.trial_figure.savefig(os.path.join(OUT, name), dpi=150, facecolor="white")
        print("сохранено", name)
    window.trial_noise.setValue(30)
    window.run_trial()

    window.tabs.setCurrentIndex(3)
    wait(application, 0.6)
    window.experiment_figure.savefig(os.path.join(OUT, "noise_curve.png"), dpi=150, facecolor="white")
    window.patterns_figure.savefig(os.path.join(OUT, "patterns.png"), dpi=150, facecolor="white")
    window.weights_figure.savefig(os.path.join(OUT, "weights.png"), dpi=150, facecolor="white")
    print("сохранены графики")
    window.close()


if __name__ == "__main__":
    main()
