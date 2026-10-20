"""Окно приложения: обучение сети Хопфилда, восстановление зашумлённых образов и опыты."""

from __future__ import annotations

import os
import random
import sys

# PySide6 подключается раньше matplotlib, чтобы графики использовали именно его.
os.environ.setdefault("QT_API", "pyside6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSlider, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from .network import (NOISE_LEVELS, Network, recognize, run_experiment,
                      single_trial, stability_check)
from .patterns import default_patterns

ACCENT = "#1f6feb"
DARK = "#16233a"
LIGHT = "#f4f7fb"
GOOD = "#1a7f37"
BAD = "#b3261e"
PIXELS = ListedColormap(["#ffffff", "#16233a"])

STYLE = f"""
QMainWindow, QWidget#Root {{ background: {LIGHT}; }}
QLabel#Title {{ color: white; font-size: 19px; font-weight: 600; }}
QLabel#Subtitle {{ color: #c7d6ee; font-size: 12px; }}
QFrame#Header {{ background: {DARK}; border: none; }}
QGroupBox {{
    background: white; border: 1px solid #d7e0ee; border-radius: 10px;
    margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600; color: #223;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QPushButton {{
    background: {ACCENT}; color: white; border: none; border-radius: 8px;
    padding: 9px 14px; font-weight: 600;
}}
QPushButton:hover {{ background: #3b82f6; }}
QPushButton#Secondary {{ background: white; color: {ACCENT}; border: 1px solid {ACCENT}; }}
QPushButton#Secondary:hover {{ background: #eaf2ff; }}
QLineEdit {{ border: 1px solid #d7e0ee; border-radius: 6px; padding: 6px; background: white; }}
QTableWidget {{
    background: white; border: 1px solid #d7e0ee; border-radius: 8px;
    gridline-color: #e6ecf5; selection-background-color: #dbeafe; selection-color: #16233a;
}}
QHeaderView::section {{
    background: #eef3fa; color: #223; border: none; border-right: 1px solid #dfe7f2;
    border-bottom: 1px solid #dfe7f2; padding: 7px; font-weight: 600;
}}
QTabWidget::pane {{ border: 1px solid #d7e0ee; border-radius: 10px; background: white; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: #4a5a72; padding: 9px 16px; margin-right: 4px;
    border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600;
}}
QTabBar::tab:selected {{ background: white; color: {ACCENT}; border: 1px solid #d7e0ee; border-bottom: 1px solid white; }}
QFrame#Card {{ background: white; border: 1px solid #d7e0ee; border-radius: 10px; }}
QLabel#CardValue {{ font-size: 20px; font-weight: 700; color: {DARK}; }}
QLabel#CardName {{ color: #5b6b83; font-size: 12px; }}
QLabel#CardHint {{ color: {GOOD}; font-size: 12px; font-weight: 600; }}
QTextBrowser {{ background: white; border: 1px solid #d7e0ee; border-radius: 8px; padding: 10px; }}
QStatusBar {{ background: white; color: #4a5a72; }}
QSlider::groove:horizontal {{ height: 6px; background: #dfe7f2; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QSlider::sub-page:horizontal {{ background: #9dc1fb; border-radius: 3px; }}
"""


def number(value: float, digits: int = 1) -> str:
    """Число с запятой в качестве десятичного разделителя."""
    return f"{value:.{digits}f}".replace(".", ",")


def card(name: str, value: str = "—", hint: str = "") -> tuple[QFrame, QLabel, QLabel]:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 10, 14, 10)
    title = QLabel(name)
    title.setObjectName("CardName")
    value_label = QLabel(value)
    value_label.setObjectName("CardValue")
    value_label.setWordWrap(True)
    hint_label = QLabel(hint)
    hint_label.setObjectName("CardHint")
    hint_label.setWordWrap(True)
    layout.addWidget(title)
    layout.addWidget(value_label)
    layout.addWidget(hint_label)
    return frame, value_label, hint_label


def table(headers: list[str], rows: int = 0) -> QTableWidget:
    widget = QTableWidget(rows, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().setVisible(False)
    widget.setAlternatingRowColors(True)
    widget.setEditTriggers(QTableWidget.NoEditTriggers)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return widget


def item(text: str, bold: bool = False, color: str | None = None) -> QTableWidgetItem:
    cell = QTableWidgetItem(text)
    cell.setTextAlignment(Qt.AlignCenter)
    if bold:
        font = cell.font()
        font.setBold(True)
        cell.setFont(font)
    if color:
        cell.setBackground(QColor(color))
    return cell


def draw_pattern(axes, cells: list[int], size: int, title: str, color: str = DARK) -> None:
    """Рисование образа в виде чёрно-белой решётки."""
    grid = [[1 if cells[row * size + column] > 0 else 0 for column in range(size)]
            for row in range(size)]
    axes.imshow(grid, cmap=PIXELS, vmin=0, vmax=1)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(title, fontsize=10, color=color)
    for spine in axes.spines.values():
        spine.set_edgecolor("#9fb3cd")


class MainWindow(QMainWindow):
    """Главное окно лабораторной работы."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Нейронная сеть Хопфилда: запоминание и восстановление образов")
        self.resize(1380, 880)
        self.letters = "ДЕШ"
        self._build()
        self.rebuild_network()

    # ---------- интерфейс ----------

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(2)
        title = QLabel("Лабораторная работа № 5. Нейронная сеть Хопфилда")
        title.setObjectName("Title")
        subtitle = QLabel(
            "Сеть запоминает три буквы фамилии как картинки 10×10 и восстанавливает их "
            "из зашумлённых образов. Проверяется, при каком уровне шума узнавание ещё надёжно."
        )
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(12)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.card_net, self.card_net_value, self.card_net_hint = card("Сеть")
        self.card_capacity, self.card_capacity_value, self.card_capacity_hint = card("Оценка ёмкости")
        self.card_limit, self.card_limit_value, self.card_limit_hint = card("Без ошибок в серии")
        self.card_stable, self.card_stable_value, self.card_stable_hint = card("Устойчивость образов")
        for widget in (self.card_net, self.card_capacity, self.card_limit, self.card_stable):
            cards.addWidget(widget)
        layout.addLayout(cards)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_patterns(), "Образы")
        self.tabs.addTab(self._tab_weights(), "Весовая матрица")
        self.tabs.addTab(self._tab_trial(), "Восстановление")
        self.tabs.addTab(self._tab_experiment(), "Опыты с шумом")
        self.tabs.addTab(self._tab_help(), "Как считается")
        layout.addWidget(self.tabs, 1)
        outer.addWidget(body, 1)
        self.statusBar().showMessage("Готово")

    def _tab_patterns(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        self.letters_input = QLineEdit(self.letters)
        self.letters_input.setMaxLength(6)
        self.letters_input.setFixedWidth(120)
        apply_button = QPushButton("Обучить сеть")
        apply_button.clicked.connect(self.rebuild_network)
        controls.addWidget(QLabel("Буквы для запоминания:"))
        controls.addWidget(self.letters_input)
        controls.addWidget(apply_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.patterns_figure = Figure(figsize=(9, 3.6), dpi=100)
        self.patterns_canvas = FigureCanvas(self.patterns_figure)
        layout.addWidget(self.patterns_canvas, 1)

        self.patterns_info = QLabel()
        self.patterns_info.setWordWrap(True)
        self.patterns_info.setStyleSheet("color: #4a5a72;")
        layout.addWidget(self.patterns_info)
        return widget

    def _tab_weights(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        self.weights_figure = Figure(figsize=(7, 5.4), dpi=100)
        self.weights_canvas = FigureCanvas(self.weights_figure)
        layout.addWidget(self.weights_canvas, 1)
        self.weights_info = QLabel()
        self.weights_info.setWordWrap(True)
        self.weights_info.setStyleSheet("color: #4a5a72;")
        layout.addWidget(self.weights_info)
        return widget

    def _tab_trial(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QGroupBox("Что подать на сеть")
        controls_layout = QHBoxLayout(controls)
        self.trial_pattern = QComboBox()
        self.trial_noise = QSlider(Qt.Horizontal)
        self.trial_noise.setRange(0, 100)
        self.trial_noise.setValue(30)
        self.trial_noise.valueChanged.connect(self.update_noise_label)
        self.trial_noise_label = QLabel("30 %")
        self.trial_noise_label.setFixedWidth(50)
        self.trial_mode = QComboBox()
        self.trial_mode.addItems(["синхронный", "асинхронный"])
        run_button = QPushButton("Подать на сеть")
        run_button.clicked.connect(self.run_trial)
        controls_layout.addWidget(QLabel("Образ:"))
        controls_layout.addWidget(self.trial_pattern)
        controls_layout.addWidget(QLabel("Зашумление:"))
        controls_layout.addWidget(self.trial_noise, 1)
        controls_layout.addWidget(self.trial_noise_label)
        controls_layout.addWidget(QLabel("Режим:"))
        controls_layout.addWidget(self.trial_mode)
        controls_layout.addWidget(run_button)
        layout.addWidget(controls)

        self.trial_figure = Figure(figsize=(9, 3.4), dpi=100)
        self.trial_canvas = FigureCanvas(self.trial_figure)
        layout.addWidget(self.trial_canvas, 1)

        self.trial_result = QLabel()
        self.trial_result.setWordWrap(True)
        self.trial_result.setStyleSheet("font-size: 13px; color: #16233a; padding: 4px;")
        layout.addWidget(self.trial_result)
        return widget

    def _tab_experiment(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        self.per_level = QSpinBox(minimum=5, maximum=50)
        self.per_level.setValue(10)
        self.experiment_mode = QComboBox()
        self.experiment_mode.addItems(["синхронный", "асинхронный"])
        run_button = QPushButton("Провести опыты")
        run_button.clicked.connect(self.run_experiment)
        controls.addWidget(QLabel("Испытаний на образ и уровень:"))
        controls.addWidget(self.per_level)
        controls.addWidget(QLabel("Режим:"))
        controls.addWidget(self.experiment_mode)
        controls.addWidget(run_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.experiment_table = table(["Зашумление,\n%", "Испыта-\nний", "Узнан,\n%",
                                       "Узнан в\nнегативе, %", "Не узнан,\n%", "Тактов\nв среднем"])
        body.addWidget(self.experiment_table, 1)
        self.experiment_figure = Figure(figsize=(6.4, 4.4), dpi=100)
        self.experiment_canvas = FigureCanvas(self.experiment_figure)
        body.addWidget(self.experiment_canvas, 1)
        layout.addLayout(body, 1)

        self.experiment_note = QLabel()
        self.experiment_note.setWordWrap(True)
        self.experiment_note.setStyleSheet("font-size: 13px; color: #16233a; padding: 4px;")
        layout.addWidget(self.experiment_note)
        return widget

    def _tab_help(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(12, 12, 12, 12)
        view = QTextBrowser()
        view.setHtml(HELP_HTML)
        layout.addWidget(view)
        return holder

    # ---------- работа ----------

    def rebuild_network(self) -> None:
        """Обучение сети на введённых буквах."""
        letters = self.letters_input.text().strip() or "ДЕШ"
        self.letters = letters
        self.patterns = default_patterns(letters)
        self.network = Network.train(self.patterns)
        self.trial_pattern.blockSignals(True)
        self.trial_pattern.clear()
        for pattern in self.patterns:
            self.trial_pattern.addItem(f"Буква «{pattern.name}»")
        self.trial_pattern.blockSignals(False)
        self._draw_patterns()
        self._draw_weights()
        self.run_trial()
        self.run_experiment()

    def _draw_patterns(self) -> None:
        figure = self.patterns_figure
        figure.clear()
        panels = figure.subplots(1, len(self.patterns))
        if len(self.patterns) == 1:
            panels = [panels]
        for panel, pattern in zip(panels, self.patterns):
            draw_pattern(panel, pattern.cells, pattern.size,
                         f"«{pattern.name}»: закрашено {pattern.filled()} из {pattern.size ** 2}")
        figure.suptitle("Образы, на которых обучена сеть", fontsize=12, color=DARK)
        figure.tight_layout()
        self.patterns_canvas.draw_idle()

        checks = stability_check(self.network)
        stable = sum(1 for _, ok, _ in checks if ok)
        self.patterns_info.setText(
            f"Каждая буква переведена в решётку {self.patterns[0].size}×{self.patterns[0].size}: "
            f"закрашенная клетка — это 1, пустая — −1. Всего нейронов {self.network.size}. "
            f"Проверка устойчивости: {stable} из {len(checks)} образов сеть оставляет без изменений."
        )
        self.card_stable_value.setText(f"{stable} из {len(checks)}")
        self.card_stable_hint.setText("образы неподвижны" if stable == len(checks)
                                      else "часть образов сеть меняет")
        self.card_net_value.setText(f"{self.network.size} нейронов")
        self.card_net_hint.setText(f"запомнено образов: {len(self.patterns)}")
        capacity = self.network.capacity()
        self.card_capacity_value.setText(number(capacity, 2) + " образа")
        self.card_capacity_hint.setText("по формуле n / (2·ln n + ln ln n)")

    def _draw_weights(self) -> None:
        figure = self.weights_figure
        figure.clear()
        axes = figure.add_subplot(111)
        image = axes.imshow(self.network.weights, cmap="coolwarm")
        figure.colorbar(image, ax=axes, shrink=0.8, label="вес связи")
        axes.set_title("Весовая матрица сети", fontsize=12, color=DARK)
        axes.set_xlabel("номер нейрона")
        axes.set_ylabel("номер нейрона")
        figure.tight_layout()
        self.weights_canvas.draw_idle()

        values = [value for row in self.network.weights for value in row]
        self.weights_info.setText(
            f"Элементов матрицы {len(values):,}".replace(",", " ") +
            f"; наибольший вес {max(values)}, наименьший {min(values)}. "
            "На главной диагонали нули: нейрон не связан сам с собой. При асинхронных "
            "обновлениях энергия не возрастает; синхронный режим может попасть в цикл."
        )

    def update_noise_label(self) -> None:
        self.trial_noise_label.setText(f"{self.trial_noise.value()} %")

    def run_trial(self) -> None:
        """Одно испытание: зашумить выбранный образ и восстановить его."""
        if not hasattr(self, "network"):
            return
        index = max(0, self.trial_pattern.currentIndex())
        percent = self.trial_noise.value()
        mode = self.trial_mode.currentText()
        generator = random.Random(percent * 100 + index)
        trial = single_trial(self.network, index, percent, generator, mode)

        figure = self.trial_figure
        figure.clear()
        panels = figure.subplots(1, 3)
        pattern = self.patterns[index]
        draw_pattern(panels[0], pattern.cells, pattern.size, "эталон")
        draw_pattern(panels[1], trial.noisy.cells, pattern.size, f"зашумлён на {percent} %")
        color = GOOD if trial.correct else (ACCENT if trial.inverted else BAD)
        draw_pattern(panels[2], trial.result, pattern.size, f"после сети: {trial.verdict}", color)
        figure.tight_layout()
        self.trial_canvas.draw_idle()

        recognized, mismatch = recognize(self.network, trial.result)
        energy_before = self.network.energy(trial.noisy.cells)
        energy_after = self.network.energy(trial.result)
        stop = ("Достигнуто неподвижное состояние." if trial.converged else
                f"Обнаружен цикл длины {trial.cycle_length}; неподвижное состояние не достигнуто."
                if trial.status == "cycle" else "Достигнут лимит; сходимость не подтверждена.")
        units = (f"Синхронных тактов: {trial.steps}." if mode == "синхронный" else
                 f"Обновлений нейронов: {trial.steps} ({trial.steps // self.network.size} полных проходов).")
        self.trial_result.setText(
            f"<b>Результат:</b> {trial.verdict}. Ближайший эталон или его негатив — "
            f"«{self.patterns[recognized].name}», расхождение {mismatch} клеток из {self.network.size}.<br>"
            f"{units} Энергия: {number(energy_before, 0)} → {number(energy_after, 0)}. {stop}"
        )

    def run_experiment(self) -> None:
        """Опыты по всем уровням зашумления."""
        if not hasattr(self, "network"):
            return
        self.experiment = run_experiment(self.network, per_level=self.per_level.value(),
                                         mode=self.experiment_mode.currentText())
        rows = self.experiment.rows
        widget = self.experiment_table
        widget.horizontalHeaderItem(5).setText(
            "Синхронных\nтактов\nв среднем" if self.experiment.mode == "синхронный"
            else "Обновлений\nнейронов\nв среднем")
        widget.setRowCount(len(rows))
        for row, data in enumerate(rows):
            good = data.share >= 99.9
            cells = [f"{data.percent:g}", str(data.trials), number(data.share),
                     number(data.inverted_share), number(data.failed_share), number(data.average_steps)]
            for column, text in enumerate(cells):
                widget.setItem(row, column, item(text, bold=column == 0,
                                                 color="#e6f4ea" if good else None))

        figure = self.experiment_figure
        figure.clear()
        axes = figure.add_subplot(111)
        percents = [row.percent for row in rows]
        axes.plot(percents, [row.share for row in rows], marker="o", color=GOOD, linewidth=2,
                  label="узнан точно")
        axes.plot(percents, [row.inverted_share for row in rows], marker="s", color=ACCENT,
                  linewidth=1.6, label="узнан в негативе")
        axes.plot(percents, [row.failed_share for row in rows], marker="^", color=BAD,
                  linewidth=1.6, label="не узнан")
        limit = self.experiment.reliable_limit()
        if limit:
            axes.axvline(limit, color="#9fb3cd", linestyle="--", linewidth=1.4)
            axes.annotate(f"в серии до {limit:g} % — без ошибок", (limit, 50), textcoords="offset points",
                          xytext=(8, 0), fontsize=9, color="#4a5a72")
        axes.set_xlabel("зашумление образа, %")
        axes.set_ylabel("доля испытаний, %")
        axes.set_title("Результаты опытов", fontsize=11)
        axes.set_ylim(-5, 105)
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)
        figure.tight_layout()
        self.experiment_canvas.draw_idle()

        total = self.experiment.per_level * len(self.patterns)
        result_note = (f"<b>На проверенных уровнях до {limit:g} % в этой серии ошибок нет.</b>"
                       if limit is not None else "На первом проверенном уровне уже есть ошибки.")
        cycles = sum(row.cycles for row in rows)
        limited = sum(row.limited for row in rows)
        self.experiment_note.setText(
            f"На каждый уровень зашумления приходится {total} испытаний "
            f"({self.experiment.per_level} на каждую букву), режим — {self.experiment.mode}.<br>"
            f"{result_note} Циклов: {cycles}, остановок по лимиту: {limited}; "
            "они включены в долю «не узнан». Результат относится к проведённым испытаниям."
        )
        self.card_limit_value.setText(f"до {limit:g} %" if limit is not None else "—")
        self.card_limit_hint.setText(f"{total} испытаний на уровень")
        self.statusBar().showMessage(
            f"Опыты выполнены: {len(rows)} уровней зашумления, режим — {self.experiment.mode}")


HELP_HTML = """
<h3>Что делает программа</h3>
<p>Сеть Хопфилда запоминает несколько картинок и умеет восстанавливать их из испорченных.
Здесь запоминаются три буквы фамилии, каждая — решётка 10×10 клеток: закрашенная клетка
это 1, пустая — −1. Значит, в сети 100 нейронов.</p>

<h3>Обучение</h3>
<p>Вес связи между нейронами считается по формуле из методических указаний:</p>
<p style="text-align:center"><b>w(i, j) = Σ a(i,k)·a(j,k)</b> при i ≠ j, и <b>w(i, i) = 0</b></p>
<p>Здесь a(i,k) — значение i-й клетки k-го образа. Матрица симметрична.
При последовательных обновлениях энергия E = −½ Σ w(i,j)·a(i)·a(j) не возрастает.
При синхронных обновлениях возможен цикл, поэтому сходимость проверяется отдельно.</p>

<h3>Работа сети</h3>
<p>Состояние нейрона на следующем такте: <b>a(i) = f( Σ w(i,j)·a(j) )</b>, где f даёт 1
при положительной сумме и −1 при неположительной.</p>
<ul>
<li><b>Синхронный режим</b> — все нейроны меняются одновременно;</li>
<li><b>асинхронный режим</b> — за один проход каждый нейрон обновляется по очереди
в случайном порядке.</li>
</ul>
<p>Работа завершается при неподвижном состоянии, обнаружении синхронного цикла
или достижении лимита в 50 проходов. Причина остановки показывается в результате.
Синхронный счётчик учитывает такты всей сети, асинхронный — отдельные обновления нейронов.</p>

<h3>Опыты с зашумлением</h3>
<p>Зашумление на P процентов означает, что у P процентов клеток цвет меняется
на противоположный. Для каждого уровня шума проводится несколько испытаний с каждой буквой,
и считается доля случаев, когда сеть вернула исходный образ.</p>
<p>При шуме больше половины клеток вход ближе к негативу исходной буквы по расстоянию
Хэмминга, но это не гарантирует переход именно к нему. Устойчивость негатива проверяется
так же, как устойчивость самого образа; точное восстановление негатива показано отдельно.</p>
<p>Проценты относятся к данной серии испытаний и не гарантируют восстановление
при любом расположении повреждённых клеток.</p>

<h3>Ёмкость сети</h3>
<p>Сколько образов сеть способна удержать, оценивается по формуле
<b>m = n / (2·ln n + ln ln n)</b>. Для ста нейронов оценка равна примерно 9,31.
Она не гарантирует устойчивость произвольного набора букв: устойчивость зависит
от самих образов и проверяется подстановкой каждого из них в сеть.</p>
"""


def run_application() -> int:
    """Запуск приложения."""
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLE)
    application.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return application.exec()
