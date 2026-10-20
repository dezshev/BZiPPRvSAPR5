"""Образы для обучения сети: буквы в виде чёрно-белых картинок.

Буква превращается в решётку 10×10: клетка со значением 1 — закрашена, −1 — пустая.
Такой вектор из ста чисел и подаётся на сеть Хопфилда.
"""

from __future__ import annotations

from dataclasses import dataclass

SIZE = 10

# Запасные образы на случай, если в системе не найдётся подходящего шрифта.
# Единица — закрашенная клетка. Буквы Д, Е и Ш — первые буквы фамилии.
BUILT_IN = {
    "Д": [
        "0011111100",
        "0011000100",
        "0011000100",
        "0011000100",
        "0110000100",
        "0110000100",
        "1100000100",
        "1111111111",
        "1000000001",
        "1000000001",
    ],
    "Е": [
        "0111111110",
        "0110000000",
        "0110000000",
        "0110000000",
        "0111111100",
        "0110000000",
        "0110000000",
        "0110000000",
        "0111111110",
        "0000000000",
    ],
    "Ш": [
        "1100110011",
        "1100110011",
        "1100110011",
        "1100110011",
        "1100110011",
        "1100110011",
        "1100110011",
        "1100110011",
        "1111111111",
        "0000000000",
    ],
}

FONT_NAMES = ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "segoeui.ttf", "tahomabd.ttf", "tahoma.ttf")


@dataclass
class Pattern:
    """Образ: название и вектор из значений 1 и −1."""

    name: str
    cells: list[int]
    size: int = SIZE

    def grid(self) -> list[list[int]]:
        """Образ в виде решётки: список строк."""
        return [self.cells[row * self.size:(row + 1) * self.size] for row in range(self.size)]

    def text(self) -> str:
        """Образ в виде рисунка из символов — удобно смотреть в консоли."""
        return "\n".join("".join("█" if value > 0 else "·" for value in row) for row in self.grid())

    def filled(self) -> int:
        """Сколько клеток закрашено."""
        return sum(1 for value in self.cells if value > 0)


def from_rows(name: str, rows: list[str], size: int = SIZE) -> Pattern:
    """Образ из строк с нулями и единицами."""
    cells = [1 if char == "1" else -1 for row in rows for char in row]
    return Pattern(name=name, cells=cells, size=size)


def render_letter(letter: str, size: int = SIZE) -> Pattern:
    """Отрисовка буквы шрифтом системы.

    Буква рисуется в квадрат size×size и переводится в чёрно-белый вид.
    Если подходящего шрифта нет, берётся запасной образ, а для незнакомых букв —
    простая заглушка в виде рамки.
    """
    letter = (letter or "?").strip()[:1].upper()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return _fallback(letter, size)

    for font_name in FONT_NAMES:
        try:
            font = ImageFont.truetype(font_name, size * 2)
        except OSError:
            continue
        image = Image.new("L", (size * 4, size * 4), color=0)
        draw = ImageDraw.Draw(image)
        draw.text((size * 2, size * 2), letter, fill=255, font=font, anchor="mm")
        box = image.getbbox()
        if not box:
            continue
        image = image.crop(box).resize((size, size), Image.LANCZOS)
        values = list(image.getdata())
        level = (max(values) + min(values)) / 2
        cells = [1 if value > level else -1 for value in values]
        if any(value > 0 for value in cells):
            return Pattern(name=letter, cells=cells, size=size)
    return _fallback(letter, size)


def _fallback(letter: str, size: int) -> Pattern:
    """Запасной образ: готовый рисунок буквы или рамка."""
    if letter in BUILT_IN and size == SIZE:
        return from_rows(letter, BUILT_IN[letter], size)
    rows = []
    for row in range(size):
        line = "".join("1" if row in (0, size - 1) or column in (0, size - 1) else "0"
                       for column in range(size))
        rows.append(line)
    return from_rows(letter, rows, size)


def default_patterns(letters: str = "ДЕШ", size: int = SIZE) -> list[Pattern]:
    """Три образа для обучения: по умолчанию первые буквы фамилии."""
    return [render_letter(letter, size) for letter in letters]
