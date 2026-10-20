"""Нейронная сеть Хопфилда: обучение, работа и опыты с зашумлёнными образами."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .patterns import Pattern

NOISE_LEVELS = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]


@dataclass
class RunResult:
    """Состояние сети и причина остановки: fixed, cycle или limit."""

    state: list[int]
    steps: int
    status: str
    cycle_length: int = 0

    @property
    def converged(self) -> bool:
        return self.status == "fixed"


@dataclass
class Network:
    """Сеть Хопфилда.

    Веса связей считаются по правилу Хебба: w[i][j] — сумма произведений
    соответствующих элементов всех запоминаемых образов, на главной диагонали нули
    (нейрон не связан сам с собой).
    """

    size: int
    weights: list[list[int]]
    patterns: list[Pattern]

    @classmethod
    def train(cls, patterns: list[Pattern]) -> "Network":
        """Обучение сети на наборе образов."""
        size = len(patterns[0].cells)
        weights = [[0] * size for _ in range(size)]
        for pattern in patterns:
            cells = pattern.cells
            for i in range(size):
                value = cells[i]
                row = weights[i]
                for j in range(size):
                    if i != j:
                        row[j] += value * cells[j]
        return cls(size=size, weights=weights, patterns=list(patterns))

    def capacity(self) -> float:
        """Оценка из пособия; устойчивость конкретных образов проверяется отдельно."""
        n = self.size
        return n / (2 * math.log(n) + math.log(math.log(n)))

    def step_synchronous(self, state: list[int]) -> list[int]:
        """Один такт синхронного режима: все нейроны меняют состояние одновременно."""
        result = []
        for i in range(self.size):
            total = sum(self.weights[i][j] * state[j] for j in range(self.size))
            result.append(1 if total > 0 else -1)
        return result

    def step_asynchronous(self, state: list[int], generator: random.Random) -> list[int]:
        """Один такт асинхронного режима: состояние меняет один случайный нейрон."""
        state = list(state)
        index = generator.randrange(self.size)
        total = sum(self.weights[index][j] * state[j] for j in range(self.size))
        state[index] = 1 if total > 0 else -1
        return state

    def run(self, state: list[int], mode: str = "синхронный", limit: int = 50,
            seed: int | None = None) -> tuple[list[int], int]:
        """Итоговое состояние и число обновлений; причина доступна в run_detailed."""
        result = self.run_detailed(state, mode, limit, seed)
        return result.state, result.steps

    def run_detailed(self, state: list[int], mode: str = "синхронный", limit: int = 50,
                     seed: int | None = None) -> RunResult:
        """Работа до неподвижного состояния, повторения состояния или лимита.

        Синхронный такт обновляет всю сеть. Асинхронный счётчик учитывает отдельные
        обновления нейронов; лимит в обоих режимах задаёт число полных проходов.
        """
        if mode not in ("синхронный", "асинхронный"):
            raise ValueError("Неизвестный режим работы сети")
        if limit < 1:
            raise ValueError("Лимит должен быть положительным")
        generator = random.Random(seed)
        state = list(state)
        if mode == "синхронный":
            seen = {tuple(state): 0}
            for number in range(1, limit + 1):
                new_state = self.step_synchronous(state)
                if new_state == state:
                    return RunResult(state, number, "fixed")
                key = tuple(new_state)
                if key in seen:
                    return RunResult(new_state, number, "cycle", number - seen[key])
                seen[key] = number
                state = new_state
            status = "fixed" if self.step_synchronous(state) == state else "limit"
            return RunResult(state, limit, status)
        # Асинхронный режим: за одну эпоху каждый нейрон обновляется ровно один раз,
        # но в случайном порядке. Сеть останавливается, когда за целую эпоху
        # ни один нейрон не поменял своего состояния.
        updates = 0
        for _ in range(limit):
            order = list(range(self.size))
            generator.shuffle(order)
            changed = False
            for index in order:
                total = sum(self.weights[index][j] * state[j] for j in range(self.size))
                value = 1 if total > 0 else -1
                updates += 1
                if value != state[index]:
                    state[index] = value
                    changed = True
            if not changed:
                return RunResult(state, updates, "fixed")
        # Последний проход мог привести к неподвижному состоянию ровно на лимите.
        status = "fixed" if self.step_synchronous(state) == state else "limit"
        return RunResult(state, updates, status)

    def energy(self, state: list[int]) -> float:
        """Энергия; при асинхронных обновлениях симметричной сети она не возрастает."""
        total = 0
        for i in range(self.size):
            row = self.weights[i]
            for j in range(self.size):
                total += row[j] * state[i] * state[j]
        return -total / 2


def add_noise(pattern: Pattern, percent: float, generator: random.Random) -> Pattern:
    """Зашумление образа: часть клеток меняет цвет на противоположный."""
    cells = list(pattern.cells)
    count = round(len(cells) * percent / 100)
    for index in generator.sample(range(len(cells)), count):
        cells[index] = -cells[index]
    return Pattern(name=f"{pattern.name} ({percent:g} %)", cells=cells, size=pattern.size)


def difference(first: list[int], second: list[int]) -> int:
    """Сколько клеток различается у двух образов."""
    return sum(1 for a, b in zip(first, second) if a != b)


def recognize(network: Network, state: list[int]) -> tuple[int, int]:
    """К какому запомненному образу пришла сеть.

    Возвращает номер образа и число несовпавших клеток. Если состояние оказалось
    противоположным образу, расстояние до него тоже учитывается. Близость сама
    по себе не означает точного восстановления; это проверяется в single_trial.
    """
    best, best_difference = -1, len(state) + 1
    for index, pattern in enumerate(network.patterns):
        direct = difference(state, pattern.cells)
        inverted = difference(state, [-value for value in pattern.cells])
        value = min(direct, inverted)
        if value < best_difference:
            best, best_difference = index, value
    return best, best_difference


@dataclass
class Trial:
    """Одно испытание: образ зашумлён и подан на сеть."""

    pattern_index: int
    percent: float
    noisy: Pattern
    result: list[int]
    steps: int
    recognized: int
    mismatch: int
    correct: bool          # сеть пришла точно к исходному образу
    inverted: bool         # сеть пришла к негативу исходного образа
    status: str = "fixed"
    cycle_length: int = 0

    @property
    def converged(self) -> bool:
        return self.status == "fixed"

    @property
    def verdict(self) -> str:
        """Короткая расшифровка результата испытания."""
        if self.status == "cycle":
            return f"цикл длины {self.cycle_length}"
        if self.status == "limit":
            return "достигнут лимит, сходимость не подтверждена"
        if self.correct:
            return "узнан"
        if self.inverted:
            return "узнан в негативе"
        return "не узнан"


@dataclass
class NoiseRow:
    """Итог по одному уровню зашумления."""

    percent: float
    trials: int
    correct: int
    inverted: int
    by_pattern: list[int]
    average_steps: float
    cycles: int = 0
    limited: int = 0

    @property
    def share(self) -> float:
        """Доля точных распознаваний, %."""
        return self.correct / self.trials * 100 if self.trials else 0.0

    @property
    def inverted_share(self) -> float:
        """Доля случаев, когда сеть пришла к негативу образа, %."""
        return self.inverted / self.trials * 100 if self.trials else 0.0

    @property
    def failed_share(self) -> float:
        """Доля случаев, когда образ не узнан совсем, %."""
        return 100.0 - self.share - self.inverted_share


@dataclass
class Experiment:
    """Результаты опытов по всем уровням зашумления."""

    rows: list[NoiseRow] = field(default_factory=list)
    mode: str = "синхронный"
    per_level: int = 10

    def reliable_limit(self, threshold: float = 100.0) -> float | None:
        """Граница последовательности проверенных уровней без падения ниже порога."""
        best = None
        for row in sorted(self.rows, key=lambda row: row.percent):
            if row.share < threshold - 1e-9:
                break
            best = row.percent
        return best


def run_experiment(network: Network, levels: list[float] | None = None, per_level: int = 10,
                   mode: str = "синхронный", seed: int = 2026) -> Experiment:
    """Опыты по заданию: на каждый уровень шума — по несколько испытаний для каждого образа."""
    levels = levels if levels is not None else NOISE_LEVELS
    generator = random.Random(seed)
    rows: list[NoiseRow] = []
    for percent in levels:
        correct = 0
        inverted = 0
        steps = 0
        trials = 0
        cycles = 0
        limited = 0
        by_pattern = [0] * len(network.patterns)
        for index, pattern in enumerate(network.patterns):
            for _ in range(per_level):
                trial = single_trial(network, index, percent, generator, mode)
                trials += 1
                steps += trial.steps
                correct += int(trial.correct)
                inverted += int(trial.inverted)
                cycles += int(trial.status == "cycle")
                limited += int(trial.status == "limit")
                by_pattern[index] += int(trial.correct)
        rows.append(NoiseRow(percent=percent, trials=trials, correct=correct, inverted=inverted,
                             by_pattern=by_pattern, average_steps=steps / trials if trials else 0,
                             cycles=cycles, limited=limited))
    return Experiment(rows=rows, mode=mode, per_level=per_level)


def single_trial(network: Network, pattern_index: int, percent: float,
                 generator: random.Random, mode: str = "синхронный") -> Trial:
    """Одно испытание: зашумить образ, прогнать через сеть и сравнить с эталоном."""
    pattern = network.patterns[pattern_index]
    noisy = add_noise(pattern, percent, generator)
    run = network.run_detailed(noisy.cells, mode=mode, seed=generator.randrange(1 << 30))
    result, steps = run.state, run.steps
    recognized, mismatch = recognize(network, result)
    correct = run.converged and result == pattern.cells
    inverted = run.converged and result == [-value for value in pattern.cells]
    return Trial(pattern_index=pattern_index, percent=percent, noisy=noisy, result=result,
                 steps=steps, recognized=recognized, mismatch=mismatch,
                 correct=correct, inverted=inverted, status=run.status, cycle_length=run.cycle_length)


def stability_check(network: Network) -> list[tuple[str, bool, int]]:
    """Проверка: остаются ли сами запомненные образы неподвижными."""
    result = []
    for pattern in network.patterns:
        stable = network.step_synchronous(pattern.cells) == pattern.cells
        result.append((pattern.name, stable, 1))
    return result
