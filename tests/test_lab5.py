"""Проверки сети Хопфилда.

Запуск из папки проекта: python -m unittest discover -s tests -v
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hopfield.network import (NOISE_LEVELS, Network, add_noise, difference, recognize,
                              Experiment, NoiseRow, run_experiment, single_trial, stability_check)
from hopfield.patterns import default_patterns, from_rows, render_letter


class TestPatterns(unittest.TestCase):
    """Подготовка образов."""

    def setUp(self) -> None:
        self.patterns = default_patterns("ДЕШ")

    def test_three_patterns(self):
        """Готовятся три образа по числу букв."""
        self.assertEqual(len(self.patterns), 3)
        self.assertEqual([pattern.name for pattern in self.patterns], ["Д", "Е", "Ш"])

    def test_values_are_plus_minus_one(self):
        """Клетки принимают только значения 1 и −1."""
        for pattern in self.patterns:
            self.assertEqual(len(pattern.cells), pattern.size ** 2)
            self.assertTrue(all(value in (1, -1) for value in pattern.cells))

    def test_patterns_are_different(self):
        """Образы отличаются друг от друга."""
        for index, first in enumerate(self.patterns):
            for second in self.patterns[index + 1:]:
                self.assertGreater(difference(first.cells, second.cells), 10)

    def test_letters_are_not_empty(self):
        """У каждой буквы есть и закрашенные, и пустые клетки."""
        for pattern in self.patterns:
            self.assertGreater(pattern.filled(), 5)
            self.assertLess(pattern.filled(), pattern.size ** 2 - 5)

    def test_any_letter_renders(self):
        """Программа готовит образ для любой буквы."""
        pattern = render_letter("Ж")
        self.assertEqual(len(pattern.cells), 100)
        self.assertTrue(any(value > 0 for value in pattern.cells))


class TestNetwork(unittest.TestCase):
    """Обучение и работа сети."""

    def setUp(self) -> None:
        self.patterns = default_patterns("ДЕШ")
        self.network = Network.train(self.patterns)

    def test_weights_are_symmetric_with_zero_diagonal(self):
        """Матрица весов симметрична, на диагонали нули."""
        for i in range(self.network.size):
            self.assertEqual(self.network.weights[i][i], 0)
            for j in range(self.network.size):
                self.assertEqual(self.network.weights[i][j], self.network.weights[j][i])

    def test_weight_matches_hebb_rule(self):
        """Вес связи равен сумме произведений элементов образов."""
        for i, j in ((0, 1), (5, 40), (99, 7)):
            manual = sum(pattern.cells[i] * pattern.cells[j] for pattern in self.patterns)
            self.assertEqual(self.network.weights[i][j], manual)

    def test_patterns_are_stable(self):
        """Запомненные образы сеть оставляет без изменений."""
        for name, stable, _ in stability_check(self.network):
            self.assertTrue(stable, name)

    def test_capacity_formula(self):
        """Ёмкость считается по формуле из методических указаний."""
        n = self.network.size
        expected = n / (2 * math.log(n) + math.log(math.log(n)))
        self.assertAlmostEqual(self.network.capacity(), expected, places=9)

    def test_energy_does_not_grow(self):
        """Энергия при асинхронной работе симметричной сети не увеличивается."""
        generator = random.Random(11)
        noisy = add_noise(self.patterns[0], 30, generator)
        before = self.network.energy(noisy.cells)
        result, _ = self.network.run(noisy.cells, mode="асинхронный", seed=11)
        self.assertLessEqual(self.network.energy(result), before + 1e-9)

    def test_asynchronous_mode_also_restores(self):
        """Асинхронный режим тоже восстанавливает слабо зашумлённый образ."""
        generator = random.Random(3)
        for index in range(len(self.patterns)):
            trial = single_trial(self.network, index, 10, generator, "асинхронный")
            self.assertTrue(trial.correct, self.patterns[index].name)


class TestNoise(unittest.TestCase):
    """Зашумление и опыты."""

    def setUp(self) -> None:
        self.patterns = default_patterns("ДЕШ")
        self.network = Network.train(self.patterns)

    def test_noise_changes_right_number_of_cells(self):
        """Зашумление меняет заданную долю клеток."""
        generator = random.Random(5)
        for percent in (10, 35, 50):
            noisy = add_noise(self.patterns[0], percent, generator)
            changed = difference(noisy.cells, self.patterns[0].cells)
            self.assertEqual(changed, round(len(self.patterns[0].cells) * percent / 100))

    def test_full_noise_gives_negative(self):
        """Стопроцентное зашумление превращает образ в негатив."""
        noisy = add_noise(self.patterns[1], 100, random.Random(1))
        self.assertEqual(noisy.cells, [-value for value in self.patterns[1].cells])

    def test_low_noise_restored(self):
        """При слабом зашумлении сеть возвращает исходный образ."""
        generator = random.Random(17)
        for index in range(len(self.patterns)):
            for percent in (10, 20):
                trial = single_trial(self.network, index, percent, generator)
                self.assertTrue(trial.correct, f"{self.patterns[index].name}, {percent} %")

    def test_experiment_covers_all_levels(self):
        """Опыты проводятся по всем уровням зашумления из задания."""
        experiment = run_experiment(self.network, per_level=5)
        self.assertEqual([row.percent for row in experiment.rows], NOISE_LEVELS)
        for row in experiment.rows:
            self.assertEqual(row.trials, 5 * len(self.patterns))
            self.assertAlmostEqual(row.share + row.inverted_share + row.failed_share, 100.0, places=6)

    def test_quality_falls_with_noise(self):
        """С ростом шума доля точных распознаваний не растёт."""
        experiment = run_experiment(self.network, per_level=10)
        shares = [row.share for row in experiment.rows if row.percent <= 50]
        self.assertTrue(all(first >= second - 1e-9 for first, second in zip(shares, shares[1:])))

    def test_reliable_limit(self):
        """До 20 % шума все образы узнаются верно."""
        experiment = run_experiment(self.network, per_level=10)
        self.assertGreaterEqual(experiment.reliable_limit() or 0, 20)

    def test_recognize_finds_nearest_pattern(self):
        """Функция узнавания находит ближайший запомненный образ."""
        index, mismatch = recognize(self.network, self.patterns[2].cells)
        self.assertEqual(index, 2)
        self.assertEqual(mismatch, 0)


class TestCustomPattern(unittest.TestCase):
    """Работа с произвольными образами."""

    def test_simple_patterns_are_restored(self):
        """Сеть запоминает и восстанавливает простые непохожие образы."""
        first = from_rows("полосы", ["1111100000"] * 10)
        second = from_rows("клетка", ["1010101010", "0101010101"] * 5)
        network = Network.train([first, second])
        generator = random.Random(9)
        for index in range(2):
            trial = single_trial(network, index, 10, generator)
            self.assertTrue(trial.correct)


class TestDynamics(unittest.TestCase):
    """Сходимость проверяется независимо от ближайшего образа и снижения энергии."""

    def setUp(self):
        self.pattern = from_rows("сплошной", ["11", "11"], size=2)
        self.network = Network.train([self.pattern])

    def test_synchronous_two_cycle_is_reported(self):
        state = [1, 1, -1, -1]
        result = self.network.run_detailed(state)
        self.assertEqual((result.status, result.cycle_length, result.steps), ("cycle", 2, 2))
        self.assertFalse(result.converged)
        self.assertNotEqual(self.network.step_synchronous(result.state), result.state)
        self.assertEqual(self.network.run(state), (result.state, result.steps))

    def test_limit_does_not_claim_convergence(self):
        result = self.network.run_detailed([1, 1, -1, -1], limit=1)
        self.assertEqual((result.status, result.steps), ("limit", 1))
        self.assertFalse(result.converged)

    def test_fixed_state_on_last_update_is_recognized(self):
        for mode in ("синхронный", "асинхронный"):
            result = self.network.run_detailed([1, 1, 1, -1], mode, limit=1, seed=0)
            self.assertTrue(result.converged)
            self.assertEqual(result.state, self.pattern.cells)

    def test_asynchronous_updates_never_increase_energy(self):
        # Все 16 состояний небольшой сети и пять порядков обновления.
        for bits in range(16):
            start = [1 if bits & (1 << i) else -1 for i in range(4)]
            for seed in range(5):
                state = start
                generator = random.Random(seed)
                for _ in range(20):
                    before = self.network.energy(state)
                    state = self.network.step_asynchronous(state, generator)
                    self.assertLessEqual(self.network.energy(state), before)
                result = self.network.run_detailed(start, "асинхронный", seed=seed)
                self.assertTrue(result.converged)
                self.assertEqual(self.network.step_synchronous(result.state), result.state)

    def test_stability_means_no_change_on_first_update(self):
        self.network.patterns = [from_rows("смешанный", ["11", "00"], size=2)]
        self.assertFalse(stability_check(self.network)[0][1])

    def test_experiment_records_cycles_as_failures(self):
        experiment = run_experiment(self.network, levels=[50], per_level=10)
        row = experiment.rows[0]
        self.assertEqual((row.correct, row.inverted, row.cycles, row.limited), (0, 0, 10, 0))
        self.assertEqual(row.failed_share, 100)
        trial = single_trial(self.network, 0, 50, random.Random(5))
        self.assertEqual(trial.verdict, "цикл длины 2")
        self.assertFalse(trial.correct or trial.inverted)

    def test_reliable_limit_does_not_skip_failed_levels(self):
        rows = [NoiseRow(p, 10, ok, 0, [ok], 2) for p, ok in [(30, 10), (10, 10), (20, 9)]]
        self.assertEqual(Experiment(rows=rows).reliable_limit(), 10)
        self.assertIsNone(Experiment(rows=rows[2:]).reliable_limit())

    def test_invalid_run_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.network.run_detailed(self.pattern.cells, "неизвестный")
        with self.assertRaises(ValueError):
            self.network.run_detailed(self.pattern.cells, limit=0)


if __name__ == "__main__":
    unittest.main()
