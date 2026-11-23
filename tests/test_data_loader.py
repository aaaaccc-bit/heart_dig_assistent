"""Tests for the background dataset loader (QThreadPool + queued signals)."""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from algorithm.dataset import DEFAULT_DATA_PATH
import ui.data_loader as data_loader
import ui.main_dashboard as main_dashboard
from ui.data_loader import DataLoader


class DataLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for_load(self, loader: DataLoader, path: Path | None = None) -> dict:
        """Run ``loader.load()`` and pump the event loop until it settles."""

        result: dict = {}
        loop = QEventLoop()
        loader.loaded.connect(lambda data: (result.__setitem__("data", data), loop.quit()))
        loader.failed.connect(lambda message: (result.__setitem__("error", message), loop.quit()))
        QTimer.singleShot(10_000, loop.quit)
        self.assertTrue(loader.load(path, dataset_name=path.stem if path else None))
        loop.exec()
        return result

    def test_loads_bundled_dataset(self) -> None:
        loader = DataLoader()
        result = self._wait_for_load(loader)
        self.assertNotIn("error", result)
        self.assertEqual(result["data"].total_count, 303)
        self.assertFalse(loader.busy)

    def test_imported_dataset_is_used_by_analysis_pages_and_report(self) -> None:
        class InlinePool:
            def start(self, task):
                task.run()

        dashboard = main_dashboard.MainDashboard()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "imported-220.csv"
            pd.read_csv(DEFAULT_DATA_PATH).iloc[:220].to_csv(source, index=False)
            loaded = self._wait_for_load(DataLoader(), source)["data"]
            dataset = loaded.analysis_dataset
            self.assertIsNotNone(dataset)
            self.assertEqual(loaded.total_count, 220)
            dashboard.set_data(loaded, source=source)
            with (
                patch("ui.main_dashboard.QThreadPool") as pool,
                patch("ui.main_dashboard.load_dataset", side_effect=AssertionError("CSV was read again")),
                patch.object(main_dashboard, "run_logistic_regression", wraps=main_dashboard.run_logistic_regression) as regression,
                patch.object(main_dashboard, "run_random_forest", wraps=main_dashboard.run_random_forest) as forest,
                patch.object(main_dashboard, "run_risk_stratification", wraps=main_dashboard.run_risk_stratification) as risk,
                patch.object(main_dashboard, "run_feature_insight", wraps=main_dashboard.run_feature_insight) as features,
            ):
                pool.globalInstance.return_value = InlinePool()
                dashboard._on_navigate("样本分析")
                self.assertEqual(dashboard._sample_regression_result.dataset_fingerprint, dataset.fingerprint)
                dashboard._sample_page._run_regression(("age", "oldpeak"))
                self.assertEqual(dashboard._sample_regression_result.dataset_fingerprint, dataset.fingerprint)
                self.assertEqual(dashboard._sample_regression_result.predictors, ("age", "oldpeak"))
                dashboard._on_navigate("风险分层")
                dashboard._on_navigate("特征洞察")
                self.assertEqual(dashboard._risk_result.dataset_fingerprint, dataset.fingerprint)
                self.assertEqual(dashboard._feature_result.dataset_fingerprint, dataset.fingerprint)
                dashboard._on_navigate("分析报告")
                report = dashboard.report_panel.record
                self.assertIsNotNone(report)
                self.assertEqual(report.trajectory_count, 220)
                self.assertIn(source.stem, report.title)
                self.assertEqual(report.parameters["回归分析"]["自变量"], ["age", "oldpeak"])
                self.assertEqual(report.parameters["随机森林分析"]["模型参数"]["数据指纹"], dataset.fingerprint)
                self.assertEqual(report.parameters["风险分层"]["验证指标"]["数据指纹"], dataset.fingerprint)
                self.assertEqual(regression.call_count, 2)
                forest.assert_called_once()
                risk.assert_called_once()
                features.assert_called_once()
            old_result = dashboard._sample_regression_result
            replacement = self._wait_for_load(DataLoader())["data"]
            dashboard._on_navigate("工作台")
            dashboard.set_data(replacement)
            dashboard._set_sample_regression_result(old_result)
            self.assertIsNone(dashboard._sample_regression_result)
            self.assertEqual(dashboard._active_analysis().dataset.rows, 303)
        dashboard.close()

    def test_failure_is_reported(self) -> None:
        original = data_loader.load_dashboard

        def broken(path=None, dataset_name=None):
            raise ValueError("数据集缺少列: age")

        data_loader.load_dashboard = broken
        try:
            result = self._wait_for_load(DataLoader())
        finally:
            data_loader.load_dashboard = original

        self.assertEqual(result.get("error"), "数据集缺少列: age")
        self.assertNotIn("data", result)

    def test_missing_import_file_is_reported(self) -> None:
        loader = DataLoader()
        result = {}
        loop = QEventLoop()
        loader.failed.connect(lambda message: (result.__setitem__("error", message), loop.quit()))
        loader.loaded.connect(lambda _data: loop.quit())
        QTimer.singleShot(10_000, loop.quit)
        self.assertTrue(loader.load(Path("does-not-exist.csv"), "does-not-exist"))
        loop.exec()

        self.assertIn("不存在", result.get("error", ""))
        self.assertFalse(loader.busy)

    def test_second_request_is_rejected_while_busy(self) -> None:
        original = data_loader.load_dashboard

        def slow(path=None):
            time.sleep(0.3)
            return original(path)

        data_loader.load_dashboard = slow
        try:
            loader = DataLoader()
            self.assertTrue(loader.load())
            self.assertTrue(loader.busy)
            self.assertFalse(loader.load())
            self.assertTrue(loader.wait(5_000))
        finally:
            data_loader.load_dashboard = original


if __name__ == "__main__":
    unittest.main()
