# This Python file uses the following encoding: utf-8
import sys
import logging
from threading import Thread

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ai.knowledge_base import get_shared_knowledge_base
from ui.login_window import DEFAULT_FONT, Widget
from ui.theme import apply_qt_palette, theme_manager


def _prewarm_knowledge_base() -> None:
    try:
        get_shared_knowledge_base()
    except Exception:
        logging.getLogger(__name__).warning(
            "知识库预热失败，首次报告生成将继续按需初始化", exc_info=True
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CardioScope")
    app.setOrganizationName("CardioScope")
    app.setStyle("Fusion")
    app.setFont(QFont(DEFAULT_FONT, 10))
    theme_manager.load()
    apply_qt_palette(app)
    Thread(
        target=_prewarm_knowledge_base,
        name="knowledge-base-warmup",
        daemon=True,
    ).start()

    window = Widget()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
