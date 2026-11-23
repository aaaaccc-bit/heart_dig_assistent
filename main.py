"""CardioScope 启动入口。

不管进程从哪个目录被拉起（Qt Creator / 命令行 / IDE），先把工作目录和导入路径
钉到项目根目录，再交给 :mod:`widget` 真正创建窗口。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def prepare_environment() -> None:
    """把工作目录和 sys.path 都固定到项目根目录。"""

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        if Path.cwd().resolve() != PROJECT_ROOT:
            os.chdir(root)
    except OSError:
        os.chdir(root)


def main() -> int:
    prepare_environment()

    from widget import main as run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
