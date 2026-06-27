"""
utils/logging_config.py

THE ONLY FILE in this project that calls logging.basicConfig().
Every other module does: logger = logging.getLogger(__name__)

Call setup_logging() once, as early as possible, in each entry point:
  - frontend/app.py (Streamlit)
  - the `if __name__ == "__main__":` block of any script run standalone
"""

import logging

import config


def setup_logging(level: int = None) -> None:
    """
    Configure root logging for the whole app.

    logging.basicConfig() is a no-op if the root logger already has
    handlers — so calling this multiple times across entry points is safe,
    it just won't reconfigure after the first call.
    """
    level = level if level is not None else config.LOG_LEVEL
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )