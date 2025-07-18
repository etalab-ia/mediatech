import logging
import os
from datetime import datetime


def setup_logging(debug=False):
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    date = datetime.now().strftime("%Y%m%d")

    # Set log level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    logging.basicConfig(
        filename=f"logs/update_{date}.log",
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Configure specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Add console handler if needed (useful during development)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    if debug:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def get_logger(name):
    """
    Returns a configured logger with the given name.

    Args:
        name (str): Usually __name__ of the calling module

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
