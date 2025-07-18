import logging
import logging.config
import sys
import colorlog
from app.core.settings import settings

LOG_FORMAT = (
    "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s%(reset)s"
)

LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

def get_app_log_config(log_level: str = None):
    """Get logging configuration with specified log level."""
    if log_level is None:
        log_level = settings.LOG_LEVEL.upper()
    
    # Determine LiteLLM log level - always WARNING in INFO mode, DEBUG only in DEBUG mode
    litellm_level = log_level
    if litellm_level == "INFO":
        litellm_level = "WARNING"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": colorlog.ColoredFormatter,
                "format": "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(name)s - %(message)s",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "handlers": ["default"],
            "level": log_level,
        },
        "loggers": {
            'httpx': {'handlers': ['default'],'level': "WARNING", "propagate": False},
            'httpcore': {'handlers': ['default'],'level': "WARNING", "propagate": False},
            # LiteLLM loggers - multiple variations to catch all
            "litellm": {"handlers": ["default"], "level": litellm_level, "propagate": False},
            "LiteLLM": {"handlers": ["default"], "level": litellm_level, "propagate": False},
            "litellm.router": {"handlers": ["default"], "level": litellm_level, "propagate": False},
            "litellm.utils": {"handlers": ["default"], "level": litellm_level, "propagate": False},
            "litellm.cost_calculator": {"handlers": ["default"], "level": litellm_level, "propagate": False},
        },
    }


def setup_logging(verbose: bool = False) -> None:
    """Setup application logging based on verbose flag.
    
    Args:
        verbose: If True, sets logging to DEBUG level. Otherwise uses INFO level.
    """
    log_level = "DEBUG" if verbose else "INFO"
    
    # Configure logging
    log_config = get_app_log_config(log_level)
    logging.config.dictConfig(log_config)
    
    # Special handling for LiteLLM - it uses its own logging mechanism
    try:
        import litellm
        if verbose:
            # In verbose mode, allow LiteLLM DEBUG logs
            litellm.set_verbose = True
        else:
            # In standard mode, suppress LiteLLM INFO logs
            litellm.set_verbose = False
            # Also configure the LiteLLM logger directly
            litellm_logger = logging.getLogger("LiteLLM")
            litellm_logger.setLevel(logging.WARNING)
    except ImportError:
        pass  # LiteLLM not available
    
    # Create a logger for this module to test
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("Verbose logging enabled - DEBUG level active")
    else:
        logger.info("Standard logging enabled - INFO level active")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified name."""
    return logging.getLogger(name)