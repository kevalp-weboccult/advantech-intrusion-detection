"""
Custom logging module for the Advantech Intrusion Detection System.

This module provides a flexible logging class that allows individual loggers
to have custom log levels. It includes methods to add custom log levels and
configure the logger with various options such as file rotation, console
output, and log formatting.

Example:
    logger = CustomLogger("my_logger")
    logger.add_custom_log_level("ALERT", 55)
    configured_logger = logger.get_logger()
    configured_logger.alert("This is an alert message.")
"""

import logging
import logging.handlers
import os
from typing import Dict, Literal, Optional, Union

class CustomLogger:
    """
    A flexible logging class that allows individual loggers to have custom log levels.

    This class provides functionality to create loggers with custom log levels,
    file rotation, and flexible configuration options for both file and console
    output.

    Attributes:
        name: The name of the logger instance.
        logger: The underlying Python logger object.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a CustomLogger instance.

        Args:
            name: The name identifier for this logger instance.
        """
        self.name: str = name
        self.logger: logging.Logger = logging.getLogger(name)
        self._custom_levels: Dict[str, int] = {}

    def add_custom_log_level(self, name: str, level: int) -> None:
        """
        Add a custom logging level globally, available to all loggers.

        This ensures that the custom log method (e.g., `important`) is attached
        to the logging.Logger class, and won't raise AttributeError for any logger.

        Args:
            name: The name of the custom log level (e.g., "TRACE", "ALERT").
            level: The numeric level for this custom log level.

        Raises:
            TypeError: If name is not a string or level is not an integer.
            ValueError: If the level number is already in use.

        Example:
            >>> logger = CustomLogger("test")
            >>> logger.add_custom_log_level("TRACE", 35)
        """
        if not isinstance(name, str) or not isinstance(level, int):
            raise TypeError(
                "Custom log level name must be a string and level must be an integer."
            )

        # If already defined globally, skip
        if name in logging._nameToLevel:
            return

        # If level number is already in use, raise error
        if level in logging._levelToName.values():
            raise ValueError(f"Log level '{level}' is already assigned.")

        logging.addLevelName(level, name)

        def custom_log_method(
            self: logging.Logger, message: str, *args, **kwargs
        ) -> None:
            """Custom log method for the dynamically added log level."""
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)

        # Attach it to the base Logger class — not individual instances
        setattr(logging.Logger, name.lower(), custom_log_method)
    
    def get_logger(
        self,
        log_file: Optional[str] = "app.log",
        log_level: Union[int, str] = logging.INFO,
        log_format: Optional[str] = None,
        rotation_type: Optional[Literal["size", "time"]] = "size",
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
        time_rotation_interval: Optional[
            Literal["S", "M", "H", "D", "midnight", "W"]
        ] = "midnight",
        time_rotation_when: int = 1,
        log_to_console: bool = True,
    ) -> logging.Logger:
        """
        Configure and return the logger instance.

        Args:
            log_file: The path to the log file. Defaults to "app.log".
            log_level: The log level (int or string). Defaults to logging.INFO.
            log_format: The log message format string. If None, uses default format.
            rotation_type: Type of log rotation ("size" or "time"). Defaults to "size".
            max_bytes: Maximum size of log file in bytes. Defaults to 5MB.
            backup_count: Number of backup log files to keep. Defaults to 5.
            time_rotation_interval: Interval for time-based rotation.
                Defaults to "midnight".
            time_rotation_when: Number of intervals for rotation. Defaults to 1.
            log_to_console: Whether to enable console logging. Defaults to True.

        Returns:
            The configured logger instance.

        Example:
            >>> logger = CustomLogger("test")
            >>> configured = logger.get_logger(
            ...     log_file="logs/test.log",
            ...     log_level=logging.DEBUG
            ... )
        """
        if not log_file:
            log_file = f"{self.name}.log"

        log_format = (
            log_format
            or "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        )
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            if rotation_type == "size":
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count
                )
            elif rotation_type == "time":
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when=time_rotation_interval,
                    interval=time_rotation_when,
                    backupCount=backup_count,
                )
            else:
                file_handler = logging.FileHandler(log_file)

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        return self.logger

def main() -> None:
    """
    Demonstration of CustomLogger usage.

    Shows how to create loggers with custom levels and use them
    for different logging scenarios.
    """
    # Logger with a custom TRACE level
    custom_logger_1 = CustomLogger("logger_one")
    custom_logger_1.add_custom_log_level("TRACE", 35)
    logger1 = custom_logger_1.get_logger("logs/logger_one.log")

    # Another logger without the TRACE level
    custom_logger_2 = CustomLogger("logger_two")
    logger2 = custom_logger_2.get_logger("logs/logger_two.log")

    logger1.info("This is an INFO message from logger one.")
    logger2.info("This is an INFO message from logger two.")

    logger1.trace("This is a TRACE message from logger one.")  # Works
    try:
        logger2.trace("This is a TRACE message from logger two.")  # Will raise error
    except AttributeError:
        print("TRACE level not available for logger two.")


if __name__ == "__main__":
    main()