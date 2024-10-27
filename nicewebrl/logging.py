"""

A simple utility for logging to files and the console. Key features:
- prepend user_id to console logs
- create user-specific log files with all stdout and stderr (useful for user-specific debugging) 

Usage:

from nicewebrl import logging
logging.setup_logging(
    LOG_DIR,
    # use whatever key you use to identify users in nicegui
    nicegui_storage_user_key='user_id'
)
logger = logging.get_logger('some name')

logger.info('some message')
logger.error('some error message')
logger.warning('some warning message')

NOTE: you only need to call setup_logging once at the beginning of your program. the settings will propogate to every other usage. afterwards, you just need to call.

# e.g.
logger = logging.get_logger(__name__)

"""

import traceback
import io

import logging
import sys
import os
from nicegui import app
from functools import lru_cache


class UserAwareOutput(io.TextIOBase):
    """
    A custom output stream that handles user-specific and global logging.

    This class extends io.TextIOBase to provide a flexible logging solution
    that can differentiate between user-specific and global log messages.
    It manages separate buffers for each user and a global buffer.

    Attributes:
        console_stream (io.TextIOBase): The underlying console stream for output.
        user_key (str): The key used to identify users in the app storage.
        log_dir (str): The directory where log files are stored.
        error_log (str): The path to the error log file.
        buffers (dict): A dictionary to store user-specific StringIO buffers.
    """

    def __init__(
            self,
            console_stream: io.TextIOBase,
            log_dir: str,
            nicegui_storage_user_key: str = 'user_id'):
        """
        Initialize the UserAwareOutput instance.

        Args:
            console_stream (io.TextIOBase): The underlying console stream for output.
            log_dir (str): The directory where log files will be stored.
            user_key (str, optional): The key used to identify users in the app storage. Defaults to 'user_id'.
        """
        self.console_stream = console_stream
        self.user_key = nicegui_storage_user_key
        self.log_dir = log_dir
        self.error_log = os.path.join(log_dir, 'error.log')
        self.buffers = {}  # Dictionary to store user-specific buffers
        self.user_to_idx = {}

    def _get_buffer(self, user_id):
        """
        Get or create a buffer for a specific user.
        """
        if user_id not in self.buffers:
            self.buffers[user_id] = io.StringIO()
            self.user_to_idx[user_id] = len(self.buffers) - 1
        return self.buffers[user_id]

    @property
    def nusers(self):
        if 'global' in self.buffers:
            return len(self.buffers) - 1  # exclude global
        else:
            return len(self.buffers)

    def write(self, s: str, get_user_id: bool = True) -> int:
        """
        Write a string to the appropriate output stream.

        Args:
            s (str): The string to write.
            get_user_id (bool, optional): Whether to attempt to get a user ID. Defaults to True.

        Returns:
            int: The number of characters written.
        """
        try:
            if get_user_id:
                return self.write_user(s)
            else:
                return self.write_global(s)
        except Exception as e:
            self.log_error(
                f"Error in UserAwareOutput.write: {str(e)}\n{traceback.format_exc()}")
            return 0

    def write_user(self, s: str) -> int:
        """
        Write a string to a user-specific log file and buffer.
        """
        try:
            user_id = app.storage.user.get(self.user_key, None)
            log_file = os.path.join(self.log_dir, f'log_{user_id}.log')

            with open(log_file, 'a') as file_stream:
                file_stream.write(s)
                file_stream.flush()

            return self._write_to_buffer(s, user_id)
        except Exception as e:
            if 'app.storage.user' in str(e):
                # Silently ignore this specific error and fall back to global write
                return self.write_global(s)
            else:
                # Log all other exceptions
                self.log_error(
                    f"Error in write_user: {str(e)}\n{traceback.format_exc()}")
                return 0

    def write_global(self, s: str) -> int:
        """
        Write a string to the global buffer.
        """
        return self._write_to_buffer(s, 'global')

    def _write_to_buffer(self, s: str, user_id: str) -> int:
        """
        Write a string to a specific buffer and flush if necessary..
        """
        buffer = self._get_buffer(user_id)
        buffer.write(s)
        if s.endswith('\n'):
            output = buffer.getvalue()
            if user_id != 'global' and output.strip():
                idx = self.user_to_idx[user_id]
                output = f"({idx}/{self.nusers}) {user_id}: {output}"
            self.console_stream.write(output)
            buffer.truncate(0)
            buffer.seek(0)
        return len(s)

    def flush(self):
        """
        Flush all buffers, writing their contents to the console stream.
        """
        try:
            for user_id, buffer in self.buffers.items():
                output = buffer.getvalue()
                if output:
                    if user_id != 'global' and output.strip():
                        output = f"{user_id}: {output}"
                    self.console_stream.write(output)
                    buffer.truncate(0)
                    buffer.seek(0)
            self.console_stream.flush()
        except Exception as e:
            self.log_error(
                f"Error in UserAwareOutput.flush: {str(e)}\n{traceback.format_exc()}")

    def log_error(self, error_message):
        """
        Log an error message to the error
        """
        with open(self.error_log, 'a') as error_file:
            error_file.write(f"{error_message}\n")


class UserAwareHandler(logging.Handler):
    """
    A custom logging handler that writes log messages to a UserAwareOutput.

    This handler is designed to work with the UserAwareOutput class, which
    manages user-specific logging streams.

    Attributes:
        user_aware_output (UserAwareOutput): The output stream to write log messages to.
    """

    def __init__(self, user_aware_output):
        """
        Initialize the UserAwareHandler.

        Args:
            user_aware_output (UserAwareOutput): The output stream to write log messages to.
        """
        super().__init__()
        self.user_aware_output = user_aware_output

    def emit(self, record):
        """
        Emit a log record.

        This method formats the log record and writes it to the UserAwareOutput.
        If an error occurs during emission, it logs the error to a separate error log file.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        try:
            msg = self.format(record)
            self.user_aware_output.write(msg + '\n')
        except Exception as e:
            # Log the error to a separate error log file
            error_msg = f"Error in UserAwareHandler.emit: {str(e)}\n{traceback.format_exc()}"
            self.user_aware_output.log_error(error_msg)
            # Optionally, you can also print to stderr for immediate visibility
            print(error_msg, file=sys.stderr)

    def flush(self):
        """
        Flush the handler.

        This method ensures that any buffered log records are written out.
        """
        self.user_aware_output.flush()

class WatchfilesFilter(logging.Filter):
    def __init__(self, ignored_dirs=('data/', '.nicegui/')):
        super().__init__()
        self.ignored_dirs = ignored_dirs

    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if 'changes detected:' in record.msg:
                changes = eval(record.msg.split(':', 1)[1])
                filtered_changes = [
                    change for change in changes
                    if not change[1].startswith(self.ignored_dirs)
                ]
                if not filtered_changes:
                    return False
                record.msg = f"{len(filtered_changes)} changes detected: {set(filtered_changes)}"
        return True

@lru_cache(maxsize=None)
def get_logger(name: str):
    return logging.getLogger(name)

def setup_logging(
        log_dir: str,
        nicegui_storage_user_key: str = 'user_id',
        watchfiles_logging: bool = False,
        ignored_watchfiles_dirs: tuple = ('data/', '.nicegui/')):
    """
    Set up logging configuration.

    Args:
    log_dir (str): Directory for log files.
    nicegui_storage_user_key (str): Key for user storage in NiceGUI.
    watchfiles_logging (bool): Flag to enable detailed watchfiles logging.
    ignored_watchfiles_dirs (tuple): Directories to ignore in watchfiles logging.

    Returns:
    logging.Logger: Configured root logger.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create UserAwareOutput instances
    user_aware_stdout = UserAwareOutput(sys.stdout, log_dir=log_dir, nicegui_storage_user_key=nicegui_storage_user_key)
    user_aware_stderr = UserAwareOutput(sys.stderr, log_dir=log_dir, nicegui_storage_user_key=nicegui_storage_user_key)

    # Create and configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handlers using UserAwareOutput
    stdout_handler = UserAwareHandler(user_aware_stdout)
    stderr_handler = UserAwareHandler(user_aware_stderr)

    # Set formatting for the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Add the handlers to the root logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Redirect sys.stdout and sys.stderr
    sys.stdout = user_aware_stdout
    sys.stderr = user_aware_stderr

    # Configure watchfiles logger
    watchfiles_logger = logging.getLogger('watchfiles')
    watchfiles_logger.addFilter(
        WatchfilesFilter(ignored_dirs=ignored_watchfiles_dirs))

    if watchfiles_logging:
        watchfiles_logger.setLevel(logging.DEBUG)
        watchfiles_handler = logging.StreamHandler()
        watchfiles_handler.setFormatter(formatter)
        watchfiles_logger.addHandler(watchfiles_handler)
    else:
        watchfiles_logger.setLevel(logging.WARNING)

    return root_logger
