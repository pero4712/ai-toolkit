import sys
import os
import subprocess
from toolkit.accelerator import get_accelerator


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        print(*args, **kwargs)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_git_sha():
    try:
        toolkit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=toolkit_dir,
            stderr=subprocess.DEVNULL,
        ).decode('utf-8').strip()
        return sha
    except Exception:
        return 'unknown'


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    sys.stdout = Logger(filename)
    sys.stderr = Logger(filename)
