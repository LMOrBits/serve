from loguru import logger

import subprocess

from pathlib import Path


class TaskCLI:
    def __init__(self, taskfile_dir: str = None):
        """
        CLI wrapper for Go Task with argparse-like behavior.

        :param taskfile_dir: Directory where Taskfile.yml is located.
        """
        self.taskfile_dir = taskfile_dir or str(Path(__file__).parent)
        

    def run(self, task_name: str, **kwargs):
        """
        Run a Task command with named arguments.

        :param task_name: Name of the task to run.
        :param kwargs: Named arguments passed as key-value pairs.
        """
        # Convert kwargs to Task variable format
        task_args = [f"{key.upper()}={value}" for key, value in kwargs.items() if value is not None]
        # Build command
        command = ["task", "--dir", self.taskfile_dir, task_name] + task_args
        logger.debug(command)
        # Run the Task command
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stderr:
            logger.error(result.stderr)
        if result.stdout:
            logger.info(result.stdout)
        return result



