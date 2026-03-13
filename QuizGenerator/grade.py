#!/usr/bin/env python
import logging
import pathlib

import typer
from dotenv import load_dotenv

from QuizGenerator import enable_debug_logging, is_debug_enabled
from QuizGenerator.grading.grade_exams import Exam

log = logging.getLogger(__name__)



def main(
  path_to_yaml: pathlib.Path,
  directory: str,
  env: str = typer.Option(str(pathlib.Path.home() / ".env"), "--env", help="Path to .env file."),
  debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
  load_dotenv(env)
  if is_debug_enabled(debug):
    enable_debug_logging()
  log.debug(f"Using grading spec: {path_to_yaml}")
  exam = Exam(path_to_yaml, directory)


if __name__ == "__main__":
  typer.run(main)
