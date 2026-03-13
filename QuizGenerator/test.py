#!/usr/bin/env python

import logging
import pathlib
import pprint

import typer

import yaml as pyyaml
import core.classes as classes

from QuizGenerator import enable_debug_logging, is_debug_enabled

log = logging.getLogger(__name__)


def main(
  yaml: pathlib.Path = typer.Option(..., help="YAML config file"),
  debug: bool = False
) -> None:
  
  if is_debug_enabled(debug):
    enable_debug_logging()
  
  log.debug(f"Using problem spec: {yaml}")
  
  with open(yaml) as f:
    data = pyyaml.safe_load(f)
  
  log.debug("\n" + pprint.pformat(data))
  
  classes.Question.from_dict(data)
  


if __name__ == "__main__":
  typer.run(main)
