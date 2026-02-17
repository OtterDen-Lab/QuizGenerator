
import logging.config
import os
import re
from pathlib import Path

import yaml


def _env_flag(name: str, *, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _remove_file_handlers(config: dict) -> None:
  handlers = config.get("handlers", {})
  file_handler_names = {
    handler_name
    for handler_name, handler_config in handlers.items()
    if handler_config.get("class") == "logging.FileHandler"
  }
  if not file_handler_names:
    return

  for handler_name in file_handler_names:
    handlers.pop(handler_name, None)

  root = config.get("root", {})
  root_handlers = root.get("handlers", [])
  root["handlers"] = [name for name in root_handlers if name not in file_handler_names]

  for logger in config.get("loggers", {}).values():
    logger_handlers = logger.get("handlers", [])
    logger["handlers"] = [name for name in logger_handlers if name not in file_handler_names]


def _ensure_file_handler_directories(config: dict) -> None:
  for handler in config.get("handlers", {}).values():
    if handler.get("class") != "logging.FileHandler":
      continue
    filename = handler.get("filename")
    if not filename:
      continue
    log_dir = os.path.dirname(filename)
    if log_dir:
      os.makedirs(log_dir, exist_ok=True)


def _find_project_root() -> Path:
  package_dir = Path(__file__).resolve().parent
  for candidate in [package_dir, *package_dir.parents]:
    if (candidate / "pyproject.toml").exists():
      return candidate
  return package_dir.parent


def _anchor_file_handler_paths(config: dict, *, base_dir: Path) -> None:
  for handler in config.get("handlers", {}).values():
    if handler.get("class") != "logging.FileHandler":
      continue
    filename = handler.get("filename")
    if not filename:
      continue
    path = Path(filename)
    if not path.is_absolute():
      handler["filename"] = str((base_dir / path).resolve())


def setup_logging() -> None:
  config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
  if os.path.exists(config_path):
    with open(config_path, 'r') as f:
      config_text = f.read()
    
    # Process environment variables in the format ${VAR:-default}
    def replace_env_vars(match) -> str:
      var_name = match.group(1)
      default_value = match.group(2)
      return os.environ.get(var_name, default_value)
    
    config_text = re.sub(r'\$\{([^}:]+):-([^}]+)\}', replace_env_vars, config_text)
    config = yaml.safe_load(config_text)

    if _env_flag("QUIZGEN_FILE_LOGGING", default=True):
      _anchor_file_handler_paths(config, base_dir=_find_project_root())
      _ensure_file_handler_directories(config)
    else:
      _remove_file_handlers(config)
    logging.config.dictConfig(config)
  else:
    # Fallback to basic configuration if logging.yaml is not found
    logging.basicConfig(level=logging.INFO)

# Call this once when your application starts
setup_logging()
