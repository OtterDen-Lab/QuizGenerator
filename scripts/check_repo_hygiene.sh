#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Keep uv cache under the repo (or caller-provided path) so checks do not
# depend on a writable home cache directory.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv_cache}"

echo "Running Ruff checks..."
if command -v ruff >/dev/null 2>&1; then
  ruff check --output-format=github .
else
  uv run ruff check --output-format=github .
fi

echo "Checking docs/workflows for removed CLI forms..."
if rg -n \
  -e 'quizgen --yaml' \
  -e '--generate_practice' \
  -e '--test_all' \
  -e '--check-deps' \
  README.md documentation .github/workflows; then
  echo "Found removed CLI syntax in docs/workflows. Please migrate to subcommands."
  exit 1
fi

echo "Repository hygiene checks passed."
