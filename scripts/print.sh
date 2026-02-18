#!/bin/bash

set -u

show_help() {
  cat <<'EOF'
Usage: scripts/print.sh [OPTIONS] [DIRECTORY]

Print all PDFs in DIRECTORY (defaults to current directory) via Preview GUI automation.

Options:
  -d, --delete   Delete each PDF after successful print submission
  -y, --yes      Skip confirmation prompt
  -h, --help     Show this help message
EOF
}

DELETE_AFTER_PRINT=0
SKIP_CONFIRM=0
TARGET_DIR="${PWD}"
TARGET_DIR_SET=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    -d|--delete)
      DELETE_AFTER_PRINT=1
      shift
      ;;
    -y|--yes)
      SKIP_CONFIRM=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
    *)
      if [ "$TARGET_DIR_SET" -eq 1 ]; then
        echo "Only one DIRECTORY argument is supported." >&2
        show_help >&2
        exit 1
      fi
      TARGET_DIR="$1"
      TARGET_DIR_SET=1
      shift
      ;;
  esac
done

if ! TARGET_DIR="$(cd "$TARGET_DIR" 2>/dev/null && pwd)"; then
  echo "Directory does not exist or is not accessible: $TARGET_DIR" >&2
  exit 1
fi

shopt -s nullglob
PDF_FILES=("$TARGET_DIR"/*.pdf "$TARGET_DIR"/*.PDF)
shopt -u nullglob

TOTAL=${#PDF_FILES[@]}
echo "Print directory: $TARGET_DIR"
echo "PDF count: $TOTAL"
if [ "$DELETE_AFTER_PRINT" -eq 1 ]; then
  echo "Delete after print: enabled"
else
  echo "Delete after print: disabled"
fi

if [ "$TOTAL" -eq 0 ]; then
  echo "No PDF files found."
  exit 0
fi

if [ "$SKIP_CONFIRM" -ne 1 ]; then
  read -r -p "Proceed with printing? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Canceled."
      exit 0
      ;;
  esac
fi

PRINTED=0
DELETED=0
FAILED=0
INDEX=0

for file in "${PDF_FILES[@]}"; do
  INDEX=$((INDEX + 1))
  echo "[$INDEX/$TOTAL] Printing: $(basename "$file")"

  if ! open -a Preview "$file"; then
    echo "  Failed to open in Preview: $file" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  sleep 1
  if ! osascript <<'EOF'
tell application "System Events"
    tell process "Preview"
        keystroke "p" using command down
        delay 0.8
        keystroke return
    end tell
end tell
EOF
  then
    echo "  Failed to send print keystrokes for: $file" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  sleep 0.8
  osascript -e 'tell application "Preview" to close front window saving no' >/dev/null 2>&1 || true
  PRINTED=$((PRINTED + 1))

  if [ "$DELETE_AFTER_PRINT" -eq 1 ]; then
    if rm -f -- "$file"; then
      DELETED=$((DELETED + 1))
      echo "  Deleted: $(basename "$file")"
    else
      echo "  Failed to delete: $file" >&2
    fi
  fi
done

echo "Done. Submitted: $PRINTED/$TOTAL, Failed: $FAILED, Deleted: $DELETED"
