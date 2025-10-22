#!/bin/bash

DIR="/Users/ssogden/repos/teaching/QuizGeneration/out"

for file in "$DIR"/*.pdf; do
    if [ -f "$file" ]; then
        echo "Printing: $(basename "$file")"
        open -a Preview "$file"
        sleep 1
        osascript <<EOF
tell application "System Events"
    tell process "Preview"
        keystroke "p" using command down
        delay 0.8
        keystroke return
    end tell
end tell
EOF
        sleep 0.8
        osascript -e 'tell application "Preview" to close front window saving no'
    fi
done

echo "Done"