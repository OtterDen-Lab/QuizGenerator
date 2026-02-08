# Question Builder UI

This is a quick, local-only web UI for building YAML-based questions.

Run the server:

```bash
python scripts/question_builder_ui.py --port 8787
```

Open:

```
http://127.0.0.1:8787
```

The UI exports YAML compatible with `FromYaml` questions.

## PDF export

Use the "Export PDF" button to generate a quiz PDF locally. This uses Typst
and writes output to `out/`. Install Typst and keep the UI local-only.

## Security note

YAML expressions and `{{ }}` templates are evaluated with Python `eval` in
`QuizGenerator/yaml_question.py`. Use trusted input only and keep the UI
local (do not expose it to the public internet).

## Beta smoke checklist

- Create a quiz with 2+ questions, reorder them, and export YAML.
- Import the exported YAML and confirm ordering + points are preserved.
- Preview a premade question and a YAML question (with MathJax).
- Export PDF with `PDF count = 1` (downloads PDF).
- Export PDF with `PDF count > 1` (downloads ZIP).
