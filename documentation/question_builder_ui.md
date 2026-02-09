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

## Derived variables (advanced)

Derived values can be declared as a list to mix simple expressions and
Python snippets in order:

```yaml
context:
  derived:
    - name: bytes
      expr: "2**bits"
    - name: pages
      python: |
        result = bytes // 4096
```

Python entries must set `result` (for a named derived value) or use `set(name, value)`
to write multiple values.

## Additional YAML nodes

The YAML parser supports additional content AST nodes beyond the UI surface.
These can be authored directly in YAML and will render correctly:

- `matrix`
- `math_expression`
- `picture`
- `line_break`
- `table_group`
- `repeated_problem_part`
- `only_latex`
- `only_html`
- `choose`

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
