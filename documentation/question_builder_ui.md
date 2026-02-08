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

## Security note

YAML expressions and `{{ }}` templates are evaluated with Python `eval` in
`QuizGenerator/yaml_question.py`. Use trusted input only and keep the UI
local (do not expose it to the public internet).
