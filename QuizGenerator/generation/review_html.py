from __future__ import annotations

import base64
import html
import json
import logging
from datetime import datetime
from io import BytesIO

import QuizGenerator.generation.contentast as ca

log = logging.getLogger(__name__)


def _inline_image_data_uri(img_data: BytesIO) -> str:
  """Embed an image as a data URI for a standalone HTML document."""
  try:
    pos = img_data.tell()
  except Exception:
    pos = None

  try:
    img_data.seek(0)
    raw = img_data.read()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
  finally:
    if pos is not None:
      try:
        img_data.seek(pos)
      except Exception:
        pass


def _normalize_expected_answers(answer: ca.Answer) -> list[str]:
  accepted: list[str] = []
  seen: set[str] = set()

  try:
    for canvas_answer in answer.get_for_canvas():
      if canvas_answer.get("answer_weight", 0) <= 0:
        continue
      candidate = " ".join(str(canvas_answer.get("answer_text", "")).split())
      if candidate not in seen:
        seen.add(candidate)
        accepted.append(candidate)
  except Exception as exc:
    log.warning("Failed to derive expected answers for review HTML: %s", exc)

  if not accepted:
    fallback = answer.get_display_string() if hasattr(answer, "get_display_string") else str(getattr(answer, "value", ""))
    accepted.append(" ".join(str(fallback).split()))

  return accepted


def _render_answer_key(answer: ca.Answer, index: int) -> str:
  label = answer.label.strip() if getattr(answer, "label", "") else ""
  unit = answer.unit.strip() if getattr(answer, "unit", "") else ""
  accepted = _normalize_expected_answers(answer)
  accepted_html = ", ".join(f"<code>{html.escape(item)}</code>" for item in accepted)
  label_html = html.escape(label) if label else f"Blank {index}"
  unit_html = f" <span class=\"review-unit\">{html.escape(unit)}</span>" if unit else ""
  return (
    "<li>"
    f"<strong>{label_html}</strong>"
    f"{unit_html}"
    f": {accepted_html}"
    "</li>"
  )


def _render_question_card(question: ca.Question, index: int) -> str:
  body_html = question.body.render(
    "standalone_html",
    review_mode=True,
    upload_func=_inline_image_data_uri,
  )
  explanation_html = question.explanation.render(
    "standalone_html",
    upload_func=_inline_image_data_uri,
  )

  answers = list(getattr(question, "answers", []) or [])
  answer_key_html = ""
  if answers:
    answer_items = [
      _render_answer_key(answer, i + 1)
      for i, answer in enumerate(answers)
    ]
    answer_key_html = (
      "<section class=\"review-section\">"
      "<h3>Accepted Answers</h3>"
      f"<ol class=\"answer-key\">{''.join(answer_items)}</ol>"
      "</section>"
    )

  explanation_block = ""
  if explanation_html.strip():
    explanation_block = (
      "<section class=\"review-section\">"
      "<h3>Walkthrough</h3>"
      f"{explanation_html}"
      "</section>"
    )

  question_number = getattr(question, "question_number", None) or index
  points = getattr(question, "value", 1)
  points_label = "point" if float(points) == 1 else "points"
  title = html.escape(getattr(question, "name", f"Question {question_number}") or f"Question {question_number}")

  return (
    f"<section class=\"review-question\" data-question-number=\"{question_number}\">"
    "<header class=\"review-question-header\">"
    f"<div class=\"review-question-title\">Question {question_number}: {title}</div>"
    f"<div class=\"review-question-points\">{points} {points_label}</div>"
    "</header>"
    f"<div class=\"review-question-body\">{body_html}</div>"
    "<div class=\"review-question-controls\">"
    f"<button type=\"button\" class=\"recheck-button\" data-action=\"recheck\">Check this question</button>"
    "</div>"
    "<details class=\"review-spoilers\">"
    "<summary>Spoilers</summary>"
    f"{answer_key_html}"
    f"{explanation_block}"
    "</details>"
    "</section>"
  )


def render_review_html_document(
  quiz_doc: ca.Document,
  *,
  source_pdf_name: str | None = None,
  copy_index: int | None = None,
  total_copies: int | None = None,
) -> str:
  """Render a self-contained practice-review HTML document."""
  title = html.escape(getattr(quiz_doc, "title", "") or "Quiz Review")
  rendered_parts: list[str] = []

  for index, element in enumerate(getattr(quiz_doc, "elements", []), start=1):
    if isinstance(element, ca.Question):
      rendered_parts.append(_render_question_card(element, index))
      continue
    rendered_parts.append(element.render("html", upload_func=_inline_image_data_uri))

  copy_label = ""
  if copy_index is not None and total_copies is not None:
    copy_label = f"Copy {copy_index} of {total_copies}"
  elif copy_index is not None:
    copy_label = f"Copy {copy_index}"

  pdf_link_html = ""
  if source_pdf_name:
    pdf_link_html = (
      f'<a class="pdf-link" href="{html.escape(source_pdf_name)}">'
      "Open matching PDF"
      "</a>"
    )

  generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f5ef;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d8d4ca;
      --accent: #0f766e;
      --good: #15803d;
      --bad: #b91c1c;
      --soft-good: #dcfce7;
      --soft-bad: #fee2e2;
      --shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #fff8e7, var(--bg) 40%, #eef2ff 100%);
      color: var(--ink);
      line-height: 1.55;
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}
    .hero {{
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(216, 212, 202, 0.8);
      border-radius: 20px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.8rem, 4vw, 2.8rem);
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .hero-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-top: 16px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 18px 0 28px;
    }}
    .toolbar button, .pdf-link {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 999px;
      padding: 0.8rem 1.1rem;
      font: inherit;
      text-decoration: none;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(17, 24, 39, 0.05);
    }}
    .toolbar button:hover, .pdf-link:hover {{
      border-color: var(--accent);
    }}
    .quiz-form {{
      display: grid;
      gap: 18px;
    }}
    .review-question {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .review-question-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      border-bottom: 1px solid rgba(216, 212, 202, 0.7);
      padding-bottom: 10px;
      margin-bottom: 16px;
    }}
    .review-question-title {{
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .review-question-points {{
      color: var(--muted);
      white-space: nowrap;
      font-size: 0.95rem;
    }}
    .review-question-body {{
      display: grid;
      gap: 12px;
    }}
    .review-question-body img,
    .review-section img {{
      max-width: min(85%, 720px);
      max-height: 320px;
      width: auto;
      height: auto;
      object-fit: contain;
    }}
    .review-question-body figure,
    .review-section figure {{
      margin: 0.75rem 0;
    }}
    .review-question-body figcaption,
    .review-section figcaption {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .review-question-body p {{
      margin: 0 0 0.5rem;
    }}
    .review-question-body pre,
    .review-section pre {{
      margin: 0.5rem 0 1rem;
      padding: 0.85rem 1rem;
      border-radius: 12px;
      background: #f8fafc;
      border: 1px solid #dbe4f0;
      overflow-x: auto;
      white-space: pre-wrap;
    }}
    .review-question-body code,
    .review-section code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.95em;
    }}
    .review-question-body table,
    .review-section table {{
      width: 100%;
      max-width: 100%;
      border-collapse: collapse;
      margin: 0.5rem 0 1rem;
    }}
    .review-question-body td,
    .review-question-body th,
    .review-section td,
    .review-section th {{
      padding: 0.45rem 0.6rem;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: normal;
    }}
    .quizgen-answer-field {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
      width: 100%;
      margin: 0.1rem 0;
    }}
    .quizgen-answer-label,
    .quizgen-answer-unit {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .quizgen-answer-input {{
      border: 2px solid var(--line);
      border-radius: 10px;
      padding: 0.55rem 0.75rem;
      min-height: 2.5rem;
      min-width: 20rem;
      background: #fff;
      color: var(--ink);
      transition: border-color 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
    }}
    .quizgen-answer-input:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }}
    .quizgen-answer-input.is-correct {{
      border-color: var(--good);
      background: var(--soft-good);
    }}
    .quizgen-answer-input.is-incorrect {{
      border-color: var(--bad);
      background: var(--soft-bad);
    }}
    .quizgen-checkbox-list {{
      display: grid;
      gap: 0.5rem;
      margin: 0.5rem 0 1rem;
    }}
    .quizgen-checkbox-option {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.7rem 0.85rem;
      border: 2px solid var(--line);
      border-radius: 12px;
      background: #fff;
      transition: border-color 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
    }}
    .quizgen-checkbox-option input {{
      width: 1.1rem;
      height: 1.1rem;
      margin: 0;
    }}
    .quizgen-checkbox-option code {{
      background: #f3f4f6;
      padding: 0.12rem 0.35rem;
      border-radius: 6px;
    }}
    .quizgen-checkbox-option.is-correct {{
      border-color: var(--good);
      background: var(--soft-good);
    }}
    .quizgen-checkbox-option.is-incorrect {{
      border-color: var(--bad);
      background: var(--soft-bad);
    }}
    .quizgen-feedback {{
      min-width: 3.5rem;
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .quizgen-feedback.is-correct {{
      color: var(--good);
    }}
    .quizgen-feedback.is-incorrect {{
      color: var(--bad);
    }}
    .review-question.submitted {{
      border-color: rgba(15, 118, 110, 0.2);
    }}
    .review-question.all-correct {{
      box-shadow: 0 16px 40px rgba(21, 128, 61, 0.1);
    }}
    .review-question-controls {{
      display: flex;
      justify-content: flex-end;
      margin-top: 14px;
    }}
    .recheck-button {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .review-spoilers {{
      margin-top: 14px;
      border-top: 1px dashed var(--line);
      padding-top: 12px;
    }}
    .review-spoilers summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
    }}
    .review-section {{
      margin-top: 14px;
    }}
    .review-section h3 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}
    .answer-warnings {{
      margin-top: 1rem;
      padding: 0.85rem 1rem;
      border: 1px solid rgba(216, 212, 202, 0.85);
      border-radius: 14px;
      background: #fff8e7;
    }}
    .answer-warnings p {{
      margin: 0 0 0.4rem;
    }}
    .answer-warnings ul {{
      margin: 0;
      padding-left: 1.25rem;
    }}
    .answer-warnings li {{
      margin: 0.25rem 0;
    }}
    .answer-key {{
      margin: 0;
      padding-left: 1.25rem;
    }}
    .answer-key li {{
      margin-bottom: 0.45rem;
    }}
    .answer-key code {{
      background: #f3f4f6;
      padding: 0.12rem 0.35rem;
      border-radius: 6px;
    }}
    .question-sep {{
      height: 1px;
      background: var(--line);
    }}
    .notice {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    @media (max-width: 700px) {{
      .review-question-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .quizgen-feedback {{
        width: 100%;
      }}
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  <div class="page">
    <header class="hero">
      <h1>{title}</h1>
      <p>Self-check practice review</p>
      <div class="hero-row">
        <div class="notice">{html.escape(copy_label) if copy_label else ""}</div>
        <div class="notice">{html.escape(generated_at)}</div>
      </div>
      <div class="toolbar">
        <button type="button" id="quizgen-check-all">Check Answers</button>
        <button type="button" id="quizgen-reset">Reset</button>
        {pdf_link_html}
      </div>
      <p class="notice">Enter your answers, then check each blank. Open the spoilers for the walkthrough after you try the problem.</p>
    </header>

    <form class="quiz-form" id="quizgen-review-form">
      {''.join(rendered_parts)}
    </form>
  </div>

  <script>
    (() => {{
      const normalize = (value) => (value ?? '').toString().trim().replace(/\\s+/g, ' ');

      const gradeInput = (input) => {{
        let accepted = [];
        try {{
          accepted = JSON.parse(input.dataset.accepted || '[]');
        }} catch (error) {{
          accepted = [];
        }}

        const actual = normalize(input.value);
        const ok = accepted.some((candidate) => normalize(candidate) === actual);
        const feedback = input.parentElement?.querySelector('.quizgen-feedback');

        input.classList.toggle('is-correct', ok);
        input.classList.toggle('is-incorrect', !ok);
        if (feedback) {{
          feedback.textContent = ok ? 'Correct' : 'Wrong';
          feedback.classList.toggle('is-correct', ok);
          feedback.classList.toggle('is-incorrect', !ok);
        }}
        return ok;
      }};

      const gradeCheckboxOption = (option) => {{
        const input = option.querySelector('input[type="checkbox"]');
        if (!(input instanceof HTMLInputElement)) {{
          return true;
        }}

        const shouldBeChecked = option.dataset.correct === 'true';
        const ok = input.checked === shouldBeChecked;
        const feedback = option.querySelector('.quizgen-feedback');

        option.classList.toggle('is-correct', ok);
        option.classList.toggle('is-incorrect', !ok);
        if (feedback) {{
          feedback.textContent = ok ? 'Correct' : 'Wrong';
          feedback.classList.toggle('is-correct', ok);
          feedback.classList.toggle('is-incorrect', !ok);
        }}
        return ok;
      }};

      const gradeQuestion = (question) => {{
        const inputs = Array.from(question.querySelectorAll('.quizgen-answer-input'));
        const checkboxOptions = Array.from(question.querySelectorAll('.quizgen-checkbox-option'));
        let allCorrect = true;
        for (const input of inputs) {{
          const ok = gradeInput(input);
          if (!ok) {{
            allCorrect = false;
          }}
        }}
        for (const option of checkboxOptions) {{
          const ok = gradeCheckboxOption(option);
          if (!ok) {{
            allCorrect = false;
          }}
        }}
        question.classList.add('submitted');
        question.classList.toggle('all-correct', allCorrect);
        return allCorrect;
      }};

      const gradeAll = () => {{
        const questions = Array.from(document.querySelectorAll('.review-question'));
        questions.forEach((question) => gradeQuestion(question));
      }};

      const resetAll = () => {{
        document.querySelectorAll('.quizgen-answer-input').forEach((input) => {{
          input.value = '';
          input.classList.remove('is-correct', 'is-incorrect');
          const feedback = input.parentElement?.querySelector('.quizgen-feedback');
          if (feedback) {{
            feedback.textContent = '';
            feedback.classList.remove('is-correct', 'is-incorrect');
          }}
        }});
        document.querySelectorAll('.quizgen-checkbox-option').forEach((option) => {{
          const input = option.querySelector('input[type="checkbox"]');
          if (input instanceof HTMLInputElement) {{
            input.checked = false;
          }}
          option.classList.remove('is-correct', 'is-incorrect');
          const feedback = option.querySelector('.quizgen-feedback');
          if (feedback) {{
            feedback.textContent = '';
            feedback.classList.remove('is-correct', 'is-incorrect');
          }}
        }});
        document.querySelectorAll('.review-question').forEach((question) => {{
          question.classList.remove('submitted', 'all-correct');
        }});
      }};

      document.getElementById('quizgen-check-all')?.addEventListener('click', gradeAll);
      document.getElementById('quizgen-reset')?.addEventListener('click', resetAll);
      document.addEventListener('click', (event) => {{
        const target = event.target;
        if (!(target instanceof HTMLElement)) {{
          return;
        }}
        const button = target.closest('.recheck-button');
        if (!button) {{
          return;
        }}
        const question = button.closest('.review-question');
        if (question) {{
          gradeQuestion(question);
        }}
      }});
      document.getElementById('quizgen-review-form')?.addEventListener('submit', (event) => {{
        event.preventDefault();
        gradeAll();
      }});

      document.addEventListener('input', (event) => {{
        const input = event.target;
        if (!(input instanceof HTMLInputElement)) {{
          return;
        }}
        if (!input.classList.contains('quizgen-answer-input')) {{
          return;
        }}
        if (input.closest('.review-question')?.classList.contains('submitted')) {{
          gradeInput(input);
        }}
      }});
      document.addEventListener('change', (event) => {{
        const input = event.target;
        if (!(input instanceof HTMLInputElement)) {{
          return;
        }}
        if (input.type !== 'checkbox') {{
          return;
        }}
        const option = input.closest('.quizgen-checkbox-option');
        if (option && option.parentElement?.closest('.review-question')?.classList.contains('submitted')) {{
          gradeCheckboxOption(option);
        }}
      }});
    }})();
  </script>
</body>
</html>
"""
