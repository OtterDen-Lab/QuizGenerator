# QuizGenerator 1.0.0 Release Checklist

## 🚨 BLOCKERS - Must fix before release

- [x] **Canvas upload infinite retry loop** (`lms_interface/canvas_interface.py`) ✅ FIXED
  Added max retry count and exponential backoff to prevent hanging on persistent API errors.

- [x] **Global `args` variable reference** (`generate.py:478`) ✅ FIXED
  Added `embed_images_typst` parameter to `generate_quiz()` signature.

- [x] **QuestionGroup breaks determinism** (`question.py:1059-1064`) ✅ FIXED
  Fixed `QuestionGroup.instantiate()` to properly return `QuestionInstance` from selected question.
  Simplified `__getattr__` to remove buggy re-instantiation logic.

- [x] **Layout optimization breaks determinism** (`quiz.py:194-267`) ✅ VERIFIED OK
  Already uses `deterministic_seeds()` based on question class/name hash - no fix needed.

- [x] **FromGenerator security documentation** ✅ FIXED
  Added prominent stderr warning when FromGenerator is used, plus Security section in README.

## ⚠️ MAJOR - Strongly recommend before release

- [x] **Fix example YAML** (`example_files/example_exam.yaml`) ✅ FIXED
  Removed `FromGenerator` usage so first-run experience does not require `--allow_generator`.

- [ ] **Add integration tests**
  Only 4 tests with ~100 lines. Need tests for: Canvas uploads, PDF compilation, YAML parsing, error paths, question groups.

- [x] **Canvas variation deduplication scope** (`generate.py`) ✅ FIXED
  Dedup moved to QuizGenerator and is now per-question during payload generation.

- [x] **LaTeX/Typst pipe deadlock** (`generate.py:401-408`) ✅ FIXED
  Replaced `wait()` with `communicate()` to avoid PIPE deadlocks.

- [x] **FromGenerator error handling** (`basic.py:157-160`) ✅ FIXED
  Now raises `RuntimeError` with generator code context instead of `exit(8)`.

## 📝 MINOR - Should address

- [x] **Spacing preset docstring mismatch** (`question.py:82`) ✅ FIXED
  Updated `parse_spacing()` docstring to match `SPACING_PRESETS`.

- [x] **Python 3.12 requirement** (`pyproject.toml:15`) ✅ FIXED
  Standardized minimum to `>=3.12` for modern tooling and simplified compatibility.

- [x] **Canvas credential preflight check** ✅ FIXED
  Validate API keys on `CanvasInterface` init with a clear error if missing.

- [x] **Log file location** ✅ FIXED
  Logs go to `out/logs/` by default and remain configurable via env vars.

- [x] **Type hints in core APIs** ✅ FIXED
  Added core API annotations for `Quiz` and `QuestionRegistry`.

- [x] **QR code temp file cleanup** ✅ FIXED
  Migrated QR codes to SVG and cleaned up `.svg` artifacts.

- [x] **Remove empty `TrueFalse` class** (`basic.py:164-165`) ✅ FIXED
  Removed unused placeholder.

- [x] **Dependency pinning strategy** ✅ FIXED
  Documented range + lockfile approach in README.

## 💡 POST-1.0 - Future enhancements

- [ ] **`--preflight` command**
  Check Typst/LaTeX/Pandoc availability and Canvas credentials with actionable errors.

- [ ] **Safe mode for FromGenerator**
  Replace arbitrary Python with Jinja2 or similar template DSL for safer dynamic questions.

- [ ] **Topic balance controls**
  Min/max counts per topic with reporting in `quiz.describe()`.

- [ ] **Progress bars for Canvas uploads**
  Visual feedback for long-running operations.

- [ ] **Parallel question generation**
  Currently sequential; could speed up large exams.

- [ ] **LMS abstraction layer**
  Support for Moodle, Blackboard beyond Canvas.

- [ ] **Web UI wrapper**
  Flask/FastAPI interface for non-CLI users.

- [ ] **Question bank versioning**
  Track question changes across semesters.

---

## Quick Fixes (< 30 min each)

| Issue | File | Fix |
|-------|------|-----|
| Global `args` | `generate.py:478` | Pass as parameter to `generate_quiz()` |
| Example YAML | `example_exam.yaml` | ✅ Fixed (removed `FromGenerator` dependency) |
| Empty class | `basic.py:164` | ✅ Fixed (removed `TrueFalse`) |
| Docstring | `question.py:82` | ✅ Fixed (examples match presets) |

## Medium Fixes (1-2 hours each)

| Issue | File | Fix |
|-------|------|-----|
| Canvas retries | `lms_interface/canvas_interface.py` | ✅ Fixed (bounded retries + backoff) |
| Pipe deadlock | `generate.py:401` | ✅ Fixed (use `communicate()`) |
| Error masking | `basic.py:157` | ✅ Fixed (raise exception instead of `exit(8)`) |
| Security docs | `README.md` | Add FromGenerator warning section |

## Complex Fixes (2-4 hours each)

| Issue | File | Fix |
|-------|------|-----|
| QuestionGroup seed | `question.py:1059` | Pass seed through `__getattr__`, defer selection |
| Layout determinism | `quiz.py:194` | Cache heights by question class, use deterministic seeds |
| Test coverage | `tests/` | Add 5-10 integration tests for critical paths |
