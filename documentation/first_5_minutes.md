# First 5 Minutes

Goal: get a PDF quiz from scratch with minimal setup.

## 1. Install

```bash
pip install QuizGenerator
```

## 2. Create a YAML file

Create `my_quiz.yaml`:

```yaml
name: "Intro Quiz"
question_order: yaml
questions:
  - name: "Warmup"
    points: 2
    class: FromText
    kwargs:
      text: "What is 2 + 2?"
  - name: "Memory Paging"
    points: 5
    class: PagingQuestion
```

This list format preserves the order you write.

## 3. Generate a PDF

```bash
quizgen generate --yaml my_quiz.yaml --num-pdfs 1
```

Your PDF is written to `out/`.

## 4. Optional: Optimize layout

To reorder questions and reduce page count:

```bash
quizgen generate --yaml my_quiz.yaml --num-pdfs 1 --optimize-space
```

This also changes Canvas order if you upload.
