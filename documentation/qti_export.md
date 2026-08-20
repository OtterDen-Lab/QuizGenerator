# Experimental Canvas QTI export

Use `quizgen generate --yaml quiz.yaml --qti-variations N` to write a Canvas Classic Quiz QTI 1.2 ZIP in `out/`. Canvas uses a QTI 1.2 dialect for imports; generic QTI 2.1 choice interactions are otherwise imported as Multiple Answers.

The prototype supports multiple-choice, dropdown, fill-in-the-blank (including numerical tolerance), and matching interactions. Dropdowns use Canvas's `multiple_dropdowns_question` metadata, bracket placeholders, and `response_lid` answer mappings. It rejects essays and mixed interactions before producing a package.

Each generated variation is currently imported as an individual Canvas question. Verify the resulting package in the target Canvas instance before using it for an assessment.
