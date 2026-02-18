#!/usr/bin/env python
"""
Typer front-end for quizgen.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

try:
    import typer
except ImportError:  # pragma: no cover - exercised only when typer is missing.
    typer = None  # type: ignore[assignment]


if typer is None:
    def main() -> None:
        raise SystemExit(
            "Typer is not installed. Install with: pip install typer\n"
            "Then run: python -m QuizGenerator.typer_cli --help"
        )
else:
    from dotenv import load_dotenv

    from lms_interface.canvas_interface import CanvasInterface
    from QuizGenerator.contentast import Answer
    from QuizGenerator.generate import (
        QuizGenError,
        _check_dependencies,
        _get_cli_version,
        explain_registered_tags,
        generate_practice_quizzes,
        generate_quiz,
        list_registered_tags,
        test_all_questions,
    )
    from QuizGenerator.performance import PerformanceTracker

    app = typer.Typer(
        add_completion=True,
        no_args_is_help=True,
        help="QuizGenerator CLI.",
    )
    tags_app = typer.Typer(
        no_args_is_help=True,
        help="Inspect tag coverage and classification for registered questions.",
    )
    app.add_typer(tags_app, name="tags")

    def _version_callback(value: bool) -> None:
        if not value:
            return
        typer.echo(f"quizgen {_get_cli_version()}")
        raise typer.Exit()

    @app.callback()
    def _app_callback(
        version: bool = typer.Option(
            False,
            "--version",
            is_eager=True,
            callback=_version_callback,
            help="Show version and exit.",
        ),
    ) -> None:
        del version

    def _enable_debug_logging() -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        for logger_name in ["QuizGenerator", "lms_interface", "__main__"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            for handler in logger.handlers:
                handler.setLevel(logging.DEBUG)

    @contextmanager
    def _quizgen_error_boundary():
        try:
            yield
        except QuizGenError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    def _configure_runtime(
        *,
        env: str,
        debug: bool,
        allow_generator: bool = False,
        float_tolerance: float | None = None,
        max_backoff_attempts: int | None = None,
    ) -> None:
        load_dotenv(env)
        if debug:
            _enable_debug_logging()
        if allow_generator:
            os.environ["QUIZGEN_ALLOW_GENERATOR"] = "1"
        if float_tolerance is not None:
            if float_tolerance < 0:
                raise QuizGenError("float_tolerance must be non-negative.")
            Answer.DEFAULT_FLOAT_TOLERANCE = float_tolerance
        if max_backoff_attempts is not None and max_backoff_attempts < 1:
            raise QuizGenError("max_backoff_attempts must be >= 1.")

    def _ensure_dependencies(*, use_typst: bool) -> None:
        ok, missing = _check_dependencies(
            require_typst=use_typst,
            require_latex=not use_typst,
        )
        if not ok:
            raise QuizGenError("\n".join(missing))

    @app.command("generate")
    def generate_command(
        quiz_yaml: str = typer.Option(..., "--yaml", help="Path to quiz YAML configuration."),
        num_pdfs: int = typer.Option(0, "--num-pdfs", min=0, help="How many PDF quizzes to create."),
        num_canvas: int = typer.Option(
            0, "--num-canvas", min=0, help="How many variations to upload to Canvas."
        ),
        seed: int | None = typer.Option(None, "--seed", help="Random seed for quiz generation."),
        env: str = typer.Option(str(Path.home() / ".env"), "--env", help="Path to .env file."),
        debug: bool = typer.Option(False, "--debug", help="Set logging level to debug."),
        prod: bool = typer.Option(False, "--prod", help="Use production Canvas environment."),
        course_id: int | None = typer.Option(None, "--course-id", help="Canvas course ID."),
        delete_assignment_group: bool = typer.Option(
            False,
            "--delete-assignment-group",
            help="Delete existing assignment group before upload.",
        ),
        quiet: bool = typer.Option(False, "--quiet", help="Disable progress bars."),
        latex: bool = typer.Option(False, "--latex", help="Use LaTeX instead of Typst."),
        typst_measurement: bool = typer.Option(
            False, "--typst-measurement", help="Enable Typst measurement."
        ),
        consistent_pages: bool = typer.Option(
            False, "--consistent-pages", help="Keep page counts consistent."
        ),
        layout_samples: int = typer.Option(10, "--layout-samples", min=1, help="Layout samples."),
        layout_safety_factor: float = typer.Option(
            1.1, "--layout-safety-factor", min=0.1, help="Safety multiplier for heights."
        ),
        optimize_space: bool = typer.Option(
            False, "--optimize-space", help="Optimize question ordering."
        ),
        embed_images_typst: bool = typer.Option(
            True,
            "--embed-images-typst/--no-embed-images-typst",
            help="Embed images in Typst output.",
        ),
        show_pdf_aids: bool = typer.Option(
            True,
            "--pdf-aids/--no-pdf-aids",
            help="Render optional PDF scaffolding aids.",
        ),
        allow_generator: bool = typer.Option(
            False,
            "--allow-generator",
            help="Enable FromGenerator questions (executes Python from YAML).",
        ),
        max_backoff_attempts: int | None = typer.Option(
            None, "--max-backoff-attempts", min=1, help="Max attempts for backoff."
        ),
        float_tolerance: float | None = typer.Option(
            None, "--float-tolerance", min=0.0, help="Default float answer tolerance."
        ),
    ) -> None:
        with _quizgen_error_boundary():
            _configure_runtime(
                env=env,
                debug=debug,
                allow_generator=allow_generator,
                float_tolerance=float_tolerance,
                max_backoff_attempts=max_backoff_attempts,
            )
            if num_canvas > 0 and course_id is None:
                raise QuizGenError("Missing --course-id for Canvas upload. Example: --course-id 12345")
            use_typst = not latex
            if num_pdfs > 0:
                _ensure_dependencies(use_typst=use_typst)
            PerformanceTracker.clear_metrics()
            generate_quiz(
                quiz_yaml,
                num_pdfs=num_pdfs,
                num_canvas=num_canvas,
                use_prod=prod,
                course_id=course_id,
                delete_assignment_group=delete_assignment_group,
                use_typst=use_typst,
                use_typst_measurement=typst_measurement,
                base_seed=seed,
                env_path=env,
                consistent_pages=consistent_pages or num_pdfs > 1,
                layout_samples=layout_samples,
                layout_safety_factor=layout_safety_factor,
                embed_images_typst=embed_images_typst,
                show_pdf_aids=show_pdf_aids,
                optimize_layout=optimize_space,
                max_backoff_attempts=max_backoff_attempts,
                quiet=quiet,
            )

    @app.command("practice")
    def practice_command(
        tags: list[str] = typer.Argument(..., metavar="TAG", help="Tag filters, e.g. course:cst334 topic:memory."),
        course_id: int = typer.Option(..., "--course-id", help="Canvas course ID."),
        practice_variations: int = typer.Option(
            5, "--practice-variations", min=1, help="Variations per group."
        ),
        practice_question_groups: int = typer.Option(
            5, "--practice-question-groups", min=1, help="Groups per question."
        ),
        practice_points: float = typer.Option(
            1.0, "--practice-points", min=0.0, help="Points per practice question."
        ),
        practice_match: Literal["any", "all"] = typer.Option(
            "any", "--practice-match", help="Tag matching mode: any|all."
        ),
        practice_tag_source: Literal["explicit", "merged", "derived"] = typer.Option(
            "merged",
            "--practice-tag-source",
            help="Tag source: explicit|merged|derived.",
        ),
        practice_assignment_group: str = typer.Option(
            "practice",
            "--practice-assignment-group",
            help="Assignment group name for created quizzes.",
        ),
        env: str = typer.Option(str(Path.home() / ".env"), "--env", help="Path to .env file."),
        debug: bool = typer.Option(False, "--debug", help="Set logging level to debug."),
        prod: bool = typer.Option(False, "--prod", help="Use production Canvas environment."),
        delete_assignment_group: bool = typer.Option(
            False,
            "--delete-assignment-group",
            help="Delete existing assignment group before upload.",
        ),
        quiet: bool = typer.Option(False, "--quiet", help="Disable progress bars."),
        allow_generator: bool = typer.Option(
            False,
            "--allow-generator",
            help="Enable FromGenerator questions (executes Python from YAML).",
        ),
        max_backoff_attempts: int | None = typer.Option(
            None, "--max-backoff-attempts", min=1, help="Max attempts for backoff."
        ),
        float_tolerance: float | None = typer.Option(
            None, "--float-tolerance", min=0.0, help="Default float answer tolerance."
        ),
    ) -> None:
        with _quizgen_error_boundary():
            _configure_runtime(
                env=env,
                debug=debug,
                allow_generator=allow_generator,
                float_tolerance=float_tolerance,
                max_backoff_attempts=max_backoff_attempts,
            )
            generate_practice_quizzes(
                tag_filters=tags,
                course_id=course_id,
                use_prod=prod,
                env_path=env,
                num_variations=practice_variations,
                question_groups=practice_question_groups,
                points_value=practice_points,
                delete_assignment_group=delete_assignment_group,
                assignment_group_name=practice_assignment_group,
                match_all=(practice_match == "all"),
                tag_source=practice_tag_source,
                quiet=quiet,
                max_backoff_attempts=max_backoff_attempts,
            )

    @app.command("test")
    def test_command(
        num_variations: int = typer.Argument(..., metavar="N", min=1, help="Variations per registered question type."),
        test_questions: list[str] | None = typer.Option(
            None,
            "--test-question",
            "--test-questions",
            "-q",
            help="Only test specific question types. Use multiple times.",
        ),
        strict: bool = typer.Option(False, "--strict", help="Skip Canvas upload if any question type fails."),
        seed: int | None = typer.Option(None, "--seed", help="Base random seed."),
        skip_missing_extras: bool = typer.Option(
            False,
            "--skip-missing-extras",
            help="Skip questions that fail due to missing optional dependencies.",
        ),
        env: str = typer.Option(str(Path.home() / ".env"), "--env", help="Path to .env file."),
        debug: bool = typer.Option(False, "--debug", help="Set logging level to debug."),
        prod: bool = typer.Option(False, "--prod", help="Use production Canvas environment."),
        course_id: int | None = typer.Option(None, "--course-id", help="Canvas course ID."),
        latex: bool = typer.Option(False, "--latex", help="Use LaTeX instead of Typst."),
        allow_generator: bool = typer.Option(
            False,
            "--allow-generator",
            help="Enable FromGenerator questions (executes Python from YAML).",
        ),
        embed_images_typst: bool = typer.Option(
            True,
            "--embed-images-typst/--no-embed-images-typst",
            help="Embed images in Typst output.",
        ),
        show_pdf_aids: bool = typer.Option(
            True,
            "--pdf-aids/--no-pdf-aids",
            help="Render optional PDF scaffolding aids.",
        ),
        float_tolerance: float | None = typer.Option(
            None, "--float-tolerance", min=0.0, help="Default float answer tolerance."
        ),
    ) -> None:
        with _quizgen_error_boundary():
            _configure_runtime(
                env=env,
                debug=debug,
                allow_generator=allow_generator,
                float_tolerance=float_tolerance,
            )
            use_typst = not latex
            _ensure_dependencies(use_typst=use_typst)
            canvas_course = None
            if course_id:
                canvas_interface = CanvasInterface(prod=prod, env_path=env)
                canvas_course = canvas_interface.get_course(course_id=course_id)
            success = test_all_questions(
                num_variations,
                generate_pdf=True,
                use_typst=use_typst,
                canvas_course=canvas_course,
                strict=strict,
                question_filter=test_questions,
                skip_missing_extras=skip_missing_extras,
                embed_images_typst=embed_images_typst,
                show_pdf_aids=show_pdf_aids,
                seed=seed,
            )
            if not success:
                raise QuizGenError("One or more questions failed during test mode.")

    @app.command("deps")
    def deps_command(
        env: str = typer.Option(str(Path.home() / ".env"), "--env", help="Path to .env file."),
        debug: bool = typer.Option(False, "--debug", help="Set logging level to debug."),
        latex: bool = typer.Option(False, "--latex", help="Use LaTeX checks instead of Typst checks."),
    ) -> None:
        with _quizgen_error_boundary():
            _configure_runtime(env=env, debug=debug)
            _ensure_dependencies(use_typst=not latex)
            typer.echo("Dependency check passed.")

    @tags_app.command("list")
    def tags_list_command(
        tag_source: Literal["explicit", "merged", "derived"] = typer.Option(
            "merged", "--tag-source", help="Tag source: explicit|merged|derived."
        ),
        include_questions: bool = typer.Option(
            False, "--include-questions", help="Include per-question tag lines."
        ),
        only_missing_explicit: bool = typer.Option(
            False,
            "--only-missing-explicit",
            help="Show only question types without explicit tags.",
        ),
        tag_filter: list[str] | None = typer.Option(
            None,
            "--filter",
            help="Optional tag filter. Use multiple times.",
        ),
        env: str = typer.Option(str(Path.home() / ".env"), "--env", help="Path to .env file."),
        debug: bool = typer.Option(False, "--debug", help="Set logging level to debug."),
    ) -> None:
        with _quizgen_error_boundary():
            _configure_runtime(env=env, debug=debug)
            list_registered_tags(
                tag_source=tag_source,
                include_questions=include_questions,
                only_missing_explicit=only_missing_explicit,
                tag_filter=tag_filter,
            )

    @tags_app.command("explain")
    def tags_explain_command(
        query: str = typer.Argument(..., help="Substring to match against question names."),
        limit: int = typer.Option(20, "--limit", min=1, help="Maximum number of matches to print."),
        env: str = typer.Option(str(Path.home() / ".env"), "--env", help="Path to .env file."),
        debug: bool = typer.Option(False, "--debug", help="Set logging level to debug."),
    ) -> None:
        with _quizgen_error_boundary():
            _configure_runtime(env=env, debug=debug)
            explain_registered_tags(query, limit=limit)

    def main() -> None:
        app()


if __name__ == "__main__":
    main()
