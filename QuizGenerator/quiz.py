#!/usr/bin/env python
from __future__ import annotations

import collections
import hashlib
import logging
import random
import re
from datetime import datetime

import yaml

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionGroup, QuestionRegistry

log = logging.getLogger(__name__)


class Quiz:
  """
  A quiz object that will build up questions and output them in a range of formats (hopefully)
  It should be that a single quiz object can contain multiples -- essentially it builds up from the questions and then can generate a variety of questions.
  """
  
  INTEREST_THRESHOLD = 1.0
  
  name: str
  questions: list[Question | QuestionGroup]
  instructions: str
  description: str | None
  question_sort_order: list[Question.Topic] | None
  practice: bool
  preserve_order_point_values: set[float]
  preserve_yaml_order: bool
  yaml_id: str | None

  def __init__(self, name: str, questions: list[Question | QuestionGroup], practice: bool, *args, **kwargs):
    self.name = name
    self.questions = questions
    self.instructions = kwargs.get("instructions", "")
    self.yaml_id = kwargs.get("yaml_id", None)

    # Parse description with content AST if provided
    raw_description = kwargs.get("description", None)
    if raw_description:
      # Create a content AST document from the description text
      desc_doc = ca.Document()
      desc_doc.add_element(ca.Paragraph([raw_description]))
      self.description = desc_doc.render("html")
    else:
      self.description = None

    self.question_sort_order = None
    self.practice = practice
    self.preserve_order_point_values = set()  # Point values that should preserve question order
    self.preserve_yaml_order = kwargs.get("preserve_yaml_order", False)

    # Plan: right now we just take in questions and then assume they have a score and a "generate" button
  
  def __iter__(self):
    return iter(self.get_ordered_questions())
  
  @classmethod
  def from_yaml(cls, path_to_yaml) -> list[Quiz]:
    with open(path_to_yaml) as fid:
      list_of_exam_dicts = list(yaml.safe_load_all(fid))
    return cls.from_exam_dicts(list_of_exam_dicts, source_path=path_to_yaml)

  @classmethod
  def from_exam_dicts(cls, list_of_exam_dicts: list[dict], *, source_path: str | None = None) -> list[Quiz]:
    quizes_loaded : list[Quiz] = []

    def _combined_tags(*sources) -> list[str] | None:
      merged: set[str] = set()
      for source in sources:
        merged.update(Question.normalize_tags(source))
      if not merged:
        return None
      return sorted(merged)

    def _require_type(value, expected, label: str):
      if not isinstance(value, expected):
        raise ValueError(f"Invalid type for {label}: expected {expected}, got {type(value)}")

    def _validate_exam_dict(exam_dict: dict):
      def _validate_tags(value, label: str):
        if isinstance(value, str):
          return
        if isinstance(value, (list, tuple, set)):
          for item in value:
            _require_type(item, str, f"{label} tag")
          return
        raise ValueError(f"Invalid type for {label}: expected string or list of strings, got {type(value)}")

      if not isinstance(exam_dict, dict):
        raise ValueError("Each YAML document must be a mapping.")

      if "questions" not in exam_dict:
        raise KeyError("Missing required top-level key: questions")

      if "custom_modules" in exam_dict:
        _require_type(exam_dict["custom_modules"], list, "custom_modules")
        for module in exam_dict["custom_modules"]:
          _require_type(module, str, "custom_modules entry")

      if "sort order" in exam_dict:
        _require_type(exam_dict["sort order"], list, "sort order")
        for item in exam_dict["sort order"]:
          _require_type(item, str, "sort order entry")

      if "question_order" in exam_dict:
        _require_type(exam_dict["question_order"], str, "question_order")

      questions = exam_dict["questions"]
      if not isinstance(questions, (dict, list)):
        raise ValueError("questions must be a mapping (point values) or a list (ordered questions).")

      if isinstance(questions, list):
        for entry in questions:
          _require_type(entry, dict, "questions entry")
          if "name" not in entry:
            raise ValueError("Each question entry must include a 'name'.")
          if "points" not in entry:
            raise ValueError(f"Question '{entry.get('name', '<unknown>')}' must include 'points'.")
          if "class" in entry:
            _require_type(entry["class"], str, "class")
          if "_config" in entry:
            _require_type(entry["_config"], dict, "_config")
            if "tags" in entry["_config"]:
              _validate_tags(entry["_config"]["tags"], "_config.tags")
          if "kwargs" in entry:
            _require_type(entry["kwargs"], dict, "kwargs")
          if "tags" in entry:
            _validate_tags(entry["tags"], "tags")
          if "question_id" in entry:
            _require_type(entry["question_id"], str, "question_id")
          if "seed_group" in entry:
            _require_type(entry["seed_group"], str, "seed_group")
        return

      # Mapping format
      for points_value, question_definitions in questions.items():
        if not isinstance(points_value, (int, float)):
          raise ValueError(f"Point values must be numeric; got {points_value!r}.")
        _require_type(question_definitions, dict, f"questions[{points_value}]")

        point_config = question_definitions.get("_config", {})
        if point_config is not None:
          _require_type(point_config, dict, f"questions[{points_value}]._config")

        for q_name, q_data in question_definitions.items():
          if q_name == "_config":
            continue
          _require_type(q_data, dict, f"questions[{points_value}]['{q_name}']")
          if "class" in q_data:
            _require_type(q_data["class"], str, f"questions[{points_value}]['{q_name}'].class")
          if "_config" in q_data:
            _require_type(q_data["_config"], dict, f"questions[{points_value}]['{q_name}']._config")
            if "tags" in q_data["_config"]:
              _validate_tags(
                q_data["_config"]["tags"],
                f"questions[{points_value}]['{q_name}']._config.tags"
              )
          if "kwargs" in q_data:
            _require_type(q_data["kwargs"], dict, f"questions[{points_value}]['{q_name}'].kwargs")
          if "tags" in q_data:
            _validate_tags(q_data["tags"], f"questions[{points_value}]['{q_name}'].tags")
          if "question_id" in q_data:
            _require_type(q_data["question_id"], str, f"questions[{points_value}]['{q_name}'].question_id")
          if "seed_group" in q_data:
            _require_type(q_data["seed_group"], str, f"questions[{points_value}]['{q_name}'].seed_group")

          group_config = q_data.get("_config", {}) or {}
          if group_config.get("group", False):
            for group_name, group_data in q_data.items():
              if group_name == "_config":
                continue
              _require_type(group_data, dict, f"questions[{points_value}]['{q_name}']['{group_name}']")
              if "tags" in group_data:
                _validate_tags(
                  group_data["tags"],
                  f"questions[{points_value}]['{q_name}']['{group_name}'].tags"
                )
              if "question_id" in group_data:
                _require_type(
                  group_data["question_id"],
                  str,
                  f"questions[{points_value}]['{q_name}']['{group_name}'].question_id"
                )
              if "seed_group" in group_data:
                _require_type(
                  group_data["seed_group"],
                  str,
                  f"questions[{points_value}]['{q_name}']['{group_name}'].seed_group"
                )

    for exam_dict in list_of_exam_dicts:
      _validate_exam_dict(exam_dict)
      # Load custom question modules if specified (Option 3: Quick-and-dirty approach)
      # Users can add custom question types by importing Python modules in their YAML:
      # custom_modules:
      #   - my_custom_questions.scheduling
      #   - university_standard_questions
      custom_modules = exam_dict.get("custom_modules", [])
      if custom_modules:
        import importlib
        for module_name in custom_modules:
          try:
            importlib.import_module(module_name)
            log.info(f"Loaded custom question module: {module_name}")
          except ImportError as e:
            log.error(f"Failed to import custom module '{module_name}': {e}")
            raise


      # Ensure premade questions are loaded before validation/creation.
      QuestionRegistry.load_premade_questions()

      # Get general quiz information from the dictionary
      name = exam_dict.get("name", f"Unnamed Exam ({datetime.now().strftime('%a %b %d %I:%M %p')})")
      yaml_id = exam_dict.get("yaml_id")
      if isinstance(name, str):
        def replace_time(match: re.Match) -> str:
          fmt = match.group(1) or "%b %d %I:%M%p"
          return datetime.now().strftime(fmt)
        name = re.sub(r"\$TIME(?:\{([^}]+)\})?", replace_time, name)
      practice = exam_dict.get("practice", False)
      description = exam_dict.get("description", None)
      sort_order = list(map(lambda t: Question.Topic.from_string(t), exam_dict.get("sort order", [])))
      sort_order = sort_order + list(filter(lambda t: t not in sort_order, Question.Topic))
      
      # Load questions from the quiz dictionary
      questions_for_exam = []
      # Track point values where order should be preserved (for layout optimization)
      preserve_order_point_values = set()
      preserve_yaml_order = False

      question_order_setting = exam_dict.get("question_order", None)
      if isinstance(question_order_setting, str):
        if question_order_setting.lower() in {"yaml", "given", "preserve"}:
          preserve_yaml_order = True
        elif question_order_setting.lower() in {"points", "value", "score"}:
          preserve_yaml_order = False
        else:
          log.warning(f"Unknown question_order '{question_order_setting}'. Using defaults.")

      questions_block = exam_dict["questions"]

      def _format_available_questions(limit: int = 20) -> str:
        available = sorted(QuestionRegistry._registry.keys())
        if len(available) <= limit:
          return ", ".join(available)
        shown = ", ".join(available[:limit])
        return f"{shown}, ... (+{len(available) - limit} more)"

      def _is_known_question_type(question_class: str) -> bool:
        key = question_class.lower()
        registry = QuestionRegistry._registry
        reverse = QuestionRegistry._class_name_to_registered_name
        if key in registry or key in reverse:
          return True
        # Backward-compat prefix handling
        for prefix in ["cst334.", "cst463."]:
          if key.startswith(prefix):
            stripped = key[len(prefix):]
            if stripped in registry or stripped in reverse:
              return True
            if "." in stripped:
              final = stripped.split(".")[-1]
              if final in registry or final in reverse:
                return True
        if "." in key:
          final = key.split(".")[-1]
          if final in registry or final in reverse:
            return True
        return False
      if isinstance(questions_block, list):
        # List format preserves YAML order by default.
        if question_order_setting is None:
          preserve_yaml_order = True
        for entry in questions_block:
          if not isinstance(entry, dict):
            raise ValueError("Each entry in questions list must be a mapping.")
          q_name = entry.get("name")
          if q_name is None:
            raise ValueError("Each question entry must include a 'name'.")
          question_value = entry.get("points")
          if question_value is None:
            raise ValueError(f"Question '{q_name}' must include 'points'.")
          q_data = dict(entry)
          q_data.pop("name", None)
          q_data.pop("points", None)
          question_config = {
            "repeat": 1,
            "topic": "MISC",
            "tags": []
          }
          question_config.update(q_data.get("_config", {}))
          q_data.pop("_config", None)

          def make_question_from_entry(q_name, q_data, **kwargs):
            kwargs = {
              "name": q_name,
              "points_value": question_value,
              **q_data.get("kwargs", {}),
              **q_data,
              **kwargs,
            }
            if "topic" in q_data:
              kwargs["topic"] = Question.Topic.from_string(q_data.get("topic", "Misc"))
            elif "topic" in question_config:
              kwargs["topic"] = Question.Topic.from_string(question_config.get("topic", "Misc"))
            merged_tags = _combined_tags(question_config.get("tags"), q_data.get("tags"), kwargs.get("tags"))
            if merged_tags:
              kwargs["tags"] = merged_tags
            question_class = q_data.get("class", "FromText")
            try:
              if not _is_known_question_type(question_class):
                available = _format_available_questions()
                raise ValueError(
                  f"Unknown question type '{question_class}' for '{q_name}'. "
                  f"Available question types: {available}"
                )
              return QuestionRegistry.create(question_class, **kwargs)
            except Exception as e:
              raise ValueError(
                f"Failed to instantiate question '{q_name}' with class '{question_class}': {e}"
              ) from e

          questions_for_exam.extend([
            make_question_from_entry(
              q_name,
              q_data,
              rng_seed_offset=repeat_number
            )
            for repeat_number in range(question_config["repeat"])
          ])
      else:
        for question_value, question_definitions in questions_block.items():
          # todo: I can also add in "extra credit" and "mix-ins" as other keys to indicate extra credit or questions that can go anywhere
          log.info(f"Parsing {question_value} point questions")

          # Check for point-value-level config
          point_config = question_definitions.pop("_config", {})
          if point_config.get("preserve_order", False):
            preserve_order_point_values.add(question_value)
            log.info(f"  Point value {question_value} will preserve question order")

          def make_question(q_name, q_data, **kwargs):
            # Build up the kwargs that we're going to pass in
            # todo: this is currently a mess due to legacy things, so before I tell others to use this make it cleaner
            kwargs= {
              "name": q_name,
              "points_value": question_value,
              **q_data.get("kwargs", {}),
              **q_data,
              **kwargs,
            }
            
            # If we are passed in a topic then use it, otherwise don't set it which will have it set to a default
            if "topic" in q_data:
              kwargs["topic"] = Question.Topic.from_string(q_data.get("topic", "Misc"))
            
            # Add in a default, where if it isn't specified we're going to simply assume it is a text
            question_class = q_data.get("class", "FromText")
            
            try:
              if not _is_known_question_type(question_class):
                available = _format_available_questions()
                raise ValueError(
                  f"Unknown question type '{question_class}' for '{q_name}'. "
                  f"Available question types: {available}"
                )
              new_question = QuestionRegistry.create(
                question_class,
                **kwargs
              )
              return new_question
            except Exception as e:
              raise ValueError(
                f"Failed to instantiate question '{q_name}' with class '{question_class}': {e}"
              ) from e
          
          for q_name, q_data in question_definitions.items():
            # Set defaults for config
            question_config = {
              "group" : False,
              "num_to_pick" : 1,
              "random_per_student" : False,
              "repeat": 1,
              "topic": "MISC",
              "tags": []
            }
            
            # Update config if it exists
            question_config.update(
              q_data.get("_config", {})
            )
            q_data.pop("_config", None)
            if "pick" in q_data:
              raise ValueError(
                f"Legacy 'pick' key found in question '{q_name}'. "
                "Use _config.group with num_to_pick instead."
              )
            if "repeat" in q_data:
              raise ValueError(
                f"Legacy 'repeat' key found in question '{q_name}'. "
                "Use _config.repeat instead."
              )
            
            # Check if it is a question group
            merged_tags = _combined_tags(question_config.get("tags"), q_data.get("tags"))
            if question_config["group"]:
              
              # todo: Find a way to allow for "num_to_pick" to ensure lack of duplicates when using duplicates.
              #    It's probably going to be somewhere in the instantiate and get_attr fields, with "_current_questions"
              #    But will require changing how we add concrete questions (but that'll just be everything returns a list
              group_questions = []
              for name, data in q_data.items():
                merged_group_data = data | {"topic": question_config["topic"]}
                child_tags = _combined_tags(merged_tags, data.get("tags"))
                if child_tags:
                  merged_group_data["tags"] = child_tags
                group_questions.append(make_question(name, merged_group_data))

              questions_for_exam.append(
                QuestionGroup(
                  questions_in_group=group_questions,
                  pick_once=(not question_config["random_per_student"]),
                  name=q_name,
                  num_to_pick=question_config.get("num_to_pick", 1),
                  random_per_student=question_config.get("random_per_student", False)
                )
              )
            
            else: # Then this is just a single question
              questions_for_exam.extend([
                make_question(
                  q_name,
                  q_data,
                  rng_seed_offset=repeat_number,
                  **({"tags": merged_tags} if merged_tags else {})
                )
                for repeat_number in range(question_config["repeat"])
              ])
      log.debug(f"len(questions_for_exam): {len(questions_for_exam)}")
      quiz_from_yaml = cls(
        name,
        questions_for_exam,
        practice,
        description=description,
        preserve_yaml_order=preserve_yaml_order,
        yaml_id=yaml_id
      )
      quiz_from_yaml.set_sort_order(sort_order)
      quiz_from_yaml.preserve_order_point_values = preserve_order_point_values
      quizes_loaded.append(quiz_from_yaml)
    return quizes_loaded
  
  def _estimate_question_height(self, question, use_typst_measurement=False, **kwargs) -> float:
    """
    Estimate the rendered height of a question for layout optimization.
    Returns height in centimeters.

    Args:
        question: Question object to measure
        use_typst_measurement: If True, use Typst's layout engine for exact measurement
        **kwargs: Additional arguments passed to question rendering

    Returns:
        Height in centimeters
    """
    sample_count = int(kwargs.pop("layout_samples", 1))
    if sample_count < 1:
      sample_count = 1
    safety_factor = float(kwargs.pop("layout_safety_factor", 1.1))

    def deterministic_seeds(count: int) -> list[int]:
      seed_input = f"{question.__class__.__name__}:{question.name}:{question.points_value}"
      digest = hashlib.sha256(seed_input.encode("utf-8")).digest()
      base_seed = int.from_bytes(digest[:4], "big")
      rng = random.Random(base_seed)
      return [rng.randint(0, 2**31 - 1) for _ in range(count)]

    use_typst = False
    if use_typst_measurement:
      from QuizGenerator.typst_utils import check_typst_available
      use_typst = check_typst_available()
      if not use_typst:
        log.debug("Typst not available, using heuristic estimation")

    def estimate_for_seed(seed: int) -> float:
      if use_typst:
        try:
          from QuizGenerator.typst_utils import measure_typst_content
          instance = question.instantiate(rng_seed=seed, **kwargs)
          typst_body = instance.body.render("typst", **kwargs)
          measured_height = measure_typst_content(typst_body, page_width_cm=18.0)
          if measured_height is not None:
            total_height = 1.5 + measured_height + question.spacing
            log.debug(
              f"Typst measurement: {question.name} = {total_height:.2f}cm "
              f"(content: {measured_height:.2f}cm, spacing: {question.spacing}cm)"
            )
            return total_height
        except Exception as e:
          log.warning(f"Error during Typst measurement: {e}, falling back to heuristics")

      # Fallback: Use heuristic estimation
      base_height = 1.5  # cm
      spacing_height = question.spacing  # cm

      instance = question.instantiate(rng_seed=seed, **kwargs)
      question_ast = question._build_question_ast(instance)
      latex_content = question_ast.render("latex")

      content_height = 0.0
      table_count = latex_content.count('\\begin{tabular}')
      content_height += table_count * 3.0

      matrix_count = latex_content.count('\\begin{') - table_count
      content_height += matrix_count * 2.0

      verbatim_count = latex_content.count('\\begin{verbatim}')
      content_height += verbatim_count * 4.0

      char_count = len(latex_content)
      content_height += (char_count / 500.0) * 0.5

      return base_height + spacing_height + content_height

    heights = [estimate_for_seed(seed) for seed in deterministic_seeds(sample_count)]
    return max(heights) * safety_factor

  def _optimize_question_order(self, questions, **kwargs) -> list[Question]:
    """
    Optimize question ordering to minimize PDF length while respecting point-value tiers.
    Uses bin-packing heuristics to reorder questions within each point-value group.
    """
    # Group questions by point value
    from collections import defaultdict
    point_groups = defaultdict(list)

    for question in questions:
      point_groups[question.points_value].append(question)

    # Track which point values should preserve order (from config)
    preserve_order_for = kwargs.pop('preserve_order_for', set())

    # For each point group, estimate heights and apply bin-packing optimization
    optimized_questions = []
    is_first_bin_overall = True  # Track if we're packing the very first bin of the entire exam

    log.debug("Optimizing question order for PDF layout...")

    def get_spacing_priority(question):
      """
      Get placement priority based on spacing. Lower values = higher priority.
      Order: LONG (9), MEDIUM (6), SHORT (4), NONE (0), then PAGE (99), EXTRA_PAGE (199)
      """
      spacing = question.spacing
      if spacing >= 199:  # EXTRA_PAGE
        return 5
      elif spacing >= 99:  # PAGE
        return 4
      elif spacing >= 9:  # LONG
        return 0
      elif spacing >= 6:  # MEDIUM
        return 1
      elif spacing >= 4:  # SHORT
        return 2
      else:  # NONE or custom small values
        return 3

    for points in sorted(point_groups.keys(), reverse=True):
      group = point_groups[points]

      # Check if this point tier should preserve order
      if points in preserve_order_for:
        # Sort by topic only (preserve original order)
        group.sort(key=lambda q: self.question_sort_order.index(q.topic))
        optimized_questions.extend(group)
        log.debug(f"  {points}pt questions: {len(group)} questions (order preserved by config)")
        # After adding preserved-order questions, we're no longer on the first bin
        is_first_bin_overall = False
        continue

      # If only 1-2 questions, no optimization needed
      if len(group) <= 2:
        # Sort by spacing priority first, then topic
        group.sort(key=lambda q: (get_spacing_priority(q), self.question_sort_order.index(q.topic)))
        optimized_questions.extend(group)
        log.debug(f"  {points}pt questions: {len(group)} questions (no optimization needed)")
        is_first_bin_overall = False
        continue

      # Estimate height for each question, preserving original index for stable sorting
      question_heights = []
      for i, q in enumerate(group):
        height = getattr(q, "layout_reserved_height", None)
        if height is None:
          height = self._estimate_question_height(q, **kwargs)
        question_heights.append((i, q, height))

      # Sort by:
      # 1. Spacing priority (LONG, MEDIUM, SHORT, NONE, then PAGE, EXTRA_PAGE)
      # 2. Height descending (within same spacing category)
      # 3. Original index (for deterministic ordering)
      question_heights.sort(key=lambda x: (get_spacing_priority(x[1]), -x[2], x[0]))

      log.debug(f"  Question heights for {points}pt questions:")
      for idx, q, h in question_heights:
        log.debug(f"    {q.name}: {h:.1f}cm (spacing={q.spacing}cm)")

      # Calculate page capacity in centimeters
      # A typical A4 page with margins has ~25cm of usable height
      # After accounting for headers and separators, estimate ~22cm per page
      base_page_capacity = 22.0  # cm

      # First page has header (title + name line) which takes ~3cm
      first_page_capacity = base_page_capacity - 3.0  # cm

      # Better bin-packing strategy: interleave large and small questions
      # Strategy: Start each page with the largest unplaced question, then fill with smaller ones
      bins = []
      placed = [False] * len(question_heights)

      while not all(placed):
        # Determine capacity for this page
        # Use first_page_capacity only for the very first bin of the entire exam
        page_capacity = first_page_capacity if (len(bins) == 0 and is_first_bin_overall) else base_page_capacity

        # Find the largest unplaced question to start a new page
        new_page = []
        page_height = 0

        # Special handling for first bin of entire exam: avoid questions with PAGE spacing (99+cm)
        # to prevent them from pushing content to page 2
        if len(bins) == 0 and is_first_bin_overall:
          # Try to find a question without PAGE/EXTRA_PAGE spacing for the first page
          # PAGE=99cm, EXTRA_PAGE=199cm - these need full pages
          found_non_page_question = False
          for i, (idx, question, height) in enumerate(question_heights):
            if not placed[i] and question.spacing < 99:
              new_page.append(question)
              page_height = height
              placed[i] = True
              found_non_page_question = True
              log.debug(f"    First bin (page 1): Selected {question.name} with spacing={question.spacing}cm")
              break

          # If all questions have PAGE spacing, fall back to normal behavior (use largest question)
          if not found_non_page_question:
            for i, (idx, question, height) in enumerate(question_heights):
              if not placed[i]:
                new_page.append(question)
                page_height = height
                placed[i] = True
                log.debug(f"    First bin (page 1): All questions have PAGE spacing, using {question.name} (spacing={question.spacing}cm)")
                break
        else:
          # Normal behavior for non-first pages
          for i, (idx, question, height) in enumerate(question_heights):
            if not placed[i]:
              new_page.append(question)
              page_height = height
              placed[i] = True
              break

        # Now try to fill the remaining space with smaller questions
        for i, (idx, question, height) in enumerate(question_heights):
          if not placed[i] and page_height + height <= page_capacity:
            new_page.append(question)
            page_height += height
            placed[i] = True

        bins.append((new_page, page_height))

        # After creating the first bin, we're no longer on the first page
        if len(bins) == 1 and is_first_bin_overall:
          is_first_bin_overall = False
          log.debug(f"    First bin created, subsequent bins will use normal page capacity")

      log.debug(f"  {points}pt questions: {len(group)} questions packed into {len(bins)} pages")
      for i, (page_questions, height) in enumerate(bins):
        log.debug(f"    Page {i+1}: {height:.1f}cm with {len(page_questions)} questions: {[q.name for q in page_questions]}")

      # Flatten bins back to ordered list
      for bin_contents, _ in bins:
        optimized_questions.extend(bin_contents)

    return optimized_questions

  def get_ordered_questions(self, **kwargs) -> list[Question]:
    if self.question_sort_order is None:
      self.question_sort_order = list(Question.Topic)

    preserve_yaml_order = kwargs.pop("preserve_yaml_order", self.preserve_yaml_order)
    optimize_layout = kwargs.pop("optimize_layout", False)

    if preserve_yaml_order:
      return list(self.questions)

    if optimize_layout:
      return self._optimize_question_order(
        self.questions,
        preserve_order_for=self.preserve_order_point_values,
        **kwargs
      )

    # Default: order by point value, then topic
    def sort_func(q):
      if self.question_sort_order is not None:
        try:
          return (-q.points_value, self.question_sort_order.index(q.topic))
        except ValueError:
          return (-q.points_value, float('inf'))
      return -q.points_value

    return sorted(self.questions, key=sort_func)

  def get_quiz(self, **kwargs) -> ca.Document:
    quiz = ca.Document(title=self.name)

    # Extract master RNG seed (if provided) and remove from kwargs
    master_seed = kwargs.pop('rng_seed', None)

    consistent_pages = kwargs.pop("consistent_pages", False)

    # Precompute reserved heights for consistent pagination (Typst only).
    if consistent_pages:
      for question in self.questions:
        if getattr(question, "layout_reserved_height", None) is None:
          question.layout_reserved_height = self._estimate_question_height(question, **dict(kwargs))

    # Check if optimization is requested (default: False)
    optimize_layout = kwargs.pop('optimize_layout', False)
    ordered_questions = self.get_ordered_questions(
      optimize_layout=optimize_layout,
      **kwargs
    )

    # Generate questions with sequential numbering for QR codes
    # Use the master seed when provided. seed_group allows selected questions
    # to share the same seed (and therefore the same workload/context).
    seed_rng = random.Random(master_seed) if master_seed is not None else random.Random()
    seed_groups: dict[str, int] = {}

    def get_seed_group(question) -> str | None:
      seed_group = getattr(question, "seed_group", None)
      if seed_group is None:
        return None
      if isinstance(seed_group, str):
        normalized = seed_group.strip()
        return normalized if normalized else None
      return str(seed_group)

    question_number = 1
    for question in ordered_questions:
      seed_group = get_seed_group(question)
      should_seed = master_seed is not None or seed_group is not None

      if should_seed:
        if seed_group is not None:
          if seed_group not in seed_groups:
            seed_groups[seed_group] = seed_rng.randint(0, 2**31 - 1)
          question_seed = seed_groups[seed_group]
        else:
          question_seed = seed_rng.randint(0, 2**31 - 1)
        instance = question.instantiate(rng_seed=question_seed, **kwargs)
        instances = instance if isinstance(instance, list) else [instance]
      else:
        instance = question.instantiate(**kwargs)
        instances = instance if isinstance(instance, list) else [instance]

      # Add question number to the AST for QR code generation
      for item in instances:
        question_ast = question._build_question_ast(item)
        question_ast.question_number = question_number
        if getattr(self, "yaml_id", None):
          question_ast.yaml_id = self.yaml_id
        quiz.add_element(question_ast)
        question_number += 1

    return quiz
  
  def describe(self):
    
    # Print out title
    print(f"Title: {self.name}")
    total_points = sum(map(lambda q: q.points_value, self.questions))
    total_questions = len(self.questions)
    
    # Print out overall information
    print(f"{total_points} points total, {total_questions} questions")
    
    # Print out the per-value information
    points_counter = collections.Counter([q.points_value for q in self.questions])
    for points in sorted(points_counter.keys(), reverse=True):
      print(f"{points_counter.get(points)} x {points}points")
    
    # Either get the sort order or default to the order in the enum class
    sort_order = self.question_sort_order
    if sort_order is None:
      sort_order = Question.Topic
      
    # Build per-topic information
    
    topic_information = {}
    topic_strings = {}
    for topic in sort_order:
      topic_strings = {"name": topic.name}
      
      question_count = len(list(map(lambda q: q.points_value, filter(lambda q: q.topic == topic, self.questions))))
      topic_points = sum(map(lambda q: q.points_value, filter(lambda q: q.topic == topic, self.questions)))
      
      # If we have questions add in some states, otherwise mark them as empty
      if question_count != 0:
        topic_strings["count_str"] = f"{question_count} questions ({ 100 * question_count / total_questions:0.1f}%)"
        topic_strings["points_str"] = f"{topic_points:2} points ({ 100 * topic_points / total_points:0.1f}%)"
      else:
        topic_strings["count_str"] = "--"
        topic_strings["points_str"] = "--"
      
      topic_information[topic] = topic_strings
    
    
    # Get padding string lengths
    paddings = collections.defaultdict(lambda: 0)
    for field in topic_strings.keys():
      paddings[field] = max(len(information[field]) for information in topic_information.values())
    
    # Print out topics information using the padding
    for topic in sort_order:
      topic_strings = topic_information[topic]
      print(f"{topic_strings['name']:{paddings['name']}} : {topic_strings['count_str']:{paddings['count_str']}} : {topic_strings['points_str']:{paddings['points_str']}}")

    tag_counter = collections.Counter()
    tag_points: dict[str, float] = collections.defaultdict(float)
    for question in self.questions:
      for tag in sorted(getattr(question, "tags", set())):
        tag_counter[tag] += 1
        tag_points[tag] += question.points_value

    if tag_counter:
      print("Tags:")
      tag_name_padding = max(len(tag) for tag in tag_counter.keys())
      count_padding = max(len(str(count)) for count in tag_counter.values())
      for tag in sorted(tag_counter.keys()):
        count = tag_counter[tag]
        points = tag_points[tag]
        print(
          f"{tag:{tag_name_padding}} : {count:{count_padding}} questions "
          f"({100 * count / total_questions:0.1f}%) : {points:0.1f} points"
        )
    
  def set_sort_order(self, sort_order):
    self.question_sort_order = sort_order

def main():
  pass
  

if __name__ == "__main__":
  main()
  
