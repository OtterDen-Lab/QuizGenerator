# Creating Custom Question Types

QuizGenerator supports two approaches for adding custom question types to your quizzes:

1. **Entry Points (Recommended)** - Create a proper Python package with questions that can be pip-installed
2. **Direct Module Import (Quick & Dirty)** - Import custom question modules directly in your YAML config

## Approach 1: Entry Points (Recommended for Distribution)

This is the standard Python plugin mechanism, ideal for:
- Sharing question banks across multiple courses or institutions
- Publishing reusable question types to PyPI
- Maintaining questions in a separate package from QuizGenerator

### Step 1: Create Your Question Package

Create a Python package with your custom questions:

```
my_custom_questions/
├── pyproject.toml
├── my_custom_questions/
│   ├── __init__.py
│   └── scheduling.py
```

**scheduling.py:**
```python
import random

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry

@QuestionRegistry.register()  # Registers as "schedulingquestion"
class SchedulingQuestion(Question):
    VERSION = "1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, topic=Question.Topic.SYSTEM_PROCESSES, **kwargs)
        self.possible_variations = float('inf')

    def _build_context(self, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        rng = context.rng

        # Generate random process workloads
        num_processes = rng.randint(3, 5)
        arrival_times = [rng.randint(0, 10) for _ in range(num_processes)]
        burst_times = [rng.randint(1, 8) for _ in range(num_processes)]

        # Calculate FIFO schedule
        schedule = self._calculate_fifo(arrival_times, burst_times)

        context.update({
            "num_processes": num_processes,
            "arrival_times": arrival_times,
            "burst_times": burst_times,
            "schedule": schedule,
        })
        return context

    def _calculate_fifo(self, arrival_times, burst_times):
        # Implementation here...
        pass

    def _build_body(self, context) -> ca.Section:
        body = ca.Section()
        body.add_element(ca.Paragraph([
            "Given the following processes with their arrival times and burst times, "
            "calculate the completion time using FIFO scheduling:"
        ]))

        # Create table of processes
        table_data = [["Process", "Arrival", "Burst"]]
        for i in range(context["num_processes"]):
            table_data.append([
                f"P{i}",
                str(context["arrival_times"][i]),
                str(context["burst_times"][i]),
            ])

        body.add_element(ca.Table(table_data))
        body.add_element(ca.Paragraph(["Completion time: "]))
        body.add_element(ca.AnswerTypes.Int(context["schedule"][-1], label="Completion time"))

        return body

    def _build_explanation(self, context) -> ca.Section:
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([
            "FIFO scheduling executes processes in order of arrival..."
        ]))
        # Add step-by-step solution
        return explanation
```

### Step 2: Register Your Questions via Entry Points

**pyproject.toml:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-custom-questions"
version = "0.1.0"
description = "Custom quiz questions for Operating Systems"
dependencies = [
    "QuizGenerator>=0.1.0",
]

[project.entry-points."quizgenerator.questions"]
# Entry point name can be anything; the class gets registered by its decorator
scheduling = "my_custom_questions.scheduling:SchedulingQuestion"
advanced_memory = "my_custom_questions.memory:AdvancedMemoryQuestion"
```

### Step 3: Install Your Package

```bash
# Install in development mode
pip install -e .

# Or install from PyPI once published
pip install my-custom-questions
```

### Step 4: Use in Your Quiz YAML

Once installed, your questions are automatically available:

```yaml
name: "Operating Systems Midterm"
questions:
  10:
    "FIFO Scheduling":
      class: SchedulingQuestion  # Automatically discovered!

    "Memory Management":
      class: AdvancedMemoryQuestion
```

**No import needed!** QuizGenerator automatically discovers and loads all registered entry points.

---

## Approach 2: Direct Module Import (Quick & Dirty)

This approach is ideal for:
- Quick prototyping and testing
- One-off custom questions
- Local/personal question banks that won't be distributed

### Step 1: Create Your Question Module

Create a Python file anywhere accessible to your quiz config:

**my_questions.py:**
```python
import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry

@QuestionRegistry.register()
class QuickMemoryQuestion(Question):
    VERSION = "1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, topic=Question.Topic.SYSTEM_MEMORY, **kwargs)

    def _build_context(self, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        rng = context.rng
        page_size = rng.choice([4, 8, 16])  # KB
        virtual_addr = rng.randint(0, 65535)

        page_number = virtual_addr // (page_size * 1024)
        offset = virtual_addr % (page_size * 1024)

        context.update({
            "page_size": page_size,
            "virtual_addr": virtual_addr,
            "page_number": page_number,
            "offset": offset,
        })
        return context

    def _build_body(self, context) -> ca.Section:
        body = ca.Section()
        body.add_element(ca.Paragraph([
            f"Given a page size of {context['page_size']}KB and virtual address {context['virtual_addr']}, "
            "calculate the page number and offset."
        ]))
        body.add_element(ca.Paragraph(["Page number: "]))
        body.add_element(ca.AnswerTypes.Int(context["page_number"], label="Page number"))
        body.add_element(ca.Paragraph(["Offset: "]))
        body.add_element(ca.AnswerTypes.Int(context["offset"], label="Offset"))
        return body

    def _build_explanation(self, context) -> ca.Section:
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([
            f"Page number = {context['virtual_addr']} / ({context['page_size']} * 1024) = {context['page_number']}"
        ]))
        explanation.add_element(ca.Paragraph([
            f"Offset = {context['virtual_addr']} % ({context['page_size']} * 1024) = {context['offset']}"
        ]))
        return explanation
```

### Step 2: Import in Your Quiz YAML

Add a `custom_modules` section to your YAML config:

```yaml
# Import custom question modules (Option 2: Quick & dirty)
custom_modules:
  - my_questions  # Import my_questions.py (must be in PYTHONPATH)
  - university_standard_questions.os  # Can import from packages too

name: "Operating Systems Quiz"
questions:
  5:
    "Virtual Memory":
      class: QuickMemoryQuestion  # Now available!
```

### Step 3: Run Quiz Generation

Make sure your module is importable:

```bash
# Option A: Run from the same directory
quizgen --yaml my_quiz.yaml --num_pdfs 3

# Option B: Add to PYTHONPATH
export PYTHONPATH="/path/to/my/questions:$PYTHONPATH"
quizgen --yaml my_quiz.yaml --num_pdfs 3
```

---

## Comparison: Which Approach Should I Use?

| Feature | Entry Points (Option 1) | Direct Import (Option 2) |
|---------|------------------------|--------------------------|
| **Setup complexity** | Requires proper package structure | Just create a .py file |
| **Installation** | `pip install` | Add to PYTHONPATH |
| **Distribution** | Can publish to PyPI | Share .py files manually |
| **Auto-discovery** | ✅ Automatic | ❌ Manual import needed |
| **Best for** | Reusable question banks | Quick prototyping |
| **Recommended for** | Production, sharing | Testing, personal use |

### Recommendation:
- **Start with Option 2** (Direct Import) while developing and testing your questions
- **Migrate to Option 1** (Entry Points) once your questions are stable and you want to share them

---

## Question Development Best Practices

Regardless of which approach you use, follow these best practices:

### 1. Always Use ContentAST
```python
# ✅ Good - uses ContentAST
def _build_body(self, context):
    body = ca.Section()
    body.add_element(ca.Paragraph(["Calculate the result:"]))
    body.add_element(ca.AnswerTypes.Int(context["result"], label="Result"))
    return body

# ❌ Bad - manual LaTeX/HTML
def _build_body(self, context):
    return "Calculate the result: \\blank{5cm}"
```

### 2. Version Your Questions
```python
class MyQuestion(Question):
    VERSION = "1.0"  # Increment when RNG logic changes
```

### 3. Implement `_build_explanation()`
Always provide solutions for your questions:
```python
def _build_explanation(self, context) -> ca.Section:
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
        "Step 1: Calculate X...",
        "Step 2: Apply formula Y...",
        f"Final answer: {context['result']}"
    ]))
    return explanation
```

### 4. Use the RNG Correctly
```python
def _build_context(self, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    rng = context.rng

    # Use rng (not random module directly) for reproducibility
    context["value"] = rng.randint(1, 100)
    context["operation"] = rng.choice(['add', 'subtract'])
    return context
```

### 5. Set Proper Topics
```python
def __init__(self, *args, **kwargs):
    super().__init__(
        *args,
        topic=Question.Topic.SYSTEM_MEMORY,  # Set appropriate topic
        **kwargs
    )
```

---

## Example: Complete Custom Question Package

Here's a complete example of a distributable question package:

**Repository structure:**
```
os-quiz-questions/
├── pyproject.toml
├── README.md
├── os_questions/
│   ├── __init__.py
│   ├── scheduling.py
│   ├── memory.py
│   └── concurrency.py
```

**pyproject.toml:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "os-quiz-questions"
version = "1.0.0"
description = "Operating Systems quiz questions for CST334"
authors = [{name = "Your Name", email = "you@university.edu"}]
requires-python = ">=3.12"
dependencies = ["QuizGenerator>=0.1.0"]

[project.entry-points."quizgenerator.questions"]
# Scheduling questions
fifo_scheduling = "os_questions.scheduling:FIFOSchedulingQuestion"
sjf_scheduling = "os_questions.scheduling:SJFSchedulingQuestion"
round_robin = "os_questions.scheduling:RoundRobinQuestion"

# Memory questions
paging = "os_questions.memory:PagingQuestion"
segmentation = "os_questions.memory:SegmentationQuestion"
tlb = "os_questions.memory:TLBQuestion"

# Concurrency questions
mutex = "os_questions.concurrency:MutexQuestion"
semaphore = "os_questions.concurrency:SemaphoreQuestion"
deadlock = "os_questions.concurrency:DeadlockQuestion"
```

Users can then install and use:
```bash
pip install os-quiz-questions
```

```yaml
# quiz.yaml - no imports needed!
name: "Operating Systems Final"
questions:
  10:
    "Scheduling":
      class: FIFOSchedulingQuestion
    "Memory":
      class: PagingQuestion
  5:
    "Concurrency":
      class: MutexQuestion
```

---

## Troubleshooting

### "FromGenerator is disabled by default"

**Problem:** You see an error stating FromGenerator is disabled.

**Solution:** Enable it explicitly:
```bash
quizgen --allow_generator --yaml my_quiz.yaml --num_pdfs 1
```
Or set `QUIZGEN_ALLOW_GENERATOR=1` in your environment.

### "Unknown question type" Error

**Problem:** QuizGenerator can't find your custom question.

**Solutions:**
1. **For Entry Points:** Make sure package is installed (`pip list | grep my-package`)
2. **For Direct Import:** Check `custom_modules` is at top of YAML before `questions:`
3. Verify `@QuestionRegistry.register()` decorator is present
4. Check that class name matches what's in YAML (case-insensitive)

### Import Errors

**Problem:** Module import fails with `ModuleNotFoundError`.

**Solutions:**
1. **For Entry Points:** Reinstall package with `pip install -e .`
2. **For Direct Import:** Add directory to PYTHONPATH: `export PYTHONPATH="/path/to/questions:$PYTHONPATH"`
3. Check that `__init__.py` files exist in all package directories

### Questions Not Regenerating with QR Codes

**Problem:** Scanning QR codes doesn't reproduce the exact question.

**Solutions:**
1. Increment `VERSION` whenever you change RNG logic
2. Always use `self.rng` instead of `random` module
3. Call `super()._build_context(rng_seed=rng_seed, **kwargs)` FIRST in your `_build_context()` method and return the updated context (`QuestionContext`)

---

## Additional Resources

- **Example Questions:** See `QuizGenerator/premade_questions/` for reference implementations
- **ContentAST Guide:** Documentation on available content elements
- **Question Base Class:** See `QuizGenerator/question.py` for full API
- **Testing Your Questions:** Use `pytest` to test question generation and validation
