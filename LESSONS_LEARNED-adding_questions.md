# Lessons Learned: Adding New Question Types

This document captures key insights from implementing new question types for the QuizGenerator system.

## Executive Summary: Critical Success Factors for LLM Agents

Based on multiple question development cycles, these are the essential practices for efficient implementation:

### 🚨 **Check First, Implement Second**
1. **Code style**: Verify project conventions (see CLAUDE.md) - this project uses 2-space indentation
2. **Existing patterns**: Search `misc.py` for `Answer` factory methods (`Answer.float_value()`, not manual rounding)
3. **Mathematical frameworks**: Use SymPy's `latex()` function for automatic formatting, never manual LaTeX construction
4. **Platform differences**: Canvas MathJax ≠ PDF LaTeX - test both early and often

### 🎯 **Core Implementation Patterns**
- **Question structure**: Extend base class + `@QuestionRegistry.register()` + three methods (`refresh()`, `get_body()`, `get_explanation()`)
- **Testing workflow**: `scratch.yaml` → `python generate_quiz.py TEST` → Canvas + PDF verification
- **Mathematical content**: SymPy for expressions, `ContentAST.Equation` for display, avoid fragmented math elements
- **Canvas compatibility**: No LaTeX in table headers (`<th>`), use bold `<td>` instead

### 🔧 **Common Anti-Patterns to Avoid**
- Manual mathematical formatting (`\nabla f = \begin{pmatrix}...` → `sp.latex(gradient_function)`)
- Magic numbers (`round(value, 4)` → `Answer.float_value()` with automatic `DEFAULT_ROUNDING_DIGITS`)
- Hardcoded constraints (1-2 variables → parameterized for 1-5+ variables)
- Ignoring teaching conventions (ask about notation preferences: horizontal vs vertical vectors)

### 📁 **Essential Reference Files**
- `misc.py`: Answer types and ContentAST system
- `cst334/memory_questions.py`: Comprehensive question examples
- `constants.py`: Standard ranges and probabilities
- `scratch.yaml`: Testing configuration

**Bottom line**: Follow existing patterns, leverage SymPy, test both platforms, and verify conventions before implementing.

---

## Gradient Descent Questions Development

**Date:** September 22, 2025
**Project:** CST463 Gradient Descent Walkthrough Questions (Issue #23)
**Duration:** Multi-session iterative development

### Overview
Implementation of interactive gradient descent walkthrough questions with table-based student input, supporting 1-5 variables with SymPy integration for automatic mathematical notation.

### Major Lessons Learned

#### 1. **Follow Project Code Style Consistently**
**Problem**: Defaulted to 4-space indentation while project uses 2-space standard.
**Impact**: Required constant corrections and style mismatches.
**Solution**: Check project conventions early (see CLAUDE.md) and configure accordingly.
**Key Learning**: Always verify and follow existing code style before starting implementation.

#### 2. **Leverage Existing Mathematical Frameworks**
**Problem**: Initially attempted manual LaTeX formatting for mathematical expressions.
**Better Approach**: Use SymPy's automatic `latex()` function for consistent, correct mathematical notation.
**Example**:
```python
# Manual approach (problematic)
gradient_latex = f"\\nabla f = \\begin{{pmatrix}} {deriv_x} \\\\ {deriv_y} \\end{{pmatrix}}"

# SymPy approach (better)
gradient_latex = sp.latex(self.gradient_function)
```
**Key Learning**: When working with mathematical content, use SymPy for automatic LaTeX generation rather than manual string construction.

#### 3. **Use Correct Answer Types and Avoid Magic Numbers**
**Problem**: Used `Answer.float()` with manual rounding: `Answer.float("key", round(value, 4))`
**Correct Approach**: Use `Answer.float_value()` which automatically applies `DEFAULT_ROUNDING_DIGITS`
**Impact**: Eliminates magic numbers and ensures consistent rounding across the project.
**Key Learning**: Check `Answer` class methods in `misc.py` for appropriate factory methods that handle formatting automatically.

#### 4. **Consider Classroom Teaching Conventions**
**Problem**: Implemented gradient as horizontal vector, but class teaches vertical vectors.
**Solution**: Asked user about teaching style and switched to vertical vector display using `inline=False`.
**Key Learning**: Always confirm mathematical notation conventions match how concepts are taught in class to minimize student confusion.

#### 5. **Canvas Table Header Compatibility Issues**
**Problem**: LaTeX rendering in HTML table headers (`<th>` tags) caused Canvas display issues.
**Solution**: Moved LaTeX content to regular table cells (`<td>`) with bold formatting.
**Root Cause**: Canvas MathJax doesn't reliably process LaTeX in table header elements.
**Key Learning**: Test Canvas rendering early, especially for table-based questions with mathematical content.

#### 6. **Mathematical Function Notation**
**Problem**: Initially showed expressions without proper function notation.
**Solution**: Include complete mathematical statements like `f(x) = expression` rather than just the expression.
**Key Learning**: Proper mathematical notation helps students understand context and connects to classroom teaching.

#### 7. **Flexible Variable Scaling**
**Problem**: Initially hardcoded for 1-2 variables, user suggested extending to 1-5 variables.
**Solution**: Generalized implementation to handle arbitrary variable counts.
**Key Learning**: Design for flexibility from the start - educational software often needs to scale to different problem sizes.

### What Worked Well

#### 1. **SymPy Integration**
- Automatic LaTeX generation eliminated manual formatting errors
- Symbolic differentiation provided exact gradients
- Seamless handling of multi-variable cases

#### 2. **TableQuestionMixin Usage**
- Simplified table creation with answer fields
- Handled cross-platform rendering automatically
- Provided consistent table formatting

#### 3. **Iterative User Feedback**
- Multiple rounds of user corrections led to robust implementation
- Each feedback cycle improved both functionality and educational value

### Development Anti-Patterns to Avoid

#### 1. **Manual Mathematical Formatting**
- Don't manually construct LaTeX for mathematical expressions
- Use SymPy's `latex()` function for automatic, correct formatting

#### 2. **Hardcoded Magic Numbers**
- Don't use `round(value, 4)` - use `Answer.float_value()` with automatic rounding
- Don't hardcode variable counts - parameterize for flexibility

#### 3. **Ignoring Platform Differences**
- Don't assume LaTeX works the same in Canvas and PDF
- Test both platforms early and often

#### 4. **Skipping Style Verification**
- Don't assume code style - check project conventions first
- Configure development environment to match project standards

### Essential Files for Mathematical Questions

```
QuizGenerator/
├── misc.py                   # Answer types and ContentAST system
├── premade_questions/
│   └── cst463/gradient_descent/
│       ├── __init__.py       # Must import new classes
│       └── gradient_descent_questions.py
└── example_files/
    └── scratch.yaml          # Testing configuration
```

---

## Matrix Math Questions Development

**Date:** September 19, 2025
**Project:** CST463 Matrix Math Questions (Issue #20)
**Duration:** Single session implementation

## Overview

This document captures key insights and lessons learned from implementing matrix math questions for the QuizGenerator system. The goal is to streamline future question development processes by documenting what worked well, what caused issues, and where to find helpful examples.

## What Went Well

### 1. **Systematic Question Structure**
- Following the established pattern of extending `Question` class and using `@QuestionRegistry.register()` worked seamlessly
- Creating a base `MatrixMathQuestion` class for common functionality was highly effective
- The three-method structure (`refresh()`, `get_body()`, `get_explanation()`) provides clear separation of concerns

### 2. **Leveraging Existing Patterns**
- **Constants file**: Found standard probability patterns in `QuizGenerator/constants.py` and `memory_questions.py` (`PROBABILITY_OF_VALID = 0.875`)
- **Answer types**: `Answer.integer()` and `Answer.string()` factory methods worked perfectly for matrix elements and dimension answers
- **ContentAST system**: Once understood, provided powerful abstraction for multi-format output

### 3. **Iterative Testing Approach**
- Using `example_files/scratch.yaml` for manual testing was invaluable
- Testing both Canvas and PDF outputs early caught rendering issues quickly

## Major Challenges & Solutions

### 1. **Question Registry Loading Issues**
**Problem**: Nested subdirectories (`cst463/math_and_data/`) weren't being loaded by the question registry.

**Root Cause**: The `load_premade_questions()` function only did shallow directory traversal.

**Solution**: Enhanced the function to recursively traverse subdirectories and handle dotted namespace syntax.

**File**: `QuizGenerator/question.py:load_premade_questions()`

**Key Learning**: Always test question loading with `python generate_quiz.py TEST` when adding new question directories.

### 2. **Canvas MathJax Rendering Problems**
**Problem**: Matrices were being split across multiple equation blocks, causing broken rendering.

**Root Cause**: Individual `ContentAST.Matrix` elements created separate `<div class='math'>` blocks.

**Solution**: Created `Matrix.to_latex()` static method to generate single equation strings that could be embedded in `ContentAST.Equation`.

**Example**:
```python
# Bad - creates multiple equation blocks
body.add_element(ContentAST.Matrix(data=matrix_a))
body.add_element(ContentAST.Text(" + "))
body.add_element(ContentAST.Matrix(data=matrix_b))

# Good - single equation block
matrix_a_latex = ContentAST.Matrix.to_latex(matrix_a, "b")
matrix_b_latex = ContentAST.Matrix.to_latex(matrix_b, "b")
body.add_element(ContentAST.Equation(f"{matrix_a_latex} + {matrix_b_latex} = "))
```

**Key Learning**: Canvas/MathJax prefers single equation blocks over fragmented math elements.

### 3. **PDF Generation Failures**
**Problem**: LaTeX compilation failed with matrix-related errors.

**Root Cause**: Missing `amsmath`, `amsfonts`, and `amssymb` packages needed for `bmatrix` environment.

**Solution**: Added required packages to `ContentAST.Document.LATEX_HEADER` in `QuizGenerator/misc.py`.

**Key Learning**: When adding new LaTeX constructs, verify all required packages are included in the document header.

### 4. **Equation Formatting Issues**
**Problem**: Unwanted periods appearing after equations on Canvas, and equations were centered instead of left-aligned.

**Root Cause**:
- The `\frac{}{}` in equation render methods was adding periods
- Default LaTeX math environments center equations

**Solutions**:
- Removed `\frac{}{}` from `ContentAST.Equation.render_html()` and `render_markdown()`
- Used `\begin{flushleft}...\end{flushleft}` around equations for left alignment

**Key Learning**: Test both Canvas and PDF output when modifying equation rendering.

## Essential Reference Files & Patterns

### 1. **Finding Implementation Examples**
- **Question structure**: Look at `QuizGenerator/premade_questions/cst334/memory_questions.py` for comprehensive examples
- **Answer types**: Check `QuizGenerator/misc.py:Answer` class for factory methods and patterns
- **Constants/probabilities**: Search `QuizGenerator/constants.py` and existing question files
- **ContentAST usage**: Best examples in memory and process questions

### 2. **Testing Patterns**
- **Quick iteration**: Use `example_files/scratch.yaml` with 1-2 question instances
- **Canvas testing**: Manual upload and verification (Canvas is picky about MathJax)
- **PDF testing**: Generate PDFs locally to catch LaTeX errors early

### 3. **Key File Locations**
```
QuizGenerator/
├── question.py              # Question registry and base classes
├── misc.py                  # ContentAST system and Answer types
├── constants.py             # Standard ranges and probabilities
├── premade_questions/
│   ├── cst334/              # Existing question examples
│   └── cst463/              # New question categories
└── example_files/
    └── scratch.yaml         # Testing configuration
```

## Development Workflow That Worked

1. **Start with structure**: Create base class and basic question skeleton
2. **Test early**: Add to scratch.yaml and test loading with `python generate_quiz.py TEST`
3. **Implement incrementally**: One question type at a time, test each before moving on
4. **Canvas testing**: Upload manually and verify MathJax rendering
5. **PDF testing**: Generate PDFs to catch LaTeX issues
6. **Refinement**: Polish explanations, formatting, and edge cases

## Common Pitfalls to Avoid

### 1. **Directory Naming**
- Don't use hyphens in Python module names (`math-and-data` → `math_and_data`)
- Ensure `__init__.py` files exist and import new classes

### 2. **ContentAST Usage**
- Don't mix individual ContentAST elements in equations (creates fragmented rendering)
- Use `ContentAST.OnlyHtml()` for Canvas-specific elements (like answer tables)
- Remember that `ContentAST.Answer()` expects `Answer` objects, not raw values

### 3. **Answer Handling**
- Use `Answer.integer()`, `Answer.string()` factory methods instead of raw constructor
- For conditional answers (like impossible matrix operations), use `Answer.string("key", "-")`

### 4. **Mathematical Notation**
- Test both `\times` and `\cdot` - user preferences may vary
- Use `bmatrix` for consistency with other math content
- Remember that Canvas MathJax and LaTeX PDF may render slightly differently

## Recommended Next Steps for Future Questions

### 1. **Before Starting**
- Review this document and identify similar question types for reference
- Check if new ContentAST elements or Answer types are needed
- Plan the question structure (base class vs individual classes)

### 2. **During Development**
- Follow the incremental testing workflow
- Test both output formats (Canvas + PDF) at each major milestone
- Keep explanations pedagogically focused with step-by-step breakdowns

### 3. **Quality Checklist**
- [ ] Questions load properly via registry
- [ ] Canvas MathJax renders correctly
- [ ] PDF LaTeX compiles without errors
- [ ] Answer validation works for all edge cases
- [ ] Explanations are comprehensive and educational
- [ ] Code follows project conventions (see CLAUDE.md)

## Tools and Commands Reference

```bash
# Test question loading
python generate_quiz.py TEST

# Generate Canvas quizzes for testing
python generate_quiz.py --num_canvas 3 --course_id 12345

# Generate PDFs for testing
python generate_quiz.py --num_pdfs 2

# Custom configuration testing
python generate_quiz.py --quiz_yaml example_files/scratch.yaml --num_pdfs 1
```

## Final Notes

The matrix questions implementation was successful due to:
1. **Systematic approach**: Following established patterns and building incrementally
2. **Early testing**: Catching issues before they compounded
3. **Reference-driven development**: Learning from existing question implementations
4. **User feedback integration**: Iterating based on Canvas/PDF testing results

The most time was spent on Canvas MathJax compatibility and PDF LaTeX setup - these should be prioritized for testing in future question development.