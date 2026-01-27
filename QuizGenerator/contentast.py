from __future__ import annotations

import abc
import enum
import textwrap
from io import BytesIO
from typing import List, Callable

import pypandoc
import markdown

from QuizGenerator.qrcode_generator import QuestionQRCode
import re

import logging
import os
import uuid
import itertools
import math
import decimal
import fractions
import numpy as np

log = logging.getLogger(__name__)


"""
Content Abstract Syntax Tree - The core content system for quiz generation.

IMPORTANT: ALWAYS use content AST elements for ALL content generation.
Never create custom LaTeX, HTML, or Markdown strings manually.

This system provides cross-format compatibility between:
- LaTeX/PDF output for printed exams
- HTML/Canvas output for online quizzes
- Markdown for documentation

Key Components:
- Section: Container for groups of elements (use for get_body/get_explanation)
- Paragraph: Text blocks that automatically handle spacing
- Equation: Mathematical equations with proper LaTeX/MathJax rendering
- Matrix: Mathematical matrices (DON'T use manual \\begin{bmatrix})
- Table: Data tables with proper formatting
- Answer: Answer input fields
- OnlyHtml/OnlyLatex: Platform-specific content

Examples:
  # Good - uses content AST
  body = Section()
  body.add_element(Paragraph(["Calculate the matrix:"]))
  matrix_data = [[1, 2], [3, 4]]
  body.add_element(Matrix(data=matrix_data, bracket_type="b"))

  # Bad - manual LaTeX (inconsistent, error-prone)
  body.add_element(Text("\\\\begin{bmatrix} 1 & 2 \\\\\\\\ 3 & 4 \\\\end{bmatrix}"))
"""

class OutputFormat(enum.StrEnum):
  HTML = "html"
  TYPST = "typst"
  LATEX = "latex"
  MARKDOWN = "markdown"
  
class Element(abc.ABC):
  """
  Base class for all content AST elements providing cross-format rendering.

  This is the foundation class that all content AST elements inherit from.
  It provides the core rendering infrastructure that enables consistent
  output across LaTeX/PDF, HTML/Canvas, and Markdown formats.

  Key Features:
  - Cross-format rendering (markdown, html, latex)
  - Automatic format conversion via pypandoc
  - Element composition and nesting
  - Consistent spacing and formatting

  When to inherit from Element:
  - Creating new content types that need multi-format output
  - Building container elements that hold other elements
  - Implementing custom rendering logic for specific content types

  Example usage:
      # Most elements inherit from this automatically
      section = Section()
      section.add_element(Text("Hello world"))
      section.add_element(Equation("x = 5"))

      # Renders to any format
      latex_output = section.render("latex")
      html_output = section.render("html")
  """
  def __init__(self, elements=None, add_spacing_before=False):
    pass
    # self.elements : List[Element] = [
    #   e if isinstance(e, Element) else Text(e)
    #   for e in (elements if elements else [])
    # ]
    # self.add_spacing_before = add_spacing_before
  
  def __str__(self):
    return self.render_markdown()
  
  def render(self, output_format : OutputFormat, **kwargs) -> str:
    # Render using the appropriate method, if it exists
    method_name = f"render_{output_format}"
    if hasattr(self, method_name):
      return getattr(self, method_name)(**kwargs)
    
    return self.render_markdown(**kwargs)  # Fallback to markdown
  
  @abc.abstractmethod
  def render_markdown(self, **kwargs):
    pass
  
  @abc.abstractmethod
  def render_html(self, **kwargs):
    pass
  
  @abc.abstractmethod
  def render_latex(self, **kwargs):
    pass

  @abc.abstractmethod
  def render_typst(self, **kwargs):
    pass

  def is_mergeable(self, other: Element):
    return False

class Container(Element):
  """Elements that contain other elements.  Generally are formatting of larger pieces."""
  def __init__(self, elements=None, **kwargs):
    super().__init__(**kwargs)
    self.elements : List[Element] = elements if elements is not None else []
  
  def add_element(self, element):
    self.elements.append(element)
  
  def add_elements(self, elements):
    self.elements.extend(elements)
  
  @staticmethod
  def render_element(element, output_format: OutputFormat, **kwargs):
    if isinstance(element, Element):
      return element.render(output_format, **kwargs)
    log.warning(f"Element ({element}) is not Element.  Defaulting to forcing to a string.")
    return f"{element}"
  
  def render_markdown(self, **kwargs):
    return " ".join([
      self.render_element(element, output_format=OutputFormat.MARKDOWN, **kwargs)
      for element in self.elements
    ])

  def render_html(self, **kwargs):
    for element in self.elements:
      log.debug(f"element: {element}")
    return " ".join([
      self.render_element(element, output_format=OutputFormat.HTML, **kwargs)
      for element in self.elements
    ])

  def render_latex(self, **kwargs):
    return "".join([
      self.render_element(element, output_format=OutputFormat.LATEX, **kwargs)
      for element in self.elements
    ])
    
    latex = "".join(element.render("latex", **kwargs) for element in self.elements)
    return f"{'\n\n\\vspace{0.5cm}' if self.add_spacing_before else ''}{latex}"
  
  def render_typst(self, **kwargs):

    return " ".join([
      self.render_element(element, output_format=OutputFormat.TYPST, **kwargs)
      for element in self.elements
    ])
    
    """
    Default Typst rendering using markdown → typst conversion via pandoc.

    This provides instant Typst support for all content AST elements without
    needing explicit implementations. Override this method in subclasses
    when pandoc conversion quality is insufficient or Typst-specific
    features are needed.
    """
    # Render to markdown first
    markdown_content = self.render_markdown(**kwargs)
    
    # Convert markdown to Typst via pandoc
    typst_content = self.convert_markdown(markdown_content, 'typst')
    
    # Add spacing if needed (Typst equivalent of \vspace)
    if self.add_spacing_before:
      return f"\n{typst_content}"
    
    return typst_content if typst_content else markdown_content

class Leaf(Element):
  """Elements that are just themselves."""
  def __init__(self, content : str, **kwargs):
    super().__init__(**kwargs)
    self.content = content

  @staticmethod
  def convert_markdown(str_to_convert, output_format : OutputFormat):
    try:
      match output_format:
        
        case OutputFormat.MARKDOWN:
          return str_to_convert
        
        case OutputFormat.HTML:
          html_output = markdown.markdown(str_to_convert)
          
          # Strip surrounding <p> tags so we can control paragraphs
          if html_output.startswith("<p>") and html_output.endswith("</p>"):
            html_output = html_output[3:-4]
          
          return html_output.strip()
        
        case _:
          output = pypandoc.convert_text(
            str_to_convert,
            output_format,
            format='md',
            extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"]
          )
          return output
    except Exception as e:
      log.warning(f"Specified conversion failed. Defaulting to markdown")
      log.warning(e)
    
    return str(str_to_convert)
    
  def render_markdown(self, **kwargs):
    return self.convert_markdown(self.content, OutputFormat.MARKDOWN)
  
  def render_html(self, **kwargs):
    return self.convert_markdown(self.content, OutputFormat.HTML)
  
  def render_latex(self, **kwargs):
    return self.convert_markdown(self.content, OutputFormat.LATEX)
  
  def render_typst(self, **kwargs):
    return self.convert_markdown(self.content, OutputFormat.TYPST) #.replace("#", r"\#")

## Top-ish Level containers
class Document(Container):
  """
  Root document container for complete quiz documents with proper headers and structure.

  This class provides document-level rendering with appropriate headers, packages,
  and formatting for complete LaTeX documents. It's primarily used internally
  by the quiz generation system.

  When to use:
  - Creating standalone PDF documents (handled automatically by quiz system)
  - Need complete LaTeX document structure with packages and headers
  - Root container for entire quiz content

  Note: Most question developers should NOT use this directly.
  Use Section for question bodies and explanations instead.

  Features:
  - Complete LaTeX document headers with all necessary packages
  - Automatic title handling across all formats
  - PDF-ready formatting with proper spacing and layout

  Example (internal use):
      # Usually created automatically by quiz system
      doc = Document(title="Midterm Exam")
      doc.add_element(question_section)
      pdf_content = doc.render("latex")
  """
  
  LATEX_HEADER = textwrap.dedent(r"""
  \documentclass[12pt]{article}

  % Page layout
  \usepackage[a4paper, margin=1.5cm]{geometry}

  % Graphics for QR codes
  \usepackage{graphicx}       % For including QR code images

  % Math packages
  \usepackage[leqno,fleqn]{amsmath}        % For advanced math environments (matrices, equations)
  \setlength{\mathindent}{0pt}  % flush left
  \usepackage{amsfonts}       % For additional math fonts
  \usepackage{amssymb}        % For additional math symbols

  % Tables and formatting
  \usepackage{booktabs}       % For clean table rules
  \usepackage{array}          % For extra column formatting options
  \usepackage{verbatim}       % For verbatim environments (code blocks)
  \usepackage{enumitem}       % For customized list spacing
  \usepackage{setspace}       % For \onehalfspacing

  % Setting up Code environments
  \let\originalverbatim\verbatim
  \let\endoriginalverbatim\endverbatim
  \renewenvironment{verbatim}
    {\small\setlength{\baselineskip}{0.8\baselineskip}\originalverbatim}
    {\endoriginalverbatim}

  % Listings (for code)
  \usepackage[final]{listings}
  \lstset{
    basicstyle=\ttfamily,
    columns=fullflexible,
    frame=single,
    breaklines=true,
    postbreak=\mbox{$\hookrightarrow$\,} % You can remove or customize this
  }

  % Custom commands
  \newcounter{NumQuestions}
  \newcommand{\question}[1]{%
    \vspace{0.5cm}
    \stepcounter{NumQuestions}%
    \noindent\textbf{Question \theNumQuestions:} \hfill \rule{0.5cm}{0.15mm} / #1
    \par\vspace{0.1cm}
  }
  \newcommand{\answerblank}[1]{\rule{0pt}{10mm}\rule[-1.5mm]{#1cm}{0.15mm}}

  % Optional: spacing for itemized lists
  \setlist[itemize]{itemsep=10pt, parsep=5pt}
  \providecommand{\tightlist}{%
    \setlength{\itemsep}{10pt}\setlength{\parskip}{10pt}
  }

  \begin{document}
  """)

  TYPST_HEADER = textwrap.dedent("""
  #import "@preview/wrap-it:0.1.1": wrap-content

  // Quiz document settings
  #set page(
    paper: "us-letter",
    margin: 1.5cm,
  )

  #set text(
    size: 12pt,
  )

  // Math equation settings
  #set math.equation(numbering: none)

  // Paragraph spacing
  #set par(
    spacing: 1.0em,
    leading: 0.5em,
  )

  // Question counter and command
  #let question_num = counter("question")

  #let question(points, content, spacing: 3cm, qr_code: none) = {
    block(breakable: false)[
      #line(length: 100%, stroke: 1pt)
      #v(0cm)
      #question_num.step()

      *Question #context question_num.display():* (#points #if points == 1 [point] else [points])
      #v(0.0cm)

      /*
      #if qr_code != none {
        let fig = figure(image(qr_code, width: 2cm))
        // let fig = square(fill: teal, radius: 0.5em, width: 8em) // for debugging
        wrap-content(fig, align: top + right)[
          #h(100%)    // force the wrapper to fill line width
          #content
        ]
      } else {
        content
      }
      */
      
      #grid(
        columns: (1fr, auto),
        gutter: 1em,
        align: top,
      )[
        #content
        #v(spacing)
      ][
        #image(qr_code, width: 2cm)
      ]
      #if spacing >= 199cm {
      
        "Note: the next page is left blank for you to show work."
      }

    ]
      // Check if spacing >= 199cm (EXTRA_PAGE preset)
      // If so, add both spacing and a pagebreak for a full blank page
      if spacing >= 199cm {
        
        pagebreak()
        pagebreak()
      }
  }

  // Fill-in line for inline answer blanks (tables, etc.)
  #let fillline(width: 5cm, height: 1.2em, stroke: 0.5pt) = {
    box(width: width, height: height, baseline: 0.25em)[
      #align(bottom + left)[
        #line(length: 100%, stroke: stroke)
      ]
    ]
  }

  // Code block styling
  #show raw.where(block: true): set text(size: 8pt)
  #show raw.where(block: true): block.with(
    fill: luma(240),
    inset: 10pt,
    radius: 4pt,
  )
  """)
  
  def __init__(self, title=None):
    super().__init__()
    self.title = title
  
  def render(self, output_format, **kwargs):
    # Generate content from all elements
    content = super().render(output_format, **kwargs)
    
    # Add title if present
    if self.title and output_format == "markdown":
      content = f"# {self.title}\n\n{content}"
    elif self.title and output_format == "html":
      content = f"<h1>{self.title}</h1>\n{content}"
    elif self.title and output_format == "latex":
      content = f"\\section{{{self.title}}}\n{content}"
    
    return content
  
  def render_latex(self, **kwargs):
    latex = self.LATEX_HEADER
    latex += f"\\title{{{self.title}}}\n"
    latex += textwrap.dedent(f"""
      \\noindent\\Large {self.title} \\hfill \\normalsize Name: \\answerblank{{{5}}}

      \\vspace{{0.5cm}}
      \\onehalfspacing

    """)

    latex += "\n".join(element.render(OutputFormat.LATEX, **kwargs) for element in self.elements)

    latex += r"\end{document}"

    return latex

  def render_typst(self, **kwargs):
    """Render complete Typst document with header and title"""
    typst = self.TYPST_HEADER

    # Add title and name line using grid for proper alignment
    typst += f"\n#grid(\n"
    typst += f"  columns: (1fr, auto),\n"
    typst += f"  align: (left, right),\n"
    typst += f"  [#text(size: 14pt, weight: \"bold\")[{self.title}]],\n"
    typst += f"  [Name: #fillline(width: 5cm)]\n"
    typst += f")\n"
    typst += f"#v(0.5cm)\n"

    # Render all elements
    typst += "".join(element.render(OutputFormat.TYPST, **kwargs) for element in self.elements)
    
    return typst

class Question(Container):
  """
  Complete question container with body, explanation, and metadata.

  This class represents a full question with both the question content
  and its explanation/solution. It handles question-level formatting
  like point values, spacing, and PDF layout.

  Note: Most question developers should NOT use this directly.
  It's created automatically by the quiz generation system.
  Focus on building Section objects for get_body() and get_explanation().

  When to use:
  - Creating complete question objects (handled by quiz system)
  - Custom question wrappers (advanced use)

  Example (internal use):
      # Usually created by quiz system from your question classes
      body = Section()
      body.add_element(Paragraph(["What is 2+2?"]))

      explanation = Section()
      explanation.add_element(Paragraph(["2+2=4"]))

      question = Question(body=body, explanation=explanation, value=5)
  """
  
  def __init__(
      self,
      body: Section,
      explanation: Section,
      name=None,
      value=1,
      interest=1.0,
      spacing=0,
      topic=None,
      question_number=None,
      **kwargs
  ):
    super().__init__()
    self.name = name
    self.explanation = explanation
    self.body = body
    self.value = value
    self.interest = interest
    self.spacing = spacing
    self.topic = topic  # todo: remove this bs.
    self.question_number = question_number  # For QR code generation
    
    self.default_kwargs = kwargs
  
  def render(self, output_format, **kwargs):
    updated_kwargs = self.default_kwargs
    updated_kwargs.update(kwargs)
    
    log.debug(f"updated_kwargs: {updated_kwargs}")
    
    # Special handling for latex and typst - use dedicated render methods
    if output_format == "typst":
      return self.render_typst(**kwargs)
    
    # Generate content from all elements
    content = self.body.render(output_format, **updated_kwargs)
    
    # If output format is latex, add in minipage and question environments
    if output_format == "latex":
      # Build question header - either with or without QR code
      if self.question_number is not None:
        try:
          from QuizGenerator.qrcode_generator import QuestionQRCode
          
          # Build extra_data dict with regeneration metadata if available
          extra_data = {}
          if hasattr(self, 'question_class_name') and hasattr(self, 'generation_seed') and hasattr(
              self, 'question_version'
          ):
            if self.question_class_name and self.generation_seed is not None and self.question_version:
              extra_data['question_type'] = self.question_class_name
              extra_data['seed'] = self.generation_seed
              extra_data['version'] = self.question_version
              # Include question-specific configuration parameters if available
              if hasattr(self, 'config_params') and self.config_params:
                extra_data['config'] = self.config_params
          
          qr_path = QuestionQRCode.generate_qr_pdf(
            self.question_number,
            self.value,
            **extra_data
          )
          # Build custom question header with QR code centered
          # Format: Question N:  [QR code centered]  __ / points
          question_header = (
              r"\vspace{0.5cm}" + "\n"
                                  r"\noindent\textbf{Question " + str(self.question_number) + r":} \hfill "
                                                                                              r"\rule{0.5cm}{0.15mm} / " + str(
            int(self.value)
          ) + "\n"
              r"\raisebox{-1cm}{"  # Reduced lift to minimize extra space above
              rf"\includegraphics[width={QuestionQRCode.DEFAULT_SIZE_CM}cm]{{{qr_path}}}"
              r"} "
              r"\par\vspace{-1cm}"
          )
        except Exception as e:
          log.warning(f"Failed to generate QR code for question {self.question_number}: {e}")
          # Fall back to standard question macro
          question_header = r"\question{" + str(int(self.value)) + r"}"
      else:
        # Use standard question macro if no question number
        question_header = r"\question{" + str(int(self.value)) + r"}"
      
      latex_lines = [
        r"\noindent\begin{minipage}{\textwidth}",
        r"\noindent\makebox[\linewidth]{\rule{\paperwidth}{1pt}}",
        question_header,
        r"\noindent\begin{minipage}{0.9\textwidth}",
        content,
        f"\\vspace{{{self.spacing}cm}}"
        r"\end{minipage}",
        r"\end{minipage}",
        "\n\n",
      ]
      content = '\n'.join(latex_lines)
    
    log.debug(f"content: \n{content}")
    
    return content
  
  def render_typst(self, **kwargs):
    """Render question in Typst format with proper formatting"""
    # Render question body
    content = self.body.render(OutputFormat.TYPST, **kwargs)
    
    # Generate QR code if question number is available
    qr_param = ""
    if self.question_number is not None:
      try:
        
        # Build extra_data dict with regeneration metadata if available
        extra_data = {}
        if hasattr(self, 'question_class_name') and hasattr(self, 'generation_seed') and hasattr(
            self, 'question_version'
        ):
          if self.question_class_name and self.generation_seed is not None and self.question_version:
            extra_data['question_type'] = self.question_class_name
            extra_data['seed'] = self.generation_seed
            extra_data['version'] = self.question_version
            # Include question-specific configuration parameters if available
            if hasattr(self, 'config_params') and self.config_params:
              extra_data['config'] = self.config_params
        
        # Generate QR code PNG
        qr_path = QuestionQRCode.generate_qr_pdf(
          self.question_number,
          self.value,
          scale=1,
          **extra_data
        )
        
        # Add QR code parameter to question function call
        qr_param = f'qr_code: "{qr_path}"'
      
      except Exception as e:
        log.warning(f"Failed to generate QR code for question {self.question_number}: {e}")
    
    # Use the question function which handles all formatting including non-breaking
    return textwrap.dedent(f"""
    #question(
        {int(self.value)},
        spacing: {self.spacing}cm{'' if not qr_param else ", "}
        {qr_param}
      )[
    """) + content + "\n]\n\n"

class Section(Container):
  """
  Primary container for question content - USE THIS for get_body() and get_explanation().

  This is the most important content AST class for question developers.
  It serves as the main container for organizing question content
  and should be the return type for your get_body() and get_explanation() methods.

  CRITICAL: Always use Section as the container for:
  - Question body content (return from get_body())
  - Question explanation/solution content (return from get_explanation())
  - Any grouped content that needs to render together

  When to use:
  - As the root container in get_body() and get_explanation() methods
  - Grouping related content elements
  - Organizing complex question content

  Example:
      def _get_body(self):
          body = Section()
          answers = []
          body.add_element(Paragraph(["Calculate the determinant:"]))

          matrix_data = [[1, 2], [3, 4]]
          body.add_element(Matrix(data=matrix_data, bracket_type="v"))

          # Answer extends Leaf - add directly to body
          ans = Answer.integer("det", self.determinant, label="Determinant")
          answers.append(ans)
          body.add_element(ans)
          return body, answers
  """
  pass

# Individual elements
class Text(Leaf):
  """
  Basic text content with automatic format conversion and selective visibility.

  This is the fundamental text element that handles plain text content
  with automatic markdown-to-format conversion. It supports emphasis
  and format-specific hiding.

  When to use:
  - Plain text content that needs cross-format rendering
  - Text that should be hidden from specific output formats
  - Simple text with optional emphasis

  DON'T use for:
  - Mathematical content (use Equation instead)
  - Code (use Code instead)
  - Structured content (use Paragraph for grouping)

  Example:
      # Basic text
      text = Text("This is plain text")

      # Emphasized text
      important = Text("Important note", emphasis=True)

      # HTML-only text (hidden from PDF)
      web_note = Text("Click submit", hide_from_latex=True)
  """
  def __init__(self, content : str, *, hide_from_latex=False, hide_from_html=False, emphasis=False):
    super().__init__(content)
    self.hide_from_latex = hide_from_latex
    self.hide_from_html = hide_from_html
    self.emphasis = emphasis
  
  def render_markdown(self, **kwargs):
    return f"{'***' if self.emphasis else ''}{self.content}{'***' if self.emphasis else ''}"

  def render_html(self, **kwargs):
    if self.hide_from_html:
      return ""
    return self.convert_markdown(self.content,OutputFormat.HTML)
    
  def render_latex(self, **kwargs):
    if self.hide_from_latex:
      return ""
    return self.convert_markdown(self.content.replace("#", r"\#"), OutputFormat.LATEX)

  def render_typst(self, **kwargs):
    """Render text to Typst, escaping special characters."""
    if self.hide_from_latex:
      return ""

    # Extract code blocks, render them, and replace with placeholders
    # This prevents the # escaping from affecting content inside code blocks
    code_blocks = []

    def save_code_block(match):
      code_content = match.group(1).strip()
      # Escape quotes for Typst raw() function
      escaped_content = code_content.replace('"', r'\"')
      rendered_block = f"""
      #box(
        raw("{escaped_content}",
          block: true
        )
      )
      """
      placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
      code_blocks.append(rendered_block)
      return placeholder

    # Replace code blocks with placeholders
    content = re.sub(
      r"```\s*(.*)\s*```",
      save_code_block,
      self.content,
      flags=re.DOTALL
    )

    # In Typst, # starts code/function calls, so we need to escape it in regular text
    # (but not in code blocks, which are now placeholders)
    content = content.replace("# ", r"\# ")

    # Restore code blocks
    for i, block in enumerate(code_blocks):
      content = content.replace(f"__CODE_BLOCK_{i}__", block)

    if self.emphasis:
      content = f"*{content}*"
    return content

  def is_mergeable(self, other: Element):
    if not isinstance(other, Text):
      return False
    if self.hide_from_latex != other.hide_from_latex:
      return False
    return True
  
  def merge(self, other: Text):
    self.content = self.render_markdown() + " " + other.render_markdown()
    self.emphasis = False

class Code(Text):
  """
  Code block formatter with proper syntax highlighting and monospace formatting.

  Use this for displaying source code, terminal output, file contents,
  or any content that should appear in monospace font with preserved formatting.

  When to use:
  - Source code examples
  - Terminal/shell output
  - File contents or configuration
  - Any monospace-formatted text

  Features:
  - Automatic code block formatting in markdown
  - Proper HTML code styling
  - LaTeX verbatim environments
  - Preserved whitespace and line breaks

  Example:
      # Code snippet
      code_block = Code(
          "if (x > 0) {\n    print('positive');\n}"
      )
      body.add_element(code_block)

      # Terminal output
      terminal = Code("$ ls -la\ntotal 24\ndrwxr-xr-x 3 user")
  """
  def __init__(self, lines, **kwargs):
    super().__init__(lines)
    self.make_normal = kwargs.get("make_normal", False)
  
  def render_markdown(self, **kwargs):
    content = "```" + self.content.rstrip() + "\n```"
    return content
  
  def render_html(self, **kwargs):
    return self.convert_markdown(textwrap.indent(self.content, "\t"), OutputFormat.HTML)
  
  def render_latex(self, **kwargs):
    return self.convert_markdown(self.render_markdown(), OutputFormat.LATEX)

  def render_typst(self, **kwargs):
    """Render code block in Typst with smaller monospace font."""
    # Use raw block with 11pt font size
    # Escape backticks in the content
    escaped_content = self.content.replace("`", r"\`")
    
    # Try to reduce individual pathway to ensure consistency
    return Text(f"```\n{escaped_content.rstrip()}\n```").render_typst()

class Equation(Leaf):
  """
  Mathematical equation renderer with LaTeX input and cross-format output.

  CRITICAL: Use this for ALL mathematical content instead of manual LaTeX strings.
  Provides consistent math rendering across PDF (LaTeX) and Canvas (MathJax).

  When to use:
  - Any mathematical expressions, equations, or formulas
  - Variables, functions, mathematical notation
  - Both inline math (within text) and display math (separate lines)

  DON'T manually write LaTeX in Text - always use Equation.

  Example:
      # Display equation (separate line, larger)
      body.add_element(Equation("x^2 + y^2 = r^2"))

      # Inline equation (within text)
      paragraph = Paragraph([
          "The solution is ",
          Equation("x = \\frac{-b}{2a}", inline=True),
          " which can be computed easily."
      ])

      # Complex equations
      body.add_element(Equation(r"\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}"))
  """
  def __init__(self, latex, inline=False):
    super().__init__("[equation]")
    self.latex = latex
    self.inline = inline
  
  def render_markdown(self, **kwargs):
    if self.inline:
      return f"${self.latex}$"
    else:
      return r"$\displaystyle " + f"{self.latex}" + r"$"

  def render_html(self, **kwargs):
    if self.inline:
      return fr"\({self.latex}\)"
    else:
      return f"<div class='math'>$$ \\displaystyle {self.latex} \\; $$</div>"

  def render_latex(self, **kwargs):
    if self.inline:
      return f"${self.latex}$~"
    else:
      return f"\\begin{{flushleft}}${self.latex}$\\end{{flushleft}}"

  def render_typst(self, **kwargs):
    """
    Render equation in Typst format.

    Typst uses LaTeX-like math syntax with $ delimiters, but with different
    symbol names. This method converts LaTeX math to Typst-compatible syntax.
    Inline: $equation$
    Display: $ equation $
    """
    # Convert LaTeX to Typst-compatible math
    typst_math = self._latex_to_typst(self.latex)

    if self.inline:
      # Inline math in Typst
      return f"${typst_math}$"
    else:
      # Display math in Typst
      return f"$ {typst_math} $"

  @staticmethod
  def _latex_to_typst(latex_str: str) -> str:
    r"""
    Convert LaTeX math syntax to Typst math syntax.

    Typst uses different conventions:
    - Greek letters: 'alpha' not '\alpha'
    - No \left/\right: auto-sizing parentheses
    - Operators: 'nabla' not '\nabla', 'times' not '\times'
    """

    # Remove \left and \right (Typst uses auto-sizing)
    latex_str = latex_str.replace(r'\left', '').replace(r'\right', '')
    
    # Hat Notation
    latex_str = re.sub(r'\\hat{([^}]+)}', r'hat("\1")', latex_str)  # \hat{...} -> hat(...)

    # Convert subscripts and superscripts from LaTeX to Typst
    # LaTeX uses braces: b_{out}, x_{10}, x^{2}
    # Typst uses parentheses for multi-char: b_(out), x_(10), x^(2)
    # Multi-character text subscripts need quotes: L_{base} -> L_("base")
    # But numbers don't: x_{10} -> x_(10)
    def convert_sub_super(match):
      content = match.group(1)
      prefix = match.group(0)[0]  # '_' or '^'
      # If it's purely numeric or a single char, no quotes needed
      if content.isdigit() or len(content) == 1:
        return f'{prefix}({content})'
      # If it's multi-char text, quote it
      return f'{prefix}("{content}")'

    latex_str = re.sub(r'_{([^}]+)}', convert_sub_super, latex_str)
    latex_str = re.sub(r'\^{([^}]+)}', convert_sub_super, latex_str)

    # Convert LaTeX Greek letters to Typst syntax (remove backslash)
    greek_letters = [
      'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
      'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
      'tau', 'phi', 'chi', 'psi', 'omega',
      'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega'
    ]

    for letter in greek_letters:
      # Use word boundaries to avoid replacing parts of other commands
      latex_str = re.sub(rf'\\{letter}\b', letter, latex_str)

    # Convert LaTeX operators to Typst syntax
    latex_str = latex_str.replace(r'\nabla', 'nabla')
    latex_str = latex_str.replace(r'\times', 'times')
    latex_str = latex_str.replace(r'\cdot', 'dot')
    latex_str = latex_str.replace(r'\partial', 'partial')
    latex_str = latex_str.replace(r'\sum', 'sum')
    latex_str = latex_str.replace(r'\prod', 'product')
    latex_str = latex_str.replace(r'\int', 'integral')
    latex_str = latex_str.replace(r'\ln', 'ln')
    latex_str = latex_str.replace(r'\log', 'log')
    latex_str = latex_str.replace(r'\exp', 'exp')
    latex_str = latex_str.replace(r'\sin', 'sin')
    latex_str = latex_str.replace(r'\cos', 'cos')
    latex_str = latex_str.replace(r'\tan', 'tan')
    latex_str = latex_str.replace(r'\max', 'max')
    latex_str = latex_str.replace(r'\min', 'min')
    latex_str = latex_str.replace(r'\sqrt', 'sqrt')
    # Convert \text{...} to "..." for Typst
    latex_str = re.sub(r'\\text\{([^}]*)\}', r'"\1"', latex_str)
    # Convert \frac{a}{b} to frac(a, b) for Typst
    latex_str = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'frac(\1, \2)', latex_str)

    # Handle matrix environments (bmatrix, pmatrix, vmatrix, Vmatrix, Bmatrix, matrix)
    # Map bracket types to Typst delimiters
    bracket_map = {
      'bmatrix': '\"[\"',   # square brackets
      'pmatrix': '\"(\"',   # parentheses (default)
      'vmatrix': '\"|\"',   # single vertical bars (determinant)
      'Vmatrix': '\"||\"',  # double vertical bars (norm)
      'Bmatrix': '\"{\"',   # curly braces
      'matrix': None,       # no brackets
    }

    for env_type, delim in bracket_map.items():
      pattern = rf'\\begin\{{{env_type}\}}(.*?)\\end\{{{env_type}\}}'

      def make_replacer(delimiter):
        def replace_matrix(match):
          content = match.group(1)
          # Split rows by \\ and columns by &
          rows = content.split(r'\\')
          rows = [r.strip() for r in rows if r.strip()]

          # Check if it's a vector (single column) or matrix
          is_vector = all('&' not in row for row in rows)

          if is_vector:
            # Single column - use semicolons to separate rows in mat()
            elements = "; ".join(rows)
          else:
            # Multiple columns - replace & with , and rows with ;
            formatted_rows = [row.replace('&', ',').strip() for row in rows]
            elements = "; ".join(formatted_rows)

          if delimiter:
            return f"mat(delim: {delimiter}, {elements})"
          else:
            return f"mat({elements})"
        return replace_matrix

      latex_str = re.sub(pattern, make_replacer(delim), latex_str, flags=re.DOTALL)

    # Handle \left\| ... \right\| for norms
    latex_str = re.sub(r'\\\|', '||', latex_str)

    return latex_str

  @classmethod
  def make_block_equation__multiline_equals(cls, lhs : str, rhs : List[str]):
    equation_lines = []
    equation_lines.extend([
      r"\begin{array}{l}",
      f"{lhs} = {rhs[0]} \\\\",
    ])
    equation_lines.extend([
      f"\\phantom{{{lhs}}} = {eq} \\\\"
      for eq in rhs[1:]
    ])
    equation_lines.extend([
      r"\end{array}",
    ])
    
    return cls('\n'.join(equation_lines))

class MathExpression(Leaf):
  """
  Compose multiple math elements into a single format-independent expression.

  This allows mixing content AST elements (like Matrix) with math operators
  and symbols, with each part rendering appropriately for the target format.

  Example:
      # Vector magnitude: ||v|| =
      body.add_element(MathExpression([
          "||",
          Matrix(data=[[1], [2], [3]], bracket_type="b"),
          "|| = "
      ]))

      # Vector addition: a + b =
      body.add_element(MathExpression([
          Matrix(data=[[1], [2]], bracket_type="b"),
          " + ",
          Matrix(data=[[3], [4]], bracket_type="b"),
          " = "
      ]))

  Parts can be:
      - Strings: rendered as-is (math operators, symbols, etc.)
      - content AST elements: call their render method for the target format
  """
  def __init__(self, parts, inline=False):
    super().__init__("[math_expression]")
    self.parts = parts  # List of strings and/or content AST elements
    self.inline = inline

  def _render_parts(self, output_format, **kwargs):
    """Render all parts for the given output format."""
    rendered = []
    for part in self.parts:
      if isinstance(part, str):
        # Convert LaTeX operators to Typst if needed
        if output_format == OutputFormat.TYPST:
          part = part.replace(r'\cdot', ' dot ')
          part = part.replace(r'\times', ' times ')
          part = part.replace(r'\div', ' div ')
        rendered.append(part)
      elif isinstance(part, Element):
        # Use dedicated math_content methods if available (cleaner than stripping delimiters)
        if output_format == OutputFormat.HTML:
          # For HTML (MathJax), use LaTeX math content
          if hasattr(part, 'math_content_latex'):
            rendered.append(part.math_content_latex())
          else:
            # Fallback: try to extract from render_html
            html = part.render_html(**kwargs)
            html = re.sub(r"<[^>]+>", "", html)
            html = re.sub(r"^\$\$?\s*\\displaystyle\s*", "", html)
            html = re.sub(r"^\$\$?\s*", "", html)
            html = re.sub(r"\s*\$\$?$", "", html)
            rendered.append(html)
        elif output_format == OutputFormat.TYPST:
          if hasattr(part, 'math_content_typst'):
            rendered.append(part.math_content_typst())
          else:
            # Fallback: try to extract from render_typst
            typst = part.render_typst(**kwargs)
            typst = re.sub(r"^\s*\$\s*", "", typst)
            typst = re.sub(r"\s*\$\s*$", "", typst)
            rendered.append(typst)
        elif output_format == OutputFormat.LATEX:
          if hasattr(part, 'math_content_latex'):
            rendered.append(part.math_content_latex())
          else:
            latex = part.render_latex(**kwargs)
            latex = re.sub(r"^\\begin\{flushleft\}", "", latex)
            latex = re.sub(r"\\end\{flushleft\}$", "", latex)
            latex = re.sub(r"^\\\[", "", latex)
            latex = re.sub(r"\\\]$", "", latex)
            latex = re.sub(r"^\$", "", latex)
            latex = re.sub(r"\$~?$", "", latex)
            rendered.append(latex)
        elif output_format == OutputFormat.MARKDOWN:
          if hasattr(part, 'math_content_latex'):
            rendered.append(part.math_content_latex())
          else:
            md = part.render_markdown(**kwargs)
            md = re.sub(r"^\$", "", md)
            md = re.sub(r"\$$", "", md)
            rendered.append(md)
      else:
        # Convert to string as fallback
        rendered.append(str(part))
    return "".join(rendered)

  def render_markdown(self, **kwargs):
    content = self._render_parts(OutputFormat.MARKDOWN, **kwargs)
    if self.inline:
      return f"${content}$"
    else:
      return f"$\\displaystyle {content}$"

  def render_html(self, **kwargs):
    content = self._render_parts(OutputFormat.HTML, **kwargs)
    if self.inline:
      return f"\\({content}\\)"
    else:
      return f"<div class='math'>$$ \\displaystyle {content} \\; $$</div>"

  def render_latex(self, **kwargs):
    content = self._render_parts(OutputFormat.LATEX, **kwargs)
    if self.inline:
      return f"${content}$~"
    else:
      return f"\\begin{{flushleft}}${content}$\\end{{flushleft}}"

  def render_typst(self, **kwargs):
    content = self._render_parts(OutputFormat.TYPST, **kwargs)
    if self.inline:
      return f"${content}$"
    else:
      return f" $ {content} $ "

class Matrix(Leaf):
  """
  Mathematical matrix renderer for consistent cross-format display.

  CRITICAL: Use this for ALL matrix and vector notation instead of manual LaTeX.

  DON'T do this:
      # Manual LaTeX (error-prone, inconsistent)
      latex_str = f"\\\\begin{{bmatrix}} {a} & {b} \\\\\\\\ {c} & {d} \\\\end{{bmatrix}}"

  DO this instead:
      # Matrix (consistent, cross-format)
      matrix_data = [[a, b], [c, d]]
      Matrix(data=matrix_data, bracket_type="b")

  For vectors (single column matrices):
      vector_data = [[v1], [v2], [v3]]  # Note: list of single-element lists
      Matrix(data=vector_data, bracket_type="b")

  For LaTeX strings in equations:
      matrix_latex = Matrix.to_latex(matrix_data, "b")
      Equation(f"A = {matrix_latex}")

  Bracket types:
      - "b": square brackets [matrix] - most common for vectors/matrices
      - "p": parentheses (matrix) - sometimes used for matrices
      - "v": vertical bars |matrix| - for determinants
      - "B": curly braces {matrix}
      - "V": double vertical bars ||matrix|| - for norms
  """
  def __init__(self, data, *, bracket_type="p", inline=False, name=None):
    """
    Creates a matrix element that renders consistently across output formats.

    Args:
        data: Matrix data as List[List[numbers/strings]] or numpy ndarray (1D or 2D)
              For vectors: [[v1], [v2], [v3]] (column vector) or np.array([v1, v2, v3])
              For matrices: [[a, b], [c, d]] or np.array([[a, b], [c, d]])
        bracket_type: Bracket style - "b" for [], "p" for (), "v" for |, etc.
        inline: Whether to use inline (smaller) matrix formatting
    """
    super().__init__("[matrix]")

    # Convert numpy ndarray to list format if needed
    if isinstance(data, np.ndarray):
      if data.ndim == 1:
        # 1D array: convert to column vector [[v1], [v2], [v3]]
        self.data = [[val] for val in data]
      elif data.ndim == 2:
        # 2D array: convert to list of lists
        self.data = data.tolist()
      else:
        raise ValueError(f"Matrix only supports 1D or 2D arrays, got {data.ndim}D")
    else:
      self.data = data

    self.bracket_type = bracket_type
    self.inline = inline
    self.name = name

  @staticmethod
  def to_latex(data, bracket_type="p"):
    """
    Convert matrix data to LaTeX string for use in equations.

    Use this when you need a LaTeX string to embed in Equation:
        matrix_latex = Matrix.to_latex([[1, 2], [3, 4]], "b")
        Equation(f"A = {matrix_latex}")

    Args:
        data: Matrix data as List[List[numbers/strings]]
        bracket_type: Bracket style ("b", "p", "v", etc.)

    Returns:
        str: LaTeX matrix string (e.g., "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}")
    """
    rows = []
    for row in data:
      rows.append(" & ".join(str(cell) for cell in row))
    matrix_content = r" \\ ".join(rows)
    return f"\\begin{{{bracket_type}matrix}} {matrix_content} \\end{{{bracket_type}matrix}}"

  def math_content_latex(self):
    """Return raw LaTeX math content without delimiters (for use in MathExpression)."""
    return Matrix.to_latex(self.data, self.bracket_type)

  def math_content_typst(self):
    """Return raw Typst math content without delimiters (for use in MathExpression)."""
    # Build matrix content
    rows = []
    for row in self.data:
      rows.append(", ".join(str(cell) for cell in row))

    # Check if it's a vector (single column)
    is_vector = all(len(row) == 1 for row in self.data)

    if is_vector:
      # Use vec() for vectors
      matrix_content = ", ".join(str(row[0]) for row in self.data)
      result = f"vec({matrix_content})"
    else:
      # Use mat() for matrices with semicolons separating rows
      matrix_content = "; ".join(rows)
      result = f"mat({matrix_content})"

    # Add bracket delimiters if needed
    if self.bracket_type == "b":  # square brackets
      matrix_content_inner = ", ".join(str(row[0]) for row in self.data) if is_vector else "; ".join(rows)
      result = f"mat(delim: \"[\", {matrix_content_inner})" if not is_vector else f"vec(delim: \"[\", {matrix_content_inner})"
    elif self.bracket_type == "v":  # vertical bars (determinant)
      result = f"mat(delim: \"|\", {'; '.join(rows)})"
    elif self.bracket_type == "V":  # double vertical bars (norm)
      result = f"mat(delim: \"||\", {'; '.join(rows)})"
    elif self.bracket_type == "B":  # curly braces
      result = f"mat(delim: \"{{\", {'; '.join(rows)})"

    return result

  def render_markdown(self, **kwargs):
    matrix_env = "smallmatrix" if self.inline else f"{self.bracket_type}matrix"
    rows = []
    for row in self.data:
      rows.append(" & ".join(str(cell) for cell in row))
    matrix_content = r" \\ ".join(rows)

    if self.inline and self.bracket_type == "p":
      return f"$\\big(\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}\\big)$"
    else:
      return f"$\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}$"

  def render_html(self, **kwargs):
    matrix_env = "smallmatrix" if self.inline else f"{self.bracket_type}matrix"
    rows = []
    if isinstance(self.data, np.ndarray):
      data = self.data.tolist()
    else:
      data = self.data
    for row in data:
      rows.append(" & ".join(str(cell) for cell in row))
    matrix_content = r" \\ ".join(rows)

    if self.inline:
      return f"<span class='math'>$\\big(\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}\\big)$</span>"
    else:
      name_str = f"\\text{{{self.name}}} = " if self.name else ""
      return f"<div class='math'>$${name_str}\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}$$</div>"

  def render_latex(self, **kwargs):
    matrix_env = "smallmatrix" if self.inline else f"{self.bracket_type}matrix"
    rows = []
    for row in self.data:
      rows.append(" & ".join(str(cell) for cell in row))
    matrix_content = r" \\ ".join(rows)

    if self.inline and self.bracket_type == "p":
      return f"$\\big(\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}\\big)$"
    else:
      return f"\\[\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}\\]"

  def render_typst(self, **kwargs):
    """Render matrix in Typst format using mat() and vec() functions."""
    # Build matrix content with semicolons separating rows
    rows = []
    for row in self.data:
      rows.append(", ".join(str(cell) for cell in row))

    # Check if it's a vector (single column)
    is_vector = all(len(row) == 1 for row in self.data)

    if is_vector:
      # Use vec() for vectors
      matrix_content = ", ".join(str(row[0]) for row in self.data)
      result = f"vec({matrix_content})"
    else:
      # Use mat() for matrices with semicolons separating rows
      matrix_content = "; ".join(rows)
      result = f"mat({matrix_content})"

    # Add bracket delimiters if needed
    if self.bracket_type == "b":  # square brackets
      result = f"mat(delim: \"[\", {matrix_content})" if not is_vector else f"vec(delim: \"[\", {matrix_content})"
    elif self.bracket_type == "v":  # vertical bars (determinant)
      result = f"mat(delim: \"|\", {matrix_content})"
    elif self.bracket_type == "B":  # curly braces
      result = f"mat(delim: \"{{\", {matrix_content})"
    # "p" (parentheses) is the default, no need to specify

    # Wrap in math mode
    if self.inline:
      return f"${result}$"
    else:
      return f"$ {result} $"

class Picture(Leaf):
  """
  Image/diagram container with proper sizing and captioning.

  Handles image content with automatic upload management for Canvas
  and proper LaTeX figure environments for PDF output.

  When to use:
  - Diagrams, charts, or visual content
  - Memory layout diagrams
  - Process flowcharts
  - Any visual aid for questions

  Features:
  - Automatic Canvas image upload handling
  - Proper LaTeX figure environments
  - Responsive sizing with width control
  - Optional captions

  Example:
      # Image with caption
      with open('diagram.png', 'rb') as f:
          img_data = BytesIO(f.read())

      picture = Picture(
          img_data=img_data,
          caption="Memory layout diagram",
          width="80%"
      )
      body.add_element(picture)
  """
  def __init__(self, img_data, caption=None, width=None):
    super().__init__("[picture]")
    self.img_data = img_data
    self.caption = caption
    self.width = width
    self.path = None  # Will be set when image is saved

  def _ensure_image_saved(self):
    """Save image data to file if not already saved."""
    if self.path is None:

      # Create imgs directory if it doesn't exist (use absolute path)
      img_dir = os.path.abspath("imgs")
      if not os.path.exists(img_dir):
        os.makedirs(img_dir)

      # Generate unique filename
      filename = f"image-{uuid.uuid4()}.png"
      self.path = os.path.join(img_dir, filename)

      # Save BytesIO data to file
      with open(self.path, 'wb') as f:
        self.img_data.seek(0)  # Reset buffer position
        f.write(self.img_data.read())

  def render_markdown(self, **kwargs):
    self._ensure_image_saved()
    if self.caption:
      return f"![{self.caption}]({self.path})"
    return f"![]({self.path})"

  def render_html(
      self,
      upload_func: Callable[[BytesIO], str] = lambda _: "",
      **kwargs
  ) -> str:
    attrs = []
    if self.width:
      attrs.append(f'width="{self.width}"')
    
    img = f'<img src="{upload_func(self.img_data)}" {" ".join(attrs)} alt="{self.caption or ""}">'
    
    if self.caption:
      return f'<figure>\n  {img}\n  <figcaption>{self.caption}</figcaption>\n</figure>'
    return img
  
  def render_latex(self, **kwargs):
    self._ensure_image_saved()

    options = []
    if self.width:
      options.append(f"width={self.width}")

    result = ["\\begin{figure}[h]"]
    result.append(f"\\centering")
    result.append(f"\\includegraphics[{','.join(options)}]{{{self.path}}}")

    if self.caption:
      result.append(f"\\caption{{{self.caption}}}")

    result.append("\\end{figure}")
    return "\n".join(result)

  def render_typst(self, **kwargs):
    self._ensure_image_saved()

    # Build the image function call
    img_params = []
    if self.width:
      img_params.append(f'width: {self.width}')

    params_str = ', '.join(img_params) if img_params else ''

    # Use Typst's figure and image functions
    result = []
    result.append("#figure(")
    result.append(f'  image("{self.path}"{", " + params_str if params_str else ""}),')

    if self.caption:
      result.append(f'  caption: [{self.caption}]')

    result.append(")")

    return "\n".join(result)

class LineBreak(Text):
  def __init__(self, *args, **kwargs):
    super().__init__("\n\n")

## Containers

class Paragraph(Container):
  """
  Text block container with proper spacing and paragraph formatting.

  IMPORTANT: Use this for grouping text content, especially in question bodies.
  Automatically handles spacing between paragraphs and combines multiple
  lines/elements into a cohesive text block.

  When to use:
  - Question instructions or problem statements
  - Multi-line text content
  - Grouping related text elements
  - Any text that should be visually separated as a paragraph

  When NOT to use:
  - Single words or short phrases (use Text)
  - Mathematical content (use Equation)
  - Structured data (use Table)

  Example:
      # Multi-line question text
      body.add_element(Paragraph([
          "Consider the following system:",
          "- Process A requires 4MB memory",
          "- Process B requires 2MB memory",
          "How much total memory is needed?"
      ]))

      # Mixed content paragraph
      para = Paragraph([
          "The equation ",
          Equation("x^2 + 1 = 0", inline=True),
          " has no real solutions."
      ])
  """
  
  def __init__(self, lines_or_elements: List[str | Element] = None):
    super().__init__(add_spacing_before=True)
    for line in lines_or_elements:
      if isinstance(line, str):
        self.elements.append(Text(line))
      else:
        self.elements.append(line)
  
  def render(self, output_format, **kwargs):
    # Add in new lines to break these up visually
    return "\n\n" + super().render(output_format, **kwargs) + "\n\n"
  
  def render_html(self, **kwargs):
    return super().render_html(**kwargs) + "<br>"
  
  def add_line(self, line: str):
    self.elements.append(Text(line))

class Table(Container):
  """
  Structured data table with cross-format rendering and proper formatting.

  Creates properly formatted tables that work in PDF, Canvas, and Markdown.
  Automatically handles headers, alignment, and responsive formatting.
  All data is converted to content AST elements for consistent rendering.

  When to use:
  - Structured data presentation (comparison tables, data sets)
  - Answer choices in tabular format
  - Organized information display
  - Memory layout diagrams, process tables, etc.

  Features:
  - Automatic alignment control (left, right, center)
  - Optional headers with proper formatting
  - Canvas-compatible HTML output
  - LaTeX booktabs for professional PDF tables

  Example:
      # Basic data table
      data = [
          ["Process A", "4MB", "Running"],
          ["Process B", "2MB", "Waiting"]
      ]
      headers = ["Process", "Memory", "Status"]
      table = Table(data=data, headers=headers, alignments=["left", "right", "center"])
      body.add_element(table)

      # Mixed content table
      data = [
          [Text("x"), Equation("x^2", inline=True)],
          [Text("y"), Equation("y^2", inline=True)]
      ]
  """
  
  def __init__(self, data, headers=None, alignments=None, padding=False, transpose=False, hide_rules=False):
    # todo: fix alignments
    # todo: implement transpose
    super().__init__()
    
    # Normalize data to content AST elements
    self.data = []
    for row in data:
      normalized_row = []
      for cell in row:
        if isinstance(cell, Element):
          normalized_row.append(cell)
        else:
          normalized_row.append(Text(str(cell)))
      self.data.append(normalized_row)
    
    # Normalize headers to content AST elements
    if headers:
      self.headers = []
      for header in headers:
        if isinstance(header, Element):
          self.headers.append(header)
        else:
          self.headers.append(Text(str(header)))
    else:
      self.headers = None
    
    self.alignments = alignments
    self.padding = padding,
    self.hide_rules = hide_rules
  
  def render_markdown(self, **kwargs):
    # Basic markdown table implementation
    result = []
    
    if self.headers:
      result.append("| " + " | ".join(str(h) for h in self.headers) + " |")
      
      if self.alignments:
        align_row = []
        for align in self.alignments:
          if align == "left":
            align_row.append(":---")
          elif align == "right":
            align_row.append("---:")
          else:  # center
            align_row.append(":---:")
        result.append("| " + " | ".join(align_row) + " |")
      else:
        result.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
    
    for row in self.data:
      result.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join(result)
  
  def render_html(self, **kwargs):
    # HTML table implementation
    result = ["<table border=\"1\" style=\"border-collapse: collapse; width: 100%;\">"]
    
    result.append("  <tbody>")
    
    # Render headers as bold first row instead of <th> tags for Canvas compatibility
    if self.headers:
      result.append("    <tr>")
      for i, header in enumerate(self.headers):
        align_attr = ""
        if self.alignments and i < len(self.alignments):
          align_attr = f' align="{self.alignments[i]}"'
        # Render header as bold content in regular <td> tag
        rendered_header = header.render(output_format="html", **kwargs)
        result.append(
          f"      <td style=\"padding: {'5px' if self.padding else '0x'}; font-weight: bold; {align_attr};\"><b>{rendered_header}</b></td>"
        )
      result.append("    </tr>")
    
    # Render data rows
    for row in self.data:
      result.append("    <tr>")
      for i, cell in enumerate(row):
        if isinstance(cell, Element):
          cell = cell.render(output_format="html", **kwargs)
        align_attr = ""
        if self.alignments and i < len(self.alignments):
          align_attr = f' align="{self.alignments[i]}"'
        result.append(f"      <td  style=\"padding: {'5px' if self.padding else '0x'} ; {align_attr};\">{cell}</td>")
      result.append("    </tr>")
    result.append("  </tbody>")
    result.append("</table>")
    
    return "\n".join(result)
  
  def render_latex(self, **kwargs):
    # LaTeX table implementation
    if self.alignments:
      col_spec = "".join(
        "l" if a == "left" else "r" if a == "right" else "c"
        for a in self.alignments
      )
    else:
      col_spec = '|'.join(["l"] * (len(self.headers) if self.headers else len(self.data[0])))
    
    result = [f"\\begin{{tabular}}{{{col_spec}}}"]
    if not self.hide_rules: result.append("\\toprule")
    
    if self.headers:
      # Now all headers are content AST elements, so render them consistently
      rendered_headers = [header.render(output_format="latex", **kwargs) for header in self.headers]
      result.append(" & ".join(rendered_headers) + " \\\\")
      if not self.hide_rules: result.append("\\midrule")
    
    for row in self.data:
      # All data cells are now content AST elements, so render them consistently
      rendered_row = [cell.render(output_format="latex", **kwargs) for cell in row]
      result.append(" & ".join(rendered_row) + " \\\\")
    
    if len(self.data) > 1 and not self.hide_rules:
      result.append("\\bottomrule")
    result.append("\\end{tabular}")
    
    return "\n\n" + "\n".join(result)
  
  def render_typst(self, **kwargs):
    """
    Render table in Typst format using native table() function.

    Typst syntax:
    #table(
      columns: N,
      align: (left, center, right),
      [Header1], [Header2],
      [Cell1], [Cell2]
    )
    """
    # Determine number of columns
    num_cols = len(self.headers) if self.headers else len(self.data[0])
    
    # Build alignment specification
    if self.alignments:
      # Map alignment strings to Typst alignment
      align_map = {"left": "left", "right": "right", "center": "center"}
      aligns = [align_map.get(a, "left") for a in self.alignments]
      align_spec = f"align: ({', '.join(aligns)})"
    else:
      align_spec = "align: left"
    
    # Start table
    result = [f"table("]
    result.append(f"  columns: {num_cols},")
    result.append(f"  {align_spec},")
    
    # Add stroke if not hiding rules
    if not self.hide_rules:
      result.append(f"  stroke: 0.5pt,")
    else:
      result.append(f"  stroke: none,")
    
    # Collect all rows (headers + data) and calculate column widths for alignment
    all_rows = []
    
    # Render headers
    if self.headers:
      header_cells = []
      for header in self.headers:
        rendered = header.render(output_format="typst", **kwargs).strip()
        header_cells.append(f"[*{rendered}*]")
      all_rows.append(header_cells)
    
    # Render data rows
    for row in self.data:
      row_cells = []
      for cell in row:
        rendered = cell.render(output_format="typst", **kwargs).strip()
        row_cells.append(f"[{rendered}]")
      all_rows.append(row_cells)
    
    # Calculate max width for each column
    col_widths = [0] * num_cols
    for row in all_rows:
      for i, cell in enumerate(row):
        col_widths[i] = max(col_widths[i], len(cell))
    
    # Format rows with padding
    for row in all_rows:
      padded_cells = []
      for i, cell in enumerate(row):
        padded_cells.append(cell.ljust(col_widths[i]))
      result.append(f"  {', '.join(padded_cells)},")
    
    result.append(")")
    
    return "\n#box(" + "\n".join(result) + "\n)"

class TableGroup(Container):
  """
  Container for displaying multiple tables side-by-side in LaTeX, stacked in HTML.

  Use this when you need to show multiple related tables together, such as
  multiple page tables in hierarchical paging questions. In LaTeX, tables
  are displayed side-by-side using minipages. In HTML/Canvas, they're stacked
  vertically for better mobile compatibility.

  When to use:
  - Multiple related tables that should be visually grouped
  - Page tables in hierarchical paging
  - Comparison of multiple data structures

  Features:
  - Automatic side-by-side layout in PDF (using minipages)
  - Vertical stacking in HTML for better readability
  - Automatic width calculation based on number of tables
  - Optional labels for each table

  Example:
      # Create table group with labels
      table_group = TableGroup()

      table_group.add_table(
          label="Page Table #0",
          table=Table(headers=["PTI", "PTE"], data=pt0_data)
      )

      table_group.add_table(
          label="Page Table #1",
          table=Table(headers=["PTI", "PTE"], data=pt1_data)
      )

      body.add_element(table_group)
  """
  def __init__(self):
    super().__init__()
    self.tables = []  # List of (label, table) tuples

  def add_table(self, table: Table, label: str = None):
    """
    Add a table to the group with an optional label.

    Args:
        table: Table to add
        label: Optional label to display above the table
    """
    self.tables.append((label, table))

  def render_html(self, **kwargs):
    # Stack tables vertically in HTML
    result = []
    for label, table in self.tables:
      if label:
        result.append(f"<p><b>{label}</b></p>")
      result.append(table.render("html", **kwargs))
    return "\n".join(result)

  def render_latex(self, **kwargs):
    if not self.tables:
      return ""

    # Calculate width based on number of tables
    num_tables = len(self.tables)
    if num_tables == 1:
      width = 0.9
    elif num_tables == 2:
      width = 0.45
    else:  # 3 or more
      width = 0.30

    result = ["\n\n"]  # Add spacing before table group

    for i, (label, table) in enumerate(self.tables):
      result.append(f"\\begin{{minipage}}{{{width}\\textwidth}}")

      if label:
        # Escape # characters in labels for LaTeX
        escaped_label = label.replace("#", r"\#")
        result.append(f"\\textbf{{{escaped_label}}}")
        result.append("\\vspace{0.1cm}")

      # Render the table
      table_latex = table.render("latex", **kwargs)
      result.append(table_latex)

      result.append("\\end{minipage}")

      # Add horizontal spacing between tables (but not after the last one)
      if i < num_tables - 1:
        result.append("\\hfill")

    return "\n".join(result)

  def render_typst(self, **kwargs):
    """
    Render table group in Typst format using grid layout for side-by-side tables.

    Uses Typst's grid() function to arrange tables horizontally with automatic
    column sizing and spacing.
    """
    if not self.tables:
      return ""

    num_tables = len(self.tables)

    # Start grid with equal-width columns and some spacing
    result = ["\n#grid("]
    result.append(f"  columns: {num_tables},")
    result.append(f"  column-gutter: 1em,")
    result.append(f"  row-gutter: 0.5em,")

    # Add each table as a grid cell
    for label, table in self.tables:
      result.append("  [")  # Start grid cell

      if label:
        # Escape # characters in labels (already done by Text.render_typst)
        result.append(f"    *{label}*")
        result.append("    #v(0.1cm)")
        result.append("")  # Empty line for spacing

      # Render the table (indent for readability)
      table_typst = table.render("typst", **kwargs)
      # Indent each line of the table
      indented_table = "\n".join(f"    {line}" if line else "" for line in table_typst.split("\n"))
      result.append(indented_table)

      result.append("  ],")  # End grid cell

    result.append(")")
    result.append("")  # Empty line after grid

    return "\n".join(result)

class AnswerBlock(Table):
  """
  Specialized table for organizing multiple answer fields with proper spacing.

  Creates a clean layout for multiple answer inputs with extra vertical
  spacing in PDF output. Inherits from Table but optimized for answers.

  When to use:
  - Questions with multiple answer fields
  - Organized answer input sections
  - Better visual grouping of related answers

  Example:
      # Multiple related answers - Answer extends Leaf, use factory methods
      memory_ans = Answer.integer("memory", self.memory_value, label="Memory used", unit="MB")
      time_ans = Answer.auto_float("time", self.time_value, label="Execution time", unit="ms")
      answer_block = AnswerBlock([memory_ans, time_ans])
      body.add_element(answer_block)

      # Single answer with better spacing
      result_ans = Answer.integer("result", self.result_value, label="Final result")
      single_answer = AnswerBlock(result_ans)
  """
  def __init__(self, answers: Answer|List[Answer]):
    if not isinstance(answers, list):
      answers = [answers]

    super().__init__(
      data=[
        [answer]
        for answer in answers
      ]
    )
    self.hide_rules = True

  def add_element(self, element):
    self.data.append(element)

  def render_latex(self, **kwargs):
    rendered_content = super().render_latex(**kwargs)
    content = (
      r"{"
      r"\setlength{\extrarowheight}{20pt}"
      + rendered_content +
      r"}"
    )
    return content

## Specialized Elements
class RepeatedProblemPart(Container):
  """
  Multi-part problem renderer for questions with labeled subparts (a), (b), (c), etc.

  Creates the specialized alignat* LaTeX format for multipart math problems
  where each subpart is labeled and aligned properly. Used primarily for
  vector math questions that need multiple similar calculations.

  When to use:
  - Questions with multiple subparts that need (a), (b), (c) labeling
  - Vector math problems with repeated calculations
  - Any math problem where subparts should be aligned

  Features:
  - Automatic subpart labeling with (a), (b), (c), etc.
  - Proper LaTeX alignat* formatting for PDF
  - HTML fallback with organized layout
  - Flexible content support (equations, matrices, etc.)

  Example:
      # Create subparts for vector dot products
      subparts = [
          (Matrix([[1], [2]]), "\\cdot", Matrix([[3], [4]])),
          (Matrix([[5], [6]]), "\\cdot", Matrix([[7], [8]]))
      ]
      repeated_part = RepeatedProblemPart(subparts)
      body.add_element(repeated_part)
  """
  def __init__(self, subpart_contents):
    """
    Create a repeated problem part with multiple subquestions.

    Args:
        subpart_contents: List of content for each subpart.
                         Each item can be:
                         - A string (rendered as equation)
                         - A Element
                         - A tuple/list of elements to be joined
    """
    super().__init__()
    self.subpart_contents = subpart_contents

  def render_markdown(self, **kwargs):
    result = []
    for i, content in enumerate(self.subpart_contents):
      letter = chr(ord('a') + i)  # Convert to (a), (b), (c), etc.
      if isinstance(content, str):
        result.append(f"({letter}) {content}")
      elif isinstance(content, (list, tuple)):
        content_str = " ".join(str(item) for item in content)
        result.append(f"({letter}) {content_str}")
      else:
        result.append(f"({letter}) {str(content)}")
    return "\n\n".join(result)

  def render_html(self, **kwargs):
    result = []
    for i, content in enumerate(self.subpart_contents):
      letter = chr(ord('a') + i)
      if isinstance(content, str):
        result.append(f"<p>({letter}) {content}</p>")
      elif isinstance(content, (list, tuple)):
        rendered_items = []
        for item in content:
          if hasattr(item, 'render'):
            rendered_items.append(item.render('html', **kwargs))
          else:
            rendered_items.append(str(item))
        content_str = " ".join(rendered_items)
        result.append(f"<p>({letter}) {content_str}</p>")
      else:
        if hasattr(content, 'render'):
          content_str = content.render('html', **kwargs)
        else:
          content_str = str(content)
        result.append(f"<p>({letter}) {content_str}</p>")
    return "\n".join(result)

  def render_latex(self, **kwargs):
    if not self.subpart_contents:
      return ""

    # Start alignat environment - use 2 columns for alignment
    result = [r"\begin{alignat*}{2}"]

    for i, content in enumerate(self.subpart_contents):
      letter = chr(ord('a') + i)
      spacing = r"\\[6pt]" if i < len(self.subpart_contents) - 1 else r" \\"

      if isinstance(content, str):
        # Treat as raw LaTeX equation content
        result.append(f"({letter})\\;& {content} &=&\\; {spacing}")
      elif isinstance(content, (list, tuple)):
        # Join multiple elements (e.g., matrix, operator, matrix)
        rendered_items = []
        for item in content:
          if hasattr(item, 'render'):
            rendered_items.append(item.render('latex', **kwargs))
          elif isinstance(item, str):
            rendered_items.append(item)
          else:
            rendered_items.append(str(item))
        content_str = " ".join(rendered_items)
        result.append(f"({letter})\\;& {content_str} &=&\\; {spacing}")
      else:
        # Single element (content AST element or string)
        if hasattr(content, 'render'):
          content_str = content.render('latex', **kwargs)
        else:
          content_str = str(content)
        result.append(f"({letter})\\;& {content_str} &=&\\; {spacing}")

    result.append(r"\end{alignat*}")
    return "\n".join(result)

class OnlyLatex(Container):
  """
  Container element that only renders content in LaTeX/PDF output format.

  Use this when you need LaTeX-specific content that should not appear
  in HTML/Canvas or Markdown outputs. Content is completely hidden
  from non-LaTeX formats.

  When to use:
  - LaTeX-specific formatting that has no HTML equivalent
  - PDF-only instructions or content
  - Complex LaTeX commands that break HTML rendering

  Example:
      # LaTeX-only spacing or formatting
      latex_only = OnlyLatex()
      latex_only.add_element(Text("\\newpage"))

      # Add to main content - only appears in PDF
      body.add_element(latex_only)
  """

  def render(self, output_format: OutputFormat, **kwargs):
    if output_format not in ("latex", "typst"):
      return ""
    return super().render(output_format=output_format, **kwargs)

class OnlyHtml(Container):
  """
  Container element that only renders content in HTML/Canvas output format.

  Use this when you need HTML-specific content that should not appear
  in LaTeX/PDF or Markdown outputs. Content is completely hidden
  from non-HTML formats.

  When to use:
  - Canvas-specific instructions or formatting
  - HTML-only interactive elements
  - Content that doesn't translate well to PDF

  Example:
      # HTML-only instructions
      html_only = OnlyHtml()
      html_only.add_element(Text("Click submit when done"))

      # Add to main content - only appears in Canvas
      body.add_element(html_only)
  """
  
  def render(self, output_format, **kwargs):
    if output_format != "html":
      return ""
    return super().render(output_format, **kwargs)

class Answer(Leaf):
  """
  Unified answer class combining data storage, Canvas export, and rendering.

  Extends Leaf to integrate seamlessly with the content AST tree while
  maintaining all Canvas export functionality.

  CRITICAL: Use this for ALL answer inputs in questions.
  Creates appropriate input fields that work across both PDF and Canvas formats.
  In PDF, renders as blank lines for students to fill in.
  In HTML/Canvas, can display the answer for checking.

  Example:
      # Basic answer field
      ans = Answer.integer("result", 42, label="Result", unit="MB")
      body.add_element(ans)
      answers.append(ans)  # Track for Canvas export
  """

  DEFAULT_ROUNDING_DIGITS = 4

  class CanvasAnswerKind(enum.Enum):
    BLANK = "fill_in_multiple_blanks_question"
    MULTIPLE_ANSWER = "multiple_answers_question"
    ESSAY = "essay_question"
    MULTIPLE_DROPDOWN = "multiple_dropdowns_question"
    NUMERICAL_QUESTION = "numerical_question"

  class VariableKind(enum.Enum):
    STR = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()
    BINARY = enum.auto()
    HEX = enum.auto()
    BINARY_OR_HEX = enum.auto()
    AUTOFLOAT = enum.auto()
    LIST = enum.auto()
    VECTOR = enum.auto()
    MATRIX = enum.auto()

  def __init__(
      self,
      value=None,
      kind: Answer.CanvasAnswerKind = None,
      variable_kind: Answer.VariableKind = None,
      *,
      # Data fields (from misc.Answer)
      display=None,
      length=None,
      correct=True,
      baffles=None,
      pdf_only=False,
      # Rendering fields (from Answer)
      label: str = "",
      unit: str = "",
      blank_length=5,
  ):
    
    # Initialize Leaf with label as content
    super().__init__(content=label if label else "")

    # Data fields
    self.key = str(uuid.uuid4())
    self.value = value
    self.kind = kind if kind is not None else Answer.CanvasAnswerKind.BLANK
    self.variable_kind = variable_kind if variable_kind is not None else Answer.VariableKind.STR

    # For list values in display, show the first option (or join them with /)
    if display is not None:
      self.display = display
    elif isinstance(value, list) and isinstance(self.variable_kind, AnswerTypes.String):
      self.display = value[0] if len(value) == 1 else " / ".join(value)
    else:
      self.display = value

    self.length = length  # Used for bits and hex to be printed appropriately
    self.correct = correct
    self.baffles = baffles
    self.pdf_only = pdf_only

    # Rendering fields
    self.label = label
    self.unit = unit
    self.blank_length = blank_length

  # Canvas export methods (from misc.Answer)
  def get_for_canvas(self, single_answer=False) -> List[dict]:
    """Generate Canvas answer dictionaries based on variable_kind."""
    
    # If this answer is PDF-only, don't send it to Canvas
    if self.pdf_only:
      return []

    canvas_answers = []
    if isinstance(self.value, list):
      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": str(alt),
          "answer_weight": 100 if self.correct else 0,
        }
        for alt in self.value
      ]
    else:
      canvas_answers = [{
        "blank_id": self.key,
        "answer_text": self.value,
        "answer_weight": 100 if self.correct else 0,
      }]

    # Add baffles (incorrect answer choices)
    if self.baffles is not None:
      for baffle in self.baffles:
        canvas_answers.append({
          "blank_id": self.key,
          "answer_text": baffle,
          "answer_weight": 0,
        })

    return canvas_answers

  def get_display_string(self) -> str:
    """Get the formatted display string for this answer (for grading/answer keys)."""
    return str(self.display if hasattr(self, 'display') else self.value)

  # Rendering methods (override Leaf's defaults)
  def render_markdown(self, **kwargs):
    return f"{self.label + (':' if len(self.label) > 0 else '')} [{self.key}] {self.unit}".strip()

  def render_html(self, show_answers=False, can_be_numerical=False, **kwargs):
    if can_be_numerical:
      return f"Calculate {self.label}"
    if show_answers:
      answer_display = self.get_display_string()
      label_part = f"{self.label}:" if self.label else ""
      unit_part = f" {self.unit}" if self.unit else ""
      return f"{label_part} <strong>{answer_display}</strong>{unit_part}".strip()
    else:
      return self.render_markdown(**kwargs)

  def render_latex(self, **kwargs):
    return fr"{self.label + (':' if len(self.label) > 0 else '')} \answerblank{{{self.blank_length}}} {self.unit}".strip()

  def render_typst(self, **kwargs):
    """Render answer blank as an underlined space in Typst."""
    blank_width = self.blank_length * 0.75  # Convert character length to cm
    blank = f"#fillline(width: {blank_width}cm)"

    label_part = f"{self.label}:" if self.label else ""
    unit_part = f" {self.unit}" if self.unit else ""

    return f"{label_part} {blank}{unit_part}".strip()
  
  # Factory methods for common answer types
  @classmethod
  def dropdown(cls, value: str, baffles: list = None, **kwargs) -> 'Answer':
    """Create a dropdown answer with wrong answer choices (baffles)"""
    return cls(
      value=value,
      kind=cls.CanvasAnswerKind.MULTIPLE_DROPDOWN,
      baffles=baffles,
      **kwargs
    )

  @classmethod
  def multiple_choice(cls, key: str, value: str, baffles: list = None, **kwargs) -> 'Answer':
    """Create a multiple choice answer with wrong answer choices (baffles)"""
    return cls(
      value=value,
      kind=cls.CanvasAnswerKind.MULTIPLE_ANSWER,
      baffles=baffles,
      **kwargs
    )

  # Static helper methods
  @staticmethod
  def _to_fraction(x):
    """Convert int/float/decimal.Decimal/fractions.Fraction/str to fractions.Fraction exactly."""
    if isinstance(x, fractions.Fraction):
      return x
    if isinstance(x, int):
      return fractions.Fraction(x, 1)
    if isinstance(x, decimal.Decimal):
      # exact conversion of decimal.Decimal to fractions.Fraction
      sign, digits, exp = x.as_tuple()
      n = 0
      for d in digits:
        n = n * 10 + d
      n = -n if sign else n
      if exp >= 0:
        return fractions.Fraction(n * (10 ** exp), 1)
      else:
        return fractions.Fraction(n, 10 ** (-exp))
    if isinstance(x, str):
      s = x.strip()
      if '/' in s:
        a, b = s.split('/', 1)
        return fractions.Fraction(int(a.strip()), int(b.strip()))
      return fractions.Fraction(decimal.Decimal(s))
    # float or other numerics
    return fractions.Fraction(decimal.Decimal(str(x)))

  @staticmethod
  def accepted_strings(value, max_denominator=720):
    """
    Return a sorted list of acceptable answer strings for Canvas.

    Generates:
    - Integer form (if value is a whole number)
    - Fixed decimal with DEFAULT_ROUNDING_DIGITS (e.g., "1.0000")
    - Trimmed decimal without trailing zeros (e.g., "1.25")
    - Simple fraction if exactly representable (e.g., "5/4")

    Examples:
      1 → ["1", "1.0000"]
      1.25 → ["1.25", "1.2500", "5/4"]
      0.123444... → ["0.1234"]
    """
    rounding_digits = Answer.DEFAULT_ROUNDING_DIGITS
    decimal.getcontext().prec = max(34, rounding_digits + 10)

    outs = set()

    # Round to our standard precision first
    q = decimal.Decimal(1).scaleb(-rounding_digits)
    rounded_decimal = decimal.Decimal(str(value)).quantize(q, rounding=decimal.ROUND_HALF_UP)

    # Normalize negative zero to positive zero
    if rounded_decimal == 0:
      rounded_decimal = abs(rounded_decimal)

    # Fixed decimal form (e.g., "1.2500")
    fixed_str = format(rounded_decimal, 'f')
    outs.add(fixed_str)

    # Trimmed decimal form (e.g., "1.25")
    trimmed_str = fixed_str.rstrip('0').rstrip('.')
    if trimmed_str.startswith('.'):
      trimmed_str = '0' + trimmed_str
    outs.add(trimmed_str)

    # Integer form (if it's a whole number after rounding)
    if rounded_decimal == rounded_decimal.to_integral_value():
      outs.add(str(int(rounded_decimal)))

    # Fraction form (only if exactly representable with reasonable denominator)
    f = Answer._to_fraction(rounded_decimal)
    fr = f.limit_denominator(max_denominator)
    if fr == f and fr.denominator > 1:
      outs.add(f"{fr.numerator}/{fr.denominator}")

    return sorted(outs, key=lambda s: (len(s), s))
  
  @staticmethod
  def fix_negative_zero(x):
    """Fix -0.0 display issue."""
    return 0.0 if x == 0 else x

  @staticmethod
  def get_spacing_variations_of_list(l, remove_space=False):
    return [', '.join(l)] + ([', '.join(l)] if remove_space else [])
  

class AnswerTypes:
  # Multibase answers that can accept either hex, binary or decimal
  class MultiBase(Answer):
    """
    These are answers that can accept answers in any sort of format, and default to displaying in hex when written out.
    This will be the parent class for Binary, Hex, and Integer answers most likely.
    """
    def get_for_canvas(self, single_answer=False) -> List[dict]:

      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0b{self.value:0{self.length if self.length is not None else 0}b}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value:0{math.ceil(self.length / 8) if self.length is not None else 0}X}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"0x{self.value:0{math.ceil(self.length / 8) if self.length is not None else 0}X}",
          "answer_weight": 100 if self.correct else 0,
        },
        {
          "blank_id": self.key,
          "answer_text": f"{self.value}",
          "answer_weight": 100 if self.correct else 0,
        },
      ]
    
      return canvas_answers

    def get_display_string(self) -> str:
      # This is going to be the default for multi-base answers, but may change later.
      hex_digits = (self.length // 4) + 1 if self.length is not None else 0
      return f"0x{self.value:0{hex_digits}X}"

  class Hex(MultiBase):
    pass

  class Binary(MultiBase):
    def get_display_string(self) -> str:
      return f"0b{self.value:0{self.length if self.length is not None else 0}b}"

  class Decimal(MultiBase):
    def get_display_string(self) -> str:
      return f"{self.value:0{self.length if self.length is not None else 0}}"

  # Concrete type answers
  class Float(Answer):
    def get_for_canvas(self, single_answer=False) -> List[dict]:
      if single_answer:
        canvas_answers = [
          {
            "numerical_answer_type": "exact_answer",
            "answer_text": round(self.value, Answer.DEFAULT_ROUNDING_DIGITS),
            "answer_exact": round(self.value, Answer.DEFAULT_ROUNDING_DIGITS),
            "answer_error_margin": 0.1,
            "answer_weight": 100 if self.correct else 0,
          }
        ]
      else:
        # Use the accepted_strings helper
        answer_strings = Answer.accepted_strings(
          self.value,
          max_denominator=60
        )
      
        canvas_answers = [
          {
            "blank_id": self.key,
            "answer_text": answer_string,
            "answer_weight": 100 if self.correct else 0,
          }
          for answer_string in answer_strings
        ]
      return canvas_answers

    def get_display_string(self) -> str:
      rounded = round(self.value, Answer.DEFAULT_ROUNDING_DIGITS)
      return f"{self.fix_negative_zero(rounded)}"

  class Int(Answer):

  # Canvas export methods (from misc.Answer)
    def get_for_canvas(self, single_answer=False) -> List[dict]:
      canvas_answers = [
        {
          "blank_id": self.key,
          "answer_text": str(int(self.value)),
          "answer_weight": 100 if self.correct else 0,
        }
      ]
      return canvas_answers

  # Open Ended
  class OpenEnded(Answer):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.kind=Answer.CanvasAnswerKind.ESSAY

  class String(Answer):
    pass

  class List(Answer):
    def __init__(self, order_matters=True, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.order_matters = order_matters
  
    def get_for_canvas(self, single_answer=False):
      if self.order_matters:
        canvas_answers = [
          {
            "blank_id": self.key,
            "answer_text": spacing_variation,
            "answer_weight": 100 if self.correct else 0,
          }
          for spacing_variation in self.get_spacing_variations_of_list(map(str, self.value))
        ]
      else:
        canvas_answers = []
        canvas_answers = [
          {
            "blank_id": self.key,
            "answer_text": spacing_variation,
            "answer_weight": 100 if self.correct else 0,
          }
          for possible_state in itertools.permutations(self.value)
          for spacing_variation in self.get_spacing_variations_of_list(possible_state)
        ]
      
      return canvas_answers

    def get_display_string(self) -> str:
      """Get the formatted display string for this answer (for grading/answer keys)."""
      return ", ".join(str(v) for v in self.value)

  # Math types
  class Vector(Answer):
    """
    These are self-contained vectors that will go in a single answer block
    """
  
  # Canvas export methods (from misc.Answer)
    def get_for_canvas(self, single_answer=False) -> List[dict]:
      # Get all answer variations
      answer_variations = [
        Answer.accepted_strings(dimension_value)
        for dimension_value in self.value
      ]
    
      canvas_answers = []
      for combination in itertools.product(*answer_variations):
        for spacing_variation in self.get_spacing_variations_of_list(list(combination)):
          canvas_answers.extend([
            {
              "blank_id": self.key,
              "answer_weight": 100 if self.correct else 0,
              "answer_text": f"{spacing_variation}",
            }, # without parenthesis
            {
              "blank_id": self.key,
              "answer_weight": 100 if self.correct else 0,
              "answer_text": f"({spacing_variation})",
            } # with parenthesis
          ])
      return canvas_answers
    
    def get_display_string(self) -> str:
      return ", ".join(
        str(self.fix_negative_zero(round(v, Answer.DEFAULT_ROUNDING_DIGITS))).rstrip('0').rstrip('.') for v in self.value
      )
    
  class CompoundAnswers(Answer):
    pass
    """
    Going forward, this might make a lot of sense to have a SubAnswer class that we can iterate over.
    We would convert into this shared format and just iterate over it whenever we need to.
    """

  class Matrix(CompoundAnswers):
    """
    Matrix answers generate multiple blank_ids (e.g., M_0_0, M_0_1, M_1_0, M_1_1).
    """
  
    def __init__(self, value, *args, **kwargs):
      super().__init__(value=value, *args, **kwargs)
    
      self.data = [
        [
          AnswerTypes.Float(
            value=self.value[i, j],
            blank_length=5
          )
          for j in range(self.value.shape[1])
        ]
        for i in range(self.value.shape[0])
      ]
  
    def get_for_canvas(self, single_answer=False) -> List[dict]:
      """Generate Canvas answers for each matrix element."""
      canvas_answers = []
    
      for sub_answer in itertools.chain.from_iterable(self.data):
        canvas_answers.extend(sub_answer.get_for_canvas())
    
      return canvas_answers
  
    def render(self, *args, **kwargs) -> str:
      table = Table(self.data)
  
      if self.label:
        return Container(
          [
            Text(f"{self.label} = "),
            table
          ]
        ).render(*args, **kwargs)
      return table.render(*args, **kwargs)
  
