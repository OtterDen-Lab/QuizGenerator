from __future__ import annotations

import textwrap
from io import BytesIO
from typing import List, Callable

import pypandoc
import markdown

from QuizGenerator.misc import log, Answer

from QuizGenerator.qrcode_generator import QuestionQRCode

class ContentAST:
  """
  Content Abstract Syntax Tree - The core content system for quiz generation.

  IMPORTANT: ALWAYS use ContentAST elements for ALL content generation.
  Never create custom LaTeX, HTML, or Markdown strings manually.

  This system provides cross-format compatibility between:
  - LaTeX/PDF output for printed exams
  - HTML/Canvas output for online quizzes
  - Markdown for documentation

  Key Components:
  - ContentAST.Section: Container for groups of elements (use for get_body/get_explanation)
  - ContentAST.Paragraph: Text blocks that automatically handle spacing
  - ContentAST.Equation: Mathematical equations with proper LaTeX/MathJax rendering
  - ContentAST.Matrix: Mathematical matrices (DON'T use manual \\begin{bmatrix})
  - ContentAST.Table: Data tables with proper formatting
  - ContentAST.Answer: Answer input fields
  - ContentAST.OnlyHtml/OnlyLatex: Platform-specific content

  Examples:
    # Good - uses ContentAST
    body = ContentAST.Section()
    body.add_element(ContentAST.Paragraph(["Calculate the matrix:"]))
    matrix_data = [[1, 2], [3, 4]]
    body.add_element(ContentAST.Matrix(data=matrix_data, bracket_type="b"))

    # Bad - manual LaTeX (inconsistent, error-prone)
    body.add_element(ContentAST.Text("\\\\begin{bmatrix} 1 & 2 \\\\\\\\ 3 & 4 \\\\end{bmatrix}"))
  """
  
  class Element:
    """
    Base class for all ContentAST elements providing cross-format rendering.

    This is the foundation class that all ContentAST elements inherit from.
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
        section = ContentAST.Section()
        section.add_element(ContentAST.Text("Hello world"))
        section.add_element(ContentAST.Equation("x = 5"))

        # Renders to any format
        latex_output = section.render("latex")
        html_output = section.render("html")
    """
    def __init__(self, elements=None, add_spacing_before=False):
      self.elements : List[ContentAST.Element] = elements or []
      self.add_spacing_before = add_spacing_before
    
    def __str__(self):
      return self.render_markdown()
    
    def add_element(self, element):
      self.elements.append(element)
    
    def add_elements(self, elements, new_paragraph=False):
      if new_paragraph:
        self.elements.append(ContentAST.Text(""))
      self.elements.extend(elements)

    def convert_markdown(self, str_to_convert, output_format):
      if output_format == "html":
        # Fast in-process HTML conversion using Python markdown library
        try:
          # Convert markdown to HTML using fast Python library
          html_output = markdown.markdown(str_to_convert)

          # Strip surrounding <p> tags for inline content (matching old behavior)
          if html_output.startswith("<p>") and html_output.endswith("</p>"):
            html_output = html_output[3:-4]

          return html_output.strip()
        except Exception as e:
          log.warning(f"Markdown conversion failed: {e}. Returning original content.")
          return str_to_convert
      else:
        # Keep using Pandoc for LaTeX and other formats (less critical path)
        try:
          output = pypandoc.convert_text(
            str_to_convert,
            output_format,
            format='md',
            extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"]
          )
          return output
        except RuntimeError as e:
          log.warning(f"Specified conversion format '{output_format}' not recognized by pypandoc. Defaulting to markdown")
        return None
    
    def render(self, output_format, **kwargs):
      method_name = f"render_{output_format}"
      if hasattr(self, method_name):
        return getattr(self, method_name)(**kwargs)
      
      return self.render_markdown(**kwargs)  # Fallback to markdown
    
    def render_markdown(self, **kwargs):
      return " ".join(element.render("markdown", **kwargs) for element in self.elements)
    
    def render_html(self, **kwargs):
      html = " ".join(element.render("html", **kwargs) for element in self.elements)
      return f"{'<br>' if self.add_spacing_before else ''}{html}"
    
    def render_latex(self, **kwargs):
      latex = "".join(element.render("latex", **kwargs) for element in self.elements)
      return f"{'\n\n\\vspace{0.5cm}' if self.add_spacing_before else ''}{latex}"

    def render_typst(self, **kwargs):
      """
      Default Typst rendering using markdown → typst conversion via pandoc.

      This provides instant Typst support for all ContentAST elements without
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

    def is_mergeable(self, other: ContentAST.Element):
      return False
  
  class OnlyLatex(Element):
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
        latex_only = ContentAST.OnlyLatex()
        latex_only.add_element(ContentAST.Text("\\newpage"))

        # Add to main content - only appears in PDF
        body.add_element(latex_only)
    """
    def render(self, output_format, **kwargs):
      if output_format != "latex":
        return ""
      return super().render(output_format, **kwargs)

  class OnlyHtml(Element):
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
        html_only = ContentAST.OnlyHtml()
        html_only.add_element(ContentAST.Text("Click submit when done"))

        # Add to main content - only appears in Canvas
        body.add_element(html_only)
    """
    def render(self, output_format, **kwargs):
      if output_format != "html":
        return ""
      return super().render(output_format, **kwargs)
  
  class Document(Element):
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
    Use ContentAST.Section for question bodies and explanations instead.

    Features:
    - Complete LaTeX document headers with all necessary packages
    - Automatic title handling across all formats
    - PDF-ready formatting with proper spacing and layout

    Example (internal use):
        # Usually created automatically by quiz system
        doc = ContentAST.Document(title="Midterm Exam")
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
    // Quiz document settings
    #set page(
      paper: "a4",
      margin: 1.5cm,
    )

    #set text(
      size: 12pt,
    )

    // Math equation settings
    #set math.equation(numbering: none)

    // Paragraph spacing
    #set par(
      spacing: 0.25em,
      leading: 0.5em,
    )

    // Question counter and command
    #let question_num = counter("question")

#let question(points, content, spacing: 3cm, qr_code: none) = {
  block(breakable: false)[
    #line(length: 100%, stroke: 1pt)
    #v(0.0cm)
    #question_num.step()

    *Question #context question_num.display():* (#points #if points == 1 [point] else [points])
    #v(0.25cm)
    // Content on the left, QR pinned right
    #if qr_code != none {
      grid(
        columns: (1fr, auto),
        column-gutter: 0.8em,
        [
          #content
        ],
        [
          #align(top + right)[
            #v(0.0cm)
            #image(qr_code, width: 1.5cm)
          ]
        ],
      )
    } else {
      content
    }

    #v(spacing)
  ]
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

      latex += "\n".join(element.render("latex", **kwargs) for element in self.elements)

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
      typst += "".join(element.render("typst", **kwargs) for element in self.elements)

      return typst
    
  ## Below here are smaller elements generally
  class Question(Element):
    """
    Complete question container with body, explanation, and metadata.

    This class represents a full question with both the question content
    and its explanation/solution. It handles question-level formatting
    like point values, spacing, and PDF layout.

    Note: Most question developers should NOT use this directly.
    It's created automatically by the quiz generation system.
    Focus on building ContentAST.Section objects for get_body() and get_explanation().

    When to use:
    - Creating complete question objects (handled by quiz system)
    - Custom question wrappers (advanced use)

    Example (internal use):
        # Usually created by quiz system from your question classes
        body = ContentAST.Section()
        body.add_element(ContentAST.Paragraph(["What is 2+2?"]))

        explanation = ContentAST.Section()
        explanation.add_element(ContentAST.Paragraph(["2+2=4"]))

        question = ContentAST.Question(body=body, explanation=explanation, value=5)
    """
    def __init__(
        self,
        body: ContentAST.Section,
        explanation: ContentAST.Section,
        name=None,
        value=1,
        interest=1.0,
        spacing=3,
        topic = None,
        question_number=None
    ):
      super().__init__()
      self.name = name
      self.explanation = explanation
      self.body = body
      self.value = value
      self.interest = interest
      self.spacing = spacing
      self.topic = topic # todo: remove this bs.
      self.question_number = question_number  # For QR code generation
    
    def render(self, output_format, **kwargs):
      # Special handling for latex and typst - use dedicated render methods
      if output_format == "typst":
        return self.render_typst(**kwargs)

      # Generate content from all elements
      content = self.body.render(output_format, **kwargs)

      # If output format is latex, add in minipage and question environments
      if output_format == "latex":
        # Build question header - either with or without QR code
        if self.question_number is not None:
          try:
            from QuizGenerator.qrcode_generator import QuestionQRCode

            # Build extra_data dict with regeneration metadata if available
            extra_data = {}
            if hasattr(self, 'question_class_name') and hasattr(self, 'generation_seed') and hasattr(self, 'question_version'):
              if self.question_class_name and self.generation_seed is not None and self.question_version:
                extra_data['question_type'] = self.question_class_name
                extra_data['seed'] = self.generation_seed
                extra_data['version'] = self.question_version

            qr_path = QuestionQRCode.generate_png_path(
              self.question_number,
              self.value,
              **extra_data
            )
            # Build custom question header with QR code centered
            # Format: Question N:  [QR code centered]  __ / points
            question_header = (
              r"\vspace{0.5cm}" + "\n"
              r"\noindent\textbf{Question " + str(self.question_number) + r":} \hfill "
              r"\rule{0.5cm}{0.15mm} / " + str(int(self.value)) + "\n"
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
      content = self.body.render("typst", **kwargs)

      # Generate QR code if question number is available
      qr_param = ""
      if self.question_number is not None:
        try:

          # Build extra_data dict with regeneration metadata if available
          extra_data = {}
          if hasattr(self, 'question_class_name') and hasattr(self, 'generation_seed') and hasattr(self, 'question_version'):
            if self.question_class_name and self.generation_seed is not None and self.question_version:
              extra_data['question_type'] = self.question_class_name
              extra_data['seed'] = self.generation_seed
              extra_data['version'] = self.question_version

          # Generate QR code PNG
          qr_path = QuestionQRCode.generate_png_path(
            self.question_number,
            self.value,
            **extra_data
          )

          # Add QR code parameter to question function call
          qr_param = f', qr_code: "{qr_path}"'

        except Exception as e:
          log.warning(f"Failed to generate QR code for question {self.question_number}: {e}")

      # Use the question function which handles all formatting including non-breaking
      return f"#question({int(self.value)}, spacing: {self.spacing}cm{qr_param})[{content}]\n"

  class Section(Element):
    """
    Primary container for question content - USE THIS for get_body() and get_explanation().

    This is the most important ContentAST class for question developers.
    It serves as the main container for organizing question content
    and should be the return type for your get_body() and get_explanation() methods.

    CRITICAL: Always use ContentAST.Section as the container for:
    - Question body content (return from get_body())
    - Question explanation/solution content (return from get_explanation())
    - Any grouped content that needs to render together

    When to use:
    - As the root container in get_body() and get_explanation() methods
    - Grouping related content elements
    - Organizing complex question content

    Example:
        def get_body(self):
            body = ContentAST.Section()
            body.add_element(ContentAST.Paragraph(["Calculate the determinant:"]))

            matrix_data = [[1, 2], [3, 4]]
            body.add_element(ContentAST.Matrix(data=matrix_data, bracket_type="v"))

            body.add_element(ContentAST.Answer(answer=self.answer, label="Determinant"))
            return body
    """
    def render_typst(self, **kwargs):
      """Render section by directly calling render on each child element."""
      return "".join(element.render("typst", **kwargs) for element in self.elements)
  
  class Text(Element):
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
    - Mathematical content (use ContentAST.Equation instead)
    - Code (use ContentAST.Code instead)
    - Structured content (use ContentAST.Paragraph for grouping)

    Example:
        # Basic text
        text = ContentAST.Text("This is plain text")

        # Emphasized text
        important = ContentAST.Text("Important note", emphasis=True)

        # HTML-only text (hidden from PDF)
        web_note = ContentAST.Text("Click submit", hide_from_latex=True)
    """
    def __init__(self, content : str, *, hide_from_latex=False, hide_from_html=False, emphasis=False):
      super().__init__()
      self.content = content
      self.hide_from_latex = hide_from_latex
      self.hide_from_html = hide_from_html
      self.emphasis = emphasis
    
    def render_markdown(self, **kwargs):
      return f"{'***' if self.emphasis else ''}{self.content}{'***' if self.emphasis else ''}"

    def render_html(self, **kwargs):
      if self.hide_from_html:
        return ""
      # If the super function returns None then we just return content as is
      conversion_results = super().convert_markdown(self.content.replace("#", r"\#"), "html").strip()
      if conversion_results is not None:
        if conversion_results.startswith("<p>") and conversion_results.endswith("</p>"):
          conversion_results = conversion_results[3:-4]
      return conversion_results or self.content
    
    def render_latex(self, **kwargs):
      if self.hide_from_latex:
        return ""
      # Escape # to prevent markdown header conversion in LaTeX
      content = super().convert_markdown(self.content.replace("#", r"\#"), "latex") or self.content
      return content

    def render_typst(self, **kwargs):
      """Render text to Typst, escaping special characters."""
      # Hide HTML-only text from Typst (since Typst generates PDFs like LaTeX)
      if self.hide_from_latex:
        return ""

      # In Typst, # starts code/function calls, so we need to escape it
      content = self.content.replace("#", r"\#")

      # Apply emphasis if needed
      if self.emphasis:
        content = f"*{content}*"

      return content

    def is_mergeable(self, other: ContentAST.Element):
      if not isinstance(other, ContentAST.Text):
        return False
      if self.hide_from_latex != other.hide_from_latex:
        return False
      return True
    
    def merge(self, other: ContentAST.Text):
      self.content = self.render_markdown() + " " + other.render_markdown()
      self.emphasis = False
  
  class TextHTML(Text):
    """
    Convenience class for HTML-only text content.

    Identical to ContentAST.Text with hide_from_latex=True.
    Use when you need text that only appears in Canvas/HTML output.

    Example:
        # Text that only appears in Canvas
        canvas_instruction = ContentAST.TextHTML("Submit your answer above")
    """
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.hide_from_html = False
      self.hide_from_latex = True

  class TextLatex(Text):
    """
    Convenience class for LaTeX-only text content.

    Identical to ContentAST.Text with hide_from_html=True.
    Use when you need text that only appears in PDF output.

    Example:
        # Text that only appears in PDF
        pdf_instruction = ContentAST.TextLatex("Write your answer in the space below")
    """
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.hide_from_html = True
      self.hide_from_latex = False
  
  class Paragraph(Element):
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
    - Single words or short phrases (use ContentAST.Text)
    - Mathematical content (use ContentAST.Equation)
    - Structured data (use ContentAST.Table)

    Example:
        # Multi-line question text
        body.add_element(ContentAST.Paragraph([
            "Consider the following system:",
            "- Process A requires 4MB memory",
            "- Process B requires 2MB memory",
            "How much total memory is needed?"
        ]))

        # Mixed content paragraph
        para = ContentAST.Paragraph([
            "The equation ",
            ContentAST.Equation("x^2 + 1 = 0", inline=True),
            " has no real solutions."
        ])
    """

    def __init__(self, lines_or_elements: List[str|ContentAST.Element] = None):
      super().__init__(add_spacing_before=True)
      for line in lines_or_elements:
        if isinstance(line, str):
          self.elements.append(ContentAST.Text(line))
        else:
          self.elements.append(line)
      
    def render(self, output_format, **kwargs):
      # Add in new lines to break these up visually
      return "\n\n" + super().render(output_format, **kwargs)
  
    def add_line(self, line: str):
      self.elements.append(ContentAST.Text(line))
  
  class Answer(Element):
    """
    Answer input field that renders as blanks in PDF and shows answers in HTML.

    CRITICAL: Use this for ALL answer inputs in questions.
    Creates appropriate input fields that work across both PDF and Canvas formats.
    In PDF, renders as blank lines for students to fill in.
    In HTML/Canvas, can display the answer for checking.

    When to use:
    - Any place where students need to input an answer
    - Numerical answers, short text answers, etc.
    - Questions requiring fill-in-the-blank responses

    Example:
        # Basic answer field
        body.add_element(ContentAST.Answer(
            answer=self.answer,
            label="Result",
            unit="MB"
        ))

        # Multiple choice or complex answers
        body.add_element(ContentAST.Answer(
            answer=[self.answer_a, self.answer_b],
            label="Choose the best answer"
        ))
    """
    def __init__(self, answer: Answer = None, label: str = "", unit: str = "", blank_length=5):
      super().__init__()
      self.answer = answer
      self.label = label
      self.unit = unit
      self.length = blank_length
    
    def render_markdown(self, **kwargs):
      if not isinstance(self.answer, list):
        key_to_display = self.answer.key
      else:
        key_to_display = self.answer[0].key
      return f"{self.label + (':' if len(self.label) > 0 else '')} [{key_to_display}] {self.unit}".strip()
    
    def render_html(self, **kwargs):
      return self.render_markdown()
    
    def render_latex(self, **kwargs):
      return fr"{self.label + (':' if len(self.label) > 0 else '')} \answerblank{{{self.length}}} {self.unit}".strip()

    def render_typst(self, **kwargs):
      """Render answer blank as an underlined space in Typst."""
      # Use the fillline function defined in TYPST_HEADER
      # Width is based on self.length (in cm)
      blank_width = self.length * 0.75  # Convert character length to cm
      blank = f"#fillline(width: {blank_width}cm)"

      label_part = f"{self.label}:" if self.label else ""
      unit_part = f" {self.unit}" if self.unit else ""

      return f"{label_part} {blank}{unit_part}".strip()

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
        code_block = ContentAST.Code(
            "if (x > 0) {\n    print('positive');\n}"
        )
        body.add_element(code_block)

        # Terminal output
        terminal = ContentAST.Code("$ ls -la\ntotal 24\ndrwxr-xr-x 3 user")
    """
    def __init__(self, lines, **kwargs):
      super().__init__(lines)
      self.make_normal = kwargs.get("make_normal", False)
    
    def render_markdown(self, **kwargs):
      content = "```\n" + self.content + "\n```"
      return content
    
    def render_html(self, **kwargs):
      
      return super().convert_markdown(textwrap.indent(self.content, "\t"), "html") or self.content
    
    def render_latex(self, **kwargs):
      content = super().convert_markdown(self.render_markdown(), "latex") or self.content
      if self.make_normal:
        content = (
          r"{\normal "
          + content +
          r"}"
        )
      return content
  
  class Equation(Element):
    """
    Mathematical equation renderer with LaTeX input and cross-format output.

    CRITICAL: Use this for ALL mathematical content instead of manual LaTeX strings.
    Provides consistent math rendering across PDF (LaTeX) and Canvas (MathJax).

    When to use:
    - Any mathematical expressions, equations, or formulas
    - Variables, functions, mathematical notation
    - Both inline math (within text) and display math (separate lines)

    DON'T manually write LaTeX in ContentAST.Text - always use ContentAST.Equation.

    Example:
        # Display equation (separate line, larger)
        body.add_element(ContentAST.Equation("x^2 + y^2 = r^2"))

        # Inline equation (within text)
        paragraph = ContentAST.Paragraph([
            "The solution is ",
            ContentAST.Equation("x = \\frac{-b}{2a}", inline=True),
            " which can be computed easily."
        ])

        # Complex equations
        body.add_element(ContentAST.Equation(r"\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}"))
    """
    def __init__(self, latex, inline=False):
      super().__init__()
      self.latex = latex
      self.inline = inline
    
    def render_markdown(self, **kwargs):
      if self.inline:
        return f"${self.latex}$"
      else:
        return r"$$ \displaystyle " + f"{self.latex}" + r" \; $$"

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

      Typst uses LaTeX-like math syntax with $ delimiters.
      Inline: $equation$
      Display: $ equation $
      """
      if self.inline:
        # Inline math in Typst
        return f"${self.latex}$"
      else:
        # Display math in Typst
        return f"$ {self.latex} $"

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

  class Matrix(Element):
    """
    Mathematical matrix renderer for consistent cross-format display.

    CRITICAL: Use this for ALL matrix and vector notation instead of manual LaTeX.

    DON'T do this:
        # Manual LaTeX (error-prone, inconsistent)
        latex_str = f"\\\\begin{{bmatrix}} {a} & {b} \\\\\\\\ {c} & {d} \\\\end{{bmatrix}}"

    DO this instead:
        # ContentAST.Matrix (consistent, cross-format)
        matrix_data = [[a, b], [c, d]]
        ContentAST.Matrix(data=matrix_data, bracket_type="b")

    For vectors (single column matrices):
        vector_data = [[v1], [v2], [v3]]  # Note: list of single-element lists
        ContentAST.Matrix(data=vector_data, bracket_type="b")

    For LaTeX strings in equations:
        matrix_latex = ContentAST.Matrix.to_latex(matrix_data, "b")
        ContentAST.Equation(f"A = {matrix_latex}")

    Bracket types:
        - "b": square brackets [matrix] - most common for vectors/matrices
        - "p": parentheses (matrix) - sometimes used for matrices
        - "v": vertical bars |matrix| - for determinants
        - "B": curly braces {matrix}
        - "V": double vertical bars ||matrix|| - for norms
    """
    def __init__(self, data, bracket_type="p", inline=False):
      """
      Creates a matrix element that renders consistently across output formats.

      Args:
          data: Matrix data as List[List[numbers/strings]]
                For vectors: [[v1], [v2], [v3]] (column vector)
                For matrices: [[a, b], [c, d]]
          bracket_type: Bracket style - "b" for [], "p" for (), "v" for |, etc.
          inline: Whether to use inline (smaller) matrix formatting
      """
      super().__init__()
      self.data = data
      self.bracket_type = bracket_type
      self.inline = inline

    @staticmethod
    def to_latex(data, bracket_type="p"):
      """
      Convert matrix data to LaTeX string for use in equations.

      Use this when you need a LaTeX string to embed in ContentAST.Equation:
          matrix_latex = ContentAST.Matrix.to_latex([[1, 2], [3, 4]], "b")
          ContentAST.Equation(f"A = {matrix_latex}")

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

    def render_markdown(self, **kwargs):
      matrix_env = "smallmatrix" if self.inline else f"{self.bracket_type}matrix"
      rows = []
      for row in self.data:
        rows.append(" & ".join(str(cell) for cell in row))
      matrix_content = r" \\ ".join(rows)

      if self.inline and self.bracket_type == "p":
        return f"$\\big(\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}\\big)$"
      else:
        return f"$$\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}$$"

    def render_html(self, **kwargs):
      matrix_env = "smallmatrix" if self.inline else f"{self.bracket_type}matrix"
      rows = []
      for row in self.data:
        rows.append(" & ".join(str(cell) for cell in row))
      matrix_content = r" \\ ".join(rows)

      if self.inline:
        return f"<span class='math'>$\\big(\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}\\big)$</span>"
      else:
        return f"<div class='math'>$$\\begin{{{matrix_env}}} {matrix_content} \\end{{{matrix_env}}}$$</div>"

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

  class Table(Element):
    """
    Structured data table with cross-format rendering and proper formatting.

    Creates properly formatted tables that work in PDF, Canvas, and Markdown.
    Automatically handles headers, alignment, and responsive formatting.
    All data is converted to ContentAST elements for consistent rendering.

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
        table = ContentAST.Table(data=data, headers=headers, alignments=["left", "right", "center"])
        body.add_element(table)

        # Mixed content table
        data = [
            [ContentAST.Text("x"), ContentAST.Equation("x^2", inline=True)],
            [ContentAST.Text("y"), ContentAST.Equation("y^2", inline=True)]
        ]
    """
    def __init__(self, data, headers=None, alignments=None, padding=False, transpose=False, hide_rules=False):
      # todo: fix alignments
      # todo: implement transpose
      super().__init__()

      # Normalize data to ContentAST elements
      self.data = []
      for row in data:
        normalized_row = []
        for cell in row:
          if isinstance(cell, ContentAST.Element):
            normalized_row.append(cell)
          else:
            normalized_row.append(ContentAST.Text(str(cell)))
        self.data.append(normalized_row)

      # Normalize headers to ContentAST elements
      if headers:
        self.headers = []
        for header in headers:
          if isinstance(header, ContentAST.Element):
            self.headers.append(header)
          else:
            self.headers.append(ContentAST.Text(str(header)))
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
          rendered_header = header.render(output_format="html")
          result.append(f"      <td style=\"padding: {'5px' if self.padding else '0x'}; font-weight: bold; {align_attr};\"><b>{rendered_header}</b></td>")
        result.append("    </tr>")

      # Render data rows
      for row in self.data:
        result.append("    <tr>")
        for i, cell in enumerate(row):
          if isinstance(cell, ContentAST.Element):
            cell = cell.render(output_format="html")
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
        col_spec = "".join("l" if a == "left" else "r" if a == "right" else "c"
                           for a in self.alignments)
      else:
        col_spec = '|'.join(["l"] * (len(self.headers) if self.headers else len(self.data[0])))

      result = [f"\\begin{{tabular}}{{{col_spec}}}"]
      if not self.hide_rules: result.append("\\toprule")

      if self.headers:
        # Now all headers are ContentAST elements, so render them consistently
        rendered_headers = [header.render(output_format="latex") for header in self.headers]
        result.append(" & ".join(rendered_headers) + " \\\\")
        if not self.hide_rules: result.append("\\midrule")

      for row in self.data:
        # All data cells are now ContentAST elements, so render them consistently
        rendered_row = [cell.render(output_format="latex") for cell in row]
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
      result = [f"#table("]
      result.append(f"  columns: {num_cols},")
      result.append(f"  {align_spec},")

      # Add stroke if not hiding rules
      if not self.hide_rules:
        result.append(f"  stroke: 0.5pt,")
      else:
        result.append(f"  stroke: none,")

      # Render headers
      if self.headers:
        for header in self.headers:
          rendered = header.render(output_format="typst", **kwargs).strip()
          result.append(f"  [*{rendered}*],")

      # Render data rows
      for row in self.data:
        for cell in row:
          rendered = cell.render(output_format="typst", **kwargs).strip()
          result.append(f"  [{rendered}],")

      result.append(")")

      return "\n" + "\n".join(result) + "\n"

  class Picture(Element):
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

        picture = ContentAST.Picture(
            img_data=img_data,
            caption="Memory layout diagram",
            width="80%"
        )
        body.add_element(picture)
    """
    def __init__(self, img_data, caption=None, width=None):
      super().__init__()
      self.img_data = img_data
      self.caption = caption
      self.width = width
    
    def render_markdown(self, **kwargs):
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
  
  class RepeatedProblemPart(Element):
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
            (ContentAST.Matrix([[1], [2]]), "\\cdot", ContentAST.Matrix([[3], [4]])),
            (ContentAST.Matrix([[5], [6]]), "\\cdot", ContentAST.Matrix([[7], [8]]))
        ]
        repeated_part = ContentAST.RepeatedProblemPart(subparts)
        body.add_element(repeated_part)
    """
    def __init__(self, subpart_contents):
      """
      Create a repeated problem part with multiple subquestions.

      Args:
          subpart_contents: List of content for each subpart.
                           Each item can be:
                           - A string (rendered as equation)
                           - A ContentAST.Element
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
          # Single element (ContentAST element or string)
          if hasattr(content, 'render'):
            content_str = content.render('latex', **kwargs)
          else:
            content_str = str(content)
          result.append(f"({letter})\\;& {content_str} &=&\\; {spacing}")

      result.append(r"\end{alignat*}")
      return "\n".join(result)

  class TableGroup(Element):
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
        table_group = ContentAST.TableGroup()

        table_group.add_table(
            label="Page Table #0",
            table=ContentAST.Table(headers=["PTI", "PTE"], data=pt0_data)
        )

        table_group.add_table(
            label="Page Table #1",
            table=ContentAST.Table(headers=["PTI", "PTE"], data=pt1_data)
        )

        body.add_element(table_group)
    """
    def __init__(self):
      super().__init__()
      self.tables = []  # List of (label, table) tuples

    def add_table(self, table: ContentAST.Table, label: str = None):
      """
      Add a table to the group with an optional label.

      Args:
          table: ContentAST.Table to add
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
        # Multiple related answers
        answers = [
            ContentAST.Answer(answer=self.memory_answer, label="Memory used", unit="MB"),
            ContentAST.Answer(answer=self.time_answer, label="Execution time", unit="ms")
        ]
        answer_block = ContentAST.AnswerBlock(answers)
        body.add_element(answer_block)

        # Single answer with better spacing
        single_answer = ContentAST.AnswerBlock(
            ContentAST.Answer(answer=self.result, label="Final result")
        )
    """
    def __init__(self, answers: ContentAST.Answer|List[ContentAST.Answer]):
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
