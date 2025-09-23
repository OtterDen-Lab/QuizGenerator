from __future__ import annotations

import textwrap
from io import BytesIO
from typing import List, Callable

import pypandoc

from QuizGenerator.misc import log, Answer


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
      try:
        output = pypandoc.convert_text(
          str_to_convert,
          output_format,
          format='md',
          extra_args=["-M2GB", "+RTS", "-K64m", "-RTS"]
        )
        # if output_format == "html":
        #   output = re.sub(r'^<p>(.*)</p>$', r'\1', output, flags=re.DOTALL)
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
      return "".join(element.render("markdown", **kwargs) for element in self.elements)
    
    def render_html(self, **kwargs):
      html = " ".join(element.render("html", **kwargs) for element in self.elements)
      return f"{'<br>' if self.add_spacing_before else ''}{html}"
    
    def render_latex(self, **kwargs):
      latex = "".join(element.render("latex", **kwargs) for element in self.elements)
      return f"{'\n\n\\vspace{0.5cm}' if self.add_spacing_before else ''}{latex}"
  
    def is_mergeable(self, other: ContentAST.Element):
      return False
  
  class OnlyLatex(Element):
    def render(self, output_format, **kwargs):
      if output_format != "latex":
        return ""
      return super().render(output_format, **kwargs)
  
  class OnlyHtml(Element):
    def render(self, output_format, **kwargs):
      if output_format != "html":
        return ""
      return super().render(output_format, **kwargs)
  
  class Document(Element):
    """Root document class that adds document-level rendering"""
    
    LATEX_HEADER = textwrap.dedent(r"""
    \documentclass[12pt]{article}

    % Page layout
    \usepackage[a4paper, margin=1.5cm]{geometry}

    % Math packages
    \usepackage{amsmath}        % For advanced math environments (matrices, equations)
    \usepackage{amsfonts}       % For additional math fonts
    \usepackage{amssymb}        % For additional math symbols

    % Tables and formatting
    \usepackage{booktabs}       % For clean table rules
    \usepackage{array}          % For extra column formatting options
    \usepackage{verbatim}       % For verbatim environments (code blocks)
    \usepackage{enumitem}       % For customized list spacing
    \usepackage{setspace}       % For \onehalfspacing
    
    % Add this after your existing packages
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
    
  ## Below here are smaller elements generally
  class Question(Element):
    def __init__(
        self,
        body: ContentAST.Section,
        explanation: ContentAST.Section,
        name=None,
        value=1,
        interest=1.0,
        spacing=3,
        topic = None
    ):
      super().__init__()
      self.name = name
      self.explanation = explanation
      self.body = body
      self.value = value
      self.interest = interest
      self.spacing = spacing
      self.topic = topic # todo: remove this bs.
    
    def render(self, output_format, **kwargs):
      # Generate content from all elements
      content = self.body.render(output_format, **kwargs)
      
      # If output format is latex, add in minipage and question environments
      if output_format == "latex":
        latex_lines = [
          r"\noindent\begin{minipage}{\textwidth}",
          r"\noindent\makebox[\linewidth]{\rule{\paperwidth}{1pt}}",
          r"\question{" + str(int(self.value)) + r"}",
          r"\noindent\begin{minipage}{0.9\textwidth}",
          content,
          f"\\vspace{{{self.spacing}cm}}"
          r"\end{minipage}",
          r"\end{minipage}",
        ]
        content = '\n'.join(latex_lines)
      
      log.debug(f"content: \n{content}")
      
      return content
  
  class Section(Element):
    """A child class representing a specific section of a question"""
    pass
  
  class Text(Element):
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
      content = super().convert_markdown(self.content, "latex") or self.content
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
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.hide_from_html = False
      self.hide_from_latex = True
      
  class TextLatex(Text):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.hide_from_html = True
      self.hide_from_latex = False
  
  class Paragraph(Element):
    """A block of text that will combine all child elements together."""
    
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
  
  class Code(Text):
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
    def __init__(self, data, headers=None, alignments=None, padding=False, transpose=False, hide_rules=False):
      """
      Generates a ContentAST.Table element
      :param data: data in the table.  List[List[Element]]
      :param headers: headers for the tables
      :param alignments: how should columns be aligned
      :param padding: Add in padding around cells in html table
      :param transpose: apple transpose to table
      """
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
  
  class Picture(Element):
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
  
  class AnswerBlock(Table):
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
