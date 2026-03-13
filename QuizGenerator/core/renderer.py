from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterator
import logging

log = logging.getLogger(__name__)


class Renderer:
  @dataclasses.dataclass
  class RendererContext:
    variables: dict[str, Any] = dataclasses.field(default_factory=dict)
  
  def render(self, node: "Node", ctx: RendererContext | None = None) -> str:
    if ctx is None:
      ctx = self.RendererContext()
    
    # Evaluate node first (most nodes return self)
    evaluated = node.evaluate(ctx.variables)
    
    # Then render the evaluated result
    method_name = f"render_{evaluated.identifier()}"
    method = getattr(self, method_name, None)
    return method(evaluated, ctx=ctx) if method else self.render_default(evaluated, ctx=ctx)
  
  def render_default(self, node: "Node", ctx: RendererContext | None = None) -> str:
    if hasattr(node, "children"):
      return self.render_children(node, ctx=ctx)
    if hasattr(node, "content"):
      return str(getattr(node, "content"))
    raise NotImplementedError(
      f"{type(self).__name__} can't render node type {type(node).__name__}"
    )
  
  def render_children(self, node: Any, sep: str = "\n", ctx: RendererContext | None = None) -> str:
    children = getattr(node, "children", [])
    return sep.join(self.render(ch, ctx=ctx) for ch in children)


class SimpleTextRenderer(Renderer):
  
  def render_container(self, node: "Container", ctx: Renderer.RendererContext | None = None) -> str:
    return self.render_children(node, sep="\n\n", ctx=ctx)
  
  def render_leaf(self, node: "Leaf", ctx: Renderer.RendererContext | None = None) -> str:
    return node.content
  
  def render_template(self, node: "Template", ctx: Renderer.RendererContext | None = None) -> str:
    return node.content