from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Iterator

import pydantic
import pydantic_core

from .renderer import Renderer

log = logging.getLogger(__name__)


class NodeRegistry:
  _registry: dict[str, type[Node]] = {}
  
  @classmethod
  def register(cls) -> Callable[[type[Node]], type[Node]]:
    def decorator(subclass: type[Node]) -> type[Node]:
      if not issubclass(subclass, Node):
        raise TypeError(f"{subclass} must be a Node subclass")
      
      name = cls.get_canonical_name(subclass.__name__)
      if name in cls._registry:
        existing = cls._registry[name]
        if existing is not subclass:
          log.warning(f"Node type '{name}' already registered as {existing}")
      
      cls._registry[name] = subclass
      return subclass
    
    return decorator
  
  @classmethod
  def create(cls, node_type: str, **kwargs) -> Node:
    return cls._registry[cls.get_canonical_name(node_type)](**kwargs)
  
  @classmethod
  def from_dict(cls, data: dict) -> Node:
    node_type = data["node_type"]
    key = cls.get_canonical_name(node_type)
    try:
      node_cls = cls._registry[key]
    except KeyError as e:
      raise KeyError(f"Unknown node_type '{node_type}'. Known: {list(cls._registry)}") from e
    return node_cls.from_dict(data)
  
  @staticmethod
  def get_canonical_name(node_name: str) -> str:
    return node_name.lower()


class Node:
  """Base node - protocol includes evaluate()"""
  parent: Node | None = None
  
  @classmethod
  def identifier(cls) -> str:
    return NodeRegistry.get_canonical_name(cls.__name__)
  
  def evaluate(self, context: dict[str, Any]) -> Node:
    """
    Evaluate dynamic content. Default: return self unchanged.
    Override in subclasses that need dynamic behavior.
    """
    return self
  
  @classmethod
  def from_dict(cls, data: dict) -> Node:
    if not dataclasses.is_dataclass(cls):
      raise NotImplementedError(f"{cls.__name__}.from_dict not implemented")
    
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
      if f.name not in data:
        continue
      val = data[f.name]
      if f.name == "children":
        kwargs["children"] = [NodeRegistry.from_dict(d) for d in val]
      else:
        kwargs[f.name] = val
    return cls(**kwargs)
  
  def to_dict(self) -> dict[str, Any]:
    if not dataclasses.is_dataclass(self):
      raise NotImplementedError(f"{type(self).__name__}.to_dict not implemented")
    
    out: dict[str, Any] = {"node_type": self.identifier()}
    for f in dataclasses.fields(self):
      val = getattr(self, f.name)
      if f.name == "children":
        out["children"] = [c.to_dict() for c in val]
      else:
        out[f.name] = val
    return out
  
  def render(self, r: Renderer, ctx: Renderer.RendererContext | None = None) -> str:
    return r.render(self, ctx)
  
  def walk(self) -> Iterator[Node]:
    yield self
    if hasattr(self, "children"):
      for child in self.children:
        yield from child.walk()
  
  def find_all(self, node_type: type[Node]) -> list[Node]:
    return [n for n in self.walk() if isinstance(n, node_type)]
  
  # Pydantic integration
  @classmethod
  def __get_pydantic_core_schema__(
      cls,
      source_type: Any,
      handler: pydantic.GetCoreSchemaHandler
  ) -> pydantic_core.core_schema.CoreSchema:
    """Tell Pydantic how to validate/serialize Node"""
    return pydantic_core.core_schema.no_info_plain_validator_function(
      cls._validate_node,
      serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
        lambda instance: instance.to_dict(),
        when_used='json'
      )
    )
  
  @classmethod
  def _validate_node(cls, value: Any) -> Node:
    """Validate input as a Node"""
    if isinstance(value, Node):
      return value
    elif isinstance(value, dict):
      return NodeRegistry.from_dict(value)
    else:
      raise ValueError(f"Cannot convert {type(value)} to Node")


@NodeRegistry.register()
@dataclasses.dataclass()
class Container(Node):
  """A node with children"""
  children: list[Node] = dataclasses.field(default_factory=list)
  
  def add_child(self, child: Node) -> None:
    child.parent = self
    self.children.append(child)
  
  def evaluate(self, context: dict[str, Any]) -> Node:
    """Evaluate all children"""
    evaluated_children = [child.evaluate(context) for child in self.children]
    new_container = Container(children=evaluated_children)
    new_container.parent = self.parent
    return new_container


@NodeRegistry.register()
@dataclasses.dataclass()
class Leaf(Node):
  """A leaf node with content"""
  content: str = ""


@NodeRegistry.register()
@dataclasses.dataclass()
class Template(Leaf):
  """Leaf with template substitution"""
  
  def evaluate(self, context: dict[str, Any]) -> Node:
    try:
      evaluated_content = self.content.format(**context)
      result = Leaf(content=evaluated_content)
      result.parent = self.parent
      return result
    except KeyError:
      return self


@NodeRegistry.register()
@dataclasses.dataclass()
class Conditional(Node):
  """If-then-else node"""
  condition_var: str = ""
  true_branch: Node = dataclasses.field(default_factory=Leaf)
  false_branch: Node = dataclasses.field(default_factory=Leaf)
  
  def evaluate(self, context: dict[str, Any]) -> Node:
    condition = context.get(self.condition_var, False)
    branch = self.true_branch if condition else self.false_branch
    return branch.evaluate(context)


@NodeRegistry.register()
@dataclasses.dataclass()
class PythonCode(Node):
  """Execute Python to generate content"""
  code: str = ""
  
  def evaluate(self, context: dict[str, Any]) -> Node:
    exec_locals = {}
    try:
      exec(self.code, {"__builtins__": __builtins__, "context": context}, exec_locals)
      result = exec_locals.get("result", Leaf(content=""))
      return result if isinstance(result, Node) else Leaf(content=str(result))
    except Exception as e:
      return Leaf(content=f"[Error: {e}]")