"""Safe mathematical expression evaluator."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from completeclaw.tools.base import Tool, ToolResult

# Allowed operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed names (constants and math functions)
_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _safe_names_val(node.id)
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            fn = _SAFE_NAMES[node.func.id]
            args = [_safe_eval(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node)}")
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _safe_names_val(name: str) -> Any:
    return _SAFE_NAMES[name]


class CalculatorTool(Tool):
    """Evaluates safe mathematical expressions.

    Parameters
    ----------
    expression:
        A Python-style mathematical expression string, e.g. ``"2 + 2"``
        or ``"sqrt(144)"``.

    Example::

        tool = CalculatorTool()
        result = tool.run(expression="sqrt(144)")
        print(result.output)  # "12.0"
    """

    name = "calculator"
    description = (
        "Evaluates a mathematical expression. "
        "Input: {'expression': '<math expression>'}. "
        "Supports +, -, *, /, **, %, //, sqrt, sin, cos, tan, log, pi, e."
    )

    def run(self, *, expression: str = "", **kwargs: Any) -> ToolResult:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            value = _safe_eval(tree)
            # Return integer string when the result is a whole number
            output = str(int(value)) if value == int(value) else str(value)
            return ToolResult(output=output)
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))
