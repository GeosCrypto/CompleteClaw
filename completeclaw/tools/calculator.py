"""Calculator tool – safe arithmetic expression evaluation."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from completeclaw.tools.base import Tool, ToolResult

# Supported operators
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math constants / functions
_SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "sqrt": math.sqrt,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return float(_SAFE_NAMES[node.id])  # type: ignore[arg-type]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {node.op!r}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator: {node.op!r}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        fn = _SAFE_NAMES.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unknown function: {node.func.id!r}")
        args = [_eval_node(a) for a in node.args]
        return float(fn(*args))  # type: ignore[operator]
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class CalculatorTool(Tool):
    """Safe arithmetic calculator.

    Evaluates mathematical expressions without using ``eval()``.
    Supports ``+``, ``-``, ``*``, ``/``, ``**``, ``//``, ``%`` and a
    selection of :mod:`math` functions (``sqrt``, ``sin``, ``cos``, …).

    Parameters
    ----------
    expression:
        The math expression to evaluate (e.g. ``"2 ** 10 + sqrt(9)"``).

    Example::

        tool = CalculatorTool()
        result = tool.run(expression="sqrt(144) + 2 ** 3")
        print(result.output)   # "20.0"
    """

    name = "calculator"
    description = (
        "Evaluates a mathematical expression.  "
        "Input: {\"expression\": \"<math expr>\"}."
    )

    def run(self, *, expression: str = "", **kwargs: Any) -> ToolResult:  # type: ignore[override]
        expression = expression.strip()
        if not expression:
            return ToolResult(output="", success=False, error="No expression provided.")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval_node(tree)
            # Return a clean string: drop the ".0" for whole numbers
            output = str(int(result)) if result == int(result) else str(result)
            return ToolResult(output=output, metadata={"expression": expression, "result": result})
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))
