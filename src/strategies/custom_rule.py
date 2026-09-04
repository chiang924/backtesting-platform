from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .base_strategy import BaseStrategy


class RuleSyntaxError(ValueError):
    """Raised when a custom rule uses unsupported or unsafe syntax."""


_PRICE_NAMES = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
_WINDOW_FUNCTIONS = {"sma", "ema", "rsi", "momentum", "er", "highest", "lowest", "atr", "zscore"}
_CROSS_FUNCTIONS = {"crosses_above", "crosses_below"}
_FUNCTION_ARITY = {**{name: 1 for name in _WINDOW_FUNCTIONS}, **{name: 2 for name in _CROSS_FUNCTIONS}}
_BINARY_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_COMPARATORS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
)


def _parse_rule(expression: str) -> ast.Expression:
    rule = expression.strip()
    if not rule:
        raise RuleSyntaxError("Entry and exit rules cannot be empty.")
    if len(rule) > 500:
        raise RuleSyntaxError("Each rule is limited to 500 characters.")
    try:
        tree = ast.parse(rule, mode="eval")
    except SyntaxError as exc:
        raise RuleSyntaxError(f"Invalid rule syntax near column {exc.offset or 1}.") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > 100:
        raise RuleSyntaxError("Rule is too complex; simplify it to fewer than 100 components.")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleSyntaxError(f"Unsupported syntax component: {type(node).__name__}.")
        if isinstance(node, ast.Name) and not isinstance(getattr(node, "ctx", None), ast.Load):
            raise RuleSyntaxError("Assignments are not allowed in a rule.")
        if isinstance(node, ast.Name) and node.id not in _PRICE_NAMES and node.id not in _FUNCTION_ARITY:
            raise RuleSyntaxError(f"Unknown name '{node.id}'. Use lowercase indicator and price names.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_ARITY:
                raise RuleSyntaxError("Only documented indicator functions are allowed.")
            expected = _FUNCTION_ARITY[node.func.id]
            if len(node.args) != expected or node.keywords:
                raise RuleSyntaxError(f"{node.func.id}() requires exactly {expected} positional argument(s).")
            if node.func.id in _WINDOW_FUNCTIONS:
                window_node = node.args[0]
                if not isinstance(window_node, ast.Constant) or isinstance(window_node.value, bool) or not isinstance(window_node.value, int):
                    raise RuleSyntaxError(f"{node.func.id}() requires a whole-number lookback.")
                if not 2 <= window_node.value <= 1000:
                    raise RuleSyntaxError("Indicator lookbacks must be between 2 and 1000 sessions.")
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, (ast.And, ast.Or)):
            raise RuleSyntaxError("Only 'and' and 'or' boolean operators are supported.")
        if isinstance(node, ast.BinOp) and type(node.op) not in _BINARY_OPERATORS:
            raise RuleSyntaxError("Only +, -, *, and / arithmetic operators are supported.")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise RuleSyntaxError("Unsupported unary operator.")
        if isinstance(node, ast.Compare) and any(type(op) not in _COMPARATORS for op in node.ops):
            raise RuleSyntaxError("Only standard numeric comparisons are supported.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise RuleSyntaxError("Rules may contain only numeric or boolean constants.")
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Set, ast.Tuple)):
            raise RuleSyntaxError("Attributes, indexing, and container literals are not allowed.")
    return tree


class _RuleEvaluator:
    def __init__(self, data: pd.DataFrame) -> None:
        if "Close" not in data.columns:
            raise RuleSyntaxError("Market data must contain a Close column.")
        self.index = data.index
        self.series: dict[str, pd.Series] = {}
        for rule_name, column in _PRICE_NAMES.items():
            if column in data.columns:
                self.series[rule_name] = pd.to_numeric(data[column], errors="coerce").astype(float)
        self.cache: dict[tuple[str, int], pd.Series] = {}

    def evaluate(self, tree: ast.Expression) -> pd.Series:
        result = self._eval(tree.body)
        if not isinstance(result, pd.Series) or not is_bool_dtype(result.dtype):
            raise RuleSyntaxError("A rule must evaluate to a true/false condition, such as close > ema(200).")
        return result.reindex(self.index).fillna(False).astype(bool)

    def _as_series(self, value: object) -> pd.Series:
        if isinstance(value, pd.Series):
            return value.reindex(self.index)
        if isinstance(value, (int, float, bool, np.number)):
            return pd.Series(value, index=self.index)
        raise RuleSyntaxError("Rule component did not produce a usable value.")

    def _as_boolean(self, value: object) -> pd.Series:
        series = self._as_series(value)
        if not is_bool_dtype(series.dtype):
            raise RuleSyntaxError("Use a comparison around each indicator before combining it with 'and', 'or', or 'not'.")
        return series.fillna(False).astype(bool)

    def _eval(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.series:
                raise RuleSyntaxError(f"Market data does not contain the '{node.id}' field.")
            return self.series[node.id]
        if isinstance(node, ast.BoolOp):
            values = [self._as_boolean(self._eval(value)) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = result & value if isinstance(node.op, ast.And) else result | value
            return result
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return ~self._as_boolean(value)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            result = _BINARY_OPERATORS[type(node.op)](self._eval(node.left), self._eval(node.right))
            if isinstance(result, pd.Series):
                return result.replace([np.inf, -np.inf], np.nan)
            return result
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            combined = pd.Series(True, index=self.index)
            for op_node, comparator_node in zip(node.ops, node.comparators):
                right = self._eval(comparator_node)
                combined &= self._as_boolean(_COMPARATORS[type(op_node)](left, right))
                left = right
            return combined
        if isinstance(node, ast.Call):
            function_name = node.func.id  # validated by _parse_rule
            if function_name in _WINDOW_FUNCTIONS:
                return self._indicator(function_name, int(node.args[0].value))
            left = self._as_series(self._eval(node.args[0]))
            right = self._as_series(self._eval(node.args[1]))
            if function_name == "crosses_above":
                return (left > right) & (left.shift(1) <= right.shift(1))
            return (left < right) & (left.shift(1) >= right.shift(1))
        raise RuleSyntaxError(f"Unsupported syntax component: {type(node).__name__}.")

    def _indicator(self, name: str, window: int) -> pd.Series:
        key = (name, window)
        if key in self.cache:
            return self.cache[key]
        close = self.series["close"]
        if name == "sma":
            result = close.rolling(window, min_periods=window).mean()
        elif name == "ema":
            result = close.ewm(span=window, adjust=False, min_periods=window).mean()
        elif name == "momentum":
            result = close.pct_change(window, fill_method=None)
        elif name == "highest":
            result = close.rolling(window, min_periods=window).max().shift(1)
        elif name == "lowest":
            result = close.rolling(window, min_periods=window).min().shift(1)
        elif name == "er":
            direction = close.diff(window).abs()
            path = close.diff().abs().rolling(window, min_periods=window).sum()
            result = (direction / path.replace(0, np.nan)).clip(0, 1)
        elif name == "rsi":
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
            average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
            relative_strength = average_gain / average_loss.replace(0, np.nan)
            result = (100 - 100 / (1 + relative_strength)).where(average_loss.ne(0), 100.0)
        elif name == "zscore":
            mean = close.rolling(window, min_periods=window).mean()
            standard_deviation = close.rolling(window, min_periods=window).std(ddof=0)
            result = (close - mean) / standard_deviation.replace(0, np.nan)
        else:  # ATR
            for field in ("high", "low"):
                if field not in self.series:
                    raise RuleSyntaxError("atr() requires High and Low columns in the market data.")
            true_range = pd.concat(
                [
                    self.series["high"] - self.series["low"],
                    (self.series["high"] - close.shift(1)).abs(),
                    (self.series["low"] - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            result = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        self.cache[key] = result.rename(f"{name}({window})")
        return self.cache[key]


class CustomRuleStrategy(BaseStrategy):
    """Compile a small, safe rule language into vectorized entry and exit signals."""

    name = "Custom Rule Strategy"
    description = "Write entry and exit conditions with audited indicators; no Python eval is used."

    def __init__(self, entry_rule: str, exit_rule: str) -> None:
        self.entry_rule = entry_rule.strip()
        self.exit_rule = exit_rule.strip()
        self._entry_tree = _parse_rule(self.entry_rule)
        self._exit_tree = _parse_rule(self.exit_rule)
        lookbacks = [
            int(node.args[0].value)
            for tree in (self._entry_tree, self._exit_tree)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _WINDOW_FUNCTIONS
        ]
        self.minimum_history = max(lookbacks, default=2) + 1

    def rule_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        evaluator = _RuleEvaluator(data)
        return pd.DataFrame(
            {
                "Entry Condition": evaluator.evaluate(self._entry_tree),
                "Exit Condition": evaluator.evaluate(self._exit_tree),
            },
            index=data.index,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        conditions = self.rule_frame(data)
        signals = self._empty_signals(data)
        signals.loc[conditions["Entry Condition"]] = 1
        # Exit wins when both expressions happen to be true on the same close.
        signals.loc[conditions["Exit Condition"]] = -1
        signals.attrs["strategy"] = self.name
        signals.attrs["entry_rule"] = self.entry_rule
        signals.attrs["exit_rule"] = self.exit_rule
        return signals
