"""Code execution tool backed by an E2B sandbox."""

from __future__ import annotations

import os
import shlex

from dotenv import load_dotenv

from agent.tools import ToolResult

try:
    from e2b_code_interpreter import Sandbox as CodeInterpreterSandbox
except ImportError:  # pragma: no cover - depends on installed E2B package variant.
    CodeInterpreterSandbox = None

try:
    from e2b import Sandbox as E2BSandbox
except ImportError:  # pragma: no cover - depends on installed E2B package variant.
    E2BSandbox = None


def _run_with_code_interpreter_sandbox(code: str) -> ToolResult:
    """Execute code with the `e2b_code_interpreter` package when available."""

    sandbox = None
    try:
        sandbox = CodeInterpreterSandbox.create()
        execution = sandbox.run_code(code, timeout=30, request_timeout=30)

        if execution.error is not None:
            error_name = getattr(execution.error, "name", "ExecutionError")
            error_value = getattr(execution.error, "value", "")
            return ToolResult(
                content=f"{error_name}: {error_value}".strip(": "),
                sources=[],
                raw_confidence=0.0,
                tool_name="code_interpreter",
            )

        stdout_lines = getattr(getattr(execution, "logs", None), "stdout", []) or []
        stdout = "\n".join(line.strip() for line in stdout_lines if line is not None).strip()
        content = stdout or (execution.text or "")
        return ToolResult(
            content=content,
            sources=[],
            raw_confidence=1.0,
            tool_name="code_interpreter",
        )
    except Exception as exc:
        return ToolResult(
            content=str(exc),
            sources=[],
            raw_confidence=0.0,
            tool_name="code_interpreter",
        )
    finally:
        if sandbox is not None:
            kill = getattr(sandbox, "kill", None)
            if callable(kill):
                try:
                    kill()
                except Exception:
                    pass


def _run_with_e2b_sandbox(code: str) -> ToolResult:
    """Execute code with the installed `e2b` package using sandbox commands."""

    sandbox = None
    try:
        sandbox = E2BSandbox.create(timeout=30)
        command = f"python -c {shlex.quote(code)}"
        result = sandbox.commands.run(
            command,
            timeout=30,
            request_timeout=30,
        )

        if result.exit_code != 0:
            error_text = (result.stderr or result.error or "").strip()
            return ToolResult(
                content=error_text or f"Execution failed with exit code {result.exit_code}.",
                sources=[],
                raw_confidence=0.0,
                tool_name="code_interpreter",
            )

        return ToolResult(
            content=(result.stdout or "").strip(),
            sources=[],
            raw_confidence=1.0,
            tool_name="code_interpreter",
        )
    except Exception as exc:
        return ToolResult(
            content=str(exc),
            sources=[],
            raw_confidence=0.0,
            tool_name="code_interpreter",
        )
    finally:
        if sandbox is not None:
            kill = getattr(sandbox, "kill", None)
            if callable(kill):
                try:
                    kill()
                except Exception:
                    pass


def run_code(code: str) -> ToolResult:
    """Execute Python code inside an isolated E2B sandbox.

    The function returns captured stdout or other text output when
    execution succeeds. If sandbox startup or execution fails, the
    returned result contains empty or error content with zero confidence.
    """

    load_dotenv()
    if not os.getenv("E2B_API_KEY"):
        return ToolResult(
            content="E2B_API_KEY is not set.",
            sources=[],
            raw_confidence=0.0,
            tool_name="code_interpreter",
        )

    if CodeInterpreterSandbox is not None:
        return _run_with_code_interpreter_sandbox(code)

    if E2BSandbox is not None:
        return _run_with_e2b_sandbox(code)

    return ToolResult(
        content="No supported E2B sandbox package is installed.",
        sources=[],
        raw_confidence=0.0,
        tool_name="code_interpreter",
    )
