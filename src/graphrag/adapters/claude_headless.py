"""Claude Code in headless mode (`claude -p`) used as a plain text-completion LLM.

Flags and JSON fields follow https://code.claude.com/docs/en/headless.md and
https://code.claude.com/docs/en/cli-reference.md. The result envelope of the CLI is only
documented through the SDK's result message, so `parse_result` checks every field it reads.

Authentication is whatever the `claude` command uses on this machine. A subscription login
is for personal use only: Anthropic does not allow routing other people's requests through
it (https://code.claude.com/docs/en/legal-and-compliance.md). Serve other users with an
`ANTHROPIC_API_KEY` instead.
"""

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from graphrag.adapters.http import RequestFailedError
from graphrag.domain.models import Completion, Usage

# Short instruction on the command line; the sources and the question travel on stdin.
INSTRUCTION = "Follow the system prompt to answer the message given on standard input."

# (command, stdin text, timeout seconds, working directory) -> (exit code, stdout, stderr)
Runner = Callable[[list[str], str, float, str], tuple[int, str, str]]


def run_subprocess(
    command: list[str], stdin: str, timeout: float, cwd: str
) -> tuple[int, str, str]:
    try:
        done = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RequestFailedError(f"claude did not answer within {timeout:.0f}s") from None
    return done.returncode, done.stdout, done.stderr


class ClaudeHeadlessLLM:
    def __init__(
        self,
        *,
        model: str | None = None,
        executable: str | None = None,
        timeout: float = 120.0,
        runner: Runner = run_subprocess,
    ) -> None:
        found = executable or shutil.which("claude")
        if found is None:
            raise RequestFailedError("the `claude` command was not found on PATH")
        self._executable = found
        self._model = model
        self._timeout = timeout
        self._runner = runner

    def command(self, system_file: str) -> list[str]:
        command = [
            self._executable,
            "-p",
            INSTRUCTION,
            "--output-format",
            "json",
            # Replaces Claude Code's own prompt: no agent behaviour. A file, not an argument:
            # on Windows `claude` is a .CMD script and a line break in an argument cuts the
            # command there, silently dropping the flags after it (tools stayed enabled).
            "--system-prompt-file",
            system_file,
            "--tools",
            "",  # no file, shell or web access
            "--max-turns",
            "1",
            "--no-session-persistence",
        ]
        if self._model:
            command += ["--model", self._model]
        return command

    def complete(self, system: str, user: str) -> Completion:
        # An empty working directory: no project hooks, no .mcp.json, no CLAUDE.md to pick up.
        with tempfile.TemporaryDirectory(prefix="graphrag-claude-") as cwd:
            system_file = Path(cwd) / "system.txt"
            system_file.write_text(system, encoding="utf-8")
            command = self.command(str(system_file))
            code, stdout, stderr = self._runner(command, user, self._timeout, cwd)
        if code != 0 and not stdout.strip():
            # Keep only the first line of stderr, cut short: never echo a prompt.
            first = (stderr.strip().splitlines() or ["no message"])[0][:200]
            raise RequestFailedError(f"claude exited with code {code}: {first}")
        return parse_result(stdout)


def parse_result(stdout: str) -> Completion:
    try:
        payload: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("claude did not return JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("unexpected response shape from claude")
    text = payload.get("result")
    if payload.get("is_error") is True:
        raise RequestFailedError(f"claude reported an error: {str(text)[:200]}")
    usage = payload.get("usage")
    if not isinstance(text, str) or not isinstance(usage, dict):
        raise ValueError("unexpected response shape from claude")
    try:
        prompt_tokens = sum(
            int(usage.get(name) or 0)
            for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        completion_tokens = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("unexpected usage shape from claude") from exc
    return Completion(text.strip(), Usage(prompt_tokens, completion_tokens))
