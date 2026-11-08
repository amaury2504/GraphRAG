import json

import pytest

from graphrag.adapters.claude_headless import ClaudeHeadlessLLM, parse_result
from graphrag.adapters.http import RequestFailedError
from graphrag.domain.models import Usage

OK = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "  Answer [1].  ",
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 2,
            "output_tokens": 7,
        },
    }
)


class Recorder:
    def __init__(self, reply: tuple[int, str, str]) -> None:
        self.reply = reply
        self.calls: list[tuple[list[str], str, float, str]] = []
        self.system_files: list[str] = []

    def __call__(
        self, command: list[str], stdin: str, timeout: float, cwd: str
    ) -> tuple[int, str, str]:
        self.calls.append((command, stdin, timeout, cwd))
        # the file only exists during the call
        with open(command[command.index("--system-prompt-file") + 1], encoding="utf-8") as f:
            self.system_files.append(f.read())
        return self.reply


def llm(runner: Recorder, model: str | None = None) -> ClaudeHeadlessLLM:
    return ClaudeHeadlessLLM(model=model, executable="claude", runner=runner)


def test_runs_without_tools_with_own_system_prompt_and_content_on_stdin() -> None:
    runner = Recorder((0, OK, ""))
    completion = llm(runner, model="sonnet").complete("SYSTEM\nline two", "SOURCES + QUESTION")
    command, stdin, _, cwd = runner.calls[0]
    assert stdin == "SOURCES + QUESTION"
    assert "SOURCES + QUESTION" not in command  # never on the command line
    assert runner.system_files == ["SYSTEM\nline two"]  # multi-line text travels in a file
    assert not any("\n" in part for part in command)  # a line break would cut a .CMD command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert "--no-session-persistence" in command
    assert command[command.index("--model") + 1] == "sonnet"
    assert "GraphRAG" not in cwd  # an empty temporary directory, not the project
    assert completion.text == "Answer [1]."
    assert completion.usage == Usage(17, 7)


def test_no_model_flag_by_default() -> None:
    runner = Recorder((0, OK, ""))
    llm(runner).complete("s", "u")
    assert "--model" not in runner.calls[0][0]


def test_non_zero_exit_without_output_is_a_request_failure_without_the_prompt() -> None:
    runner = Recorder((1, "", "boom\nSECRET-PROMPT"))
    with pytest.raises(RequestFailedError, match="code 1") as info:
        llm(runner).complete("s", "u")
    assert "SECRET-PROMPT" not in str(info.value)


def test_is_error_is_a_request_failure() -> None:
    body = json.dumps({"is_error": True, "result": "Not logged in", "usage": {}})
    with pytest.raises(RequestFailedError, match="Not logged in"):
        parse_result(body)


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        json.dumps({"result": "x"}),
        json.dumps({"result": 3, "usage": {"output_tokens": 1}}),
        json.dumps({"result": "x", "usage": {}}),
    ],
)
def test_unexpected_shapes_are_rejected(body: str) -> None:
    with pytest.raises(ValueError, match="claude"):
        parse_result(body)


def test_missing_executable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RequestFailedError, match="not found"):
        ClaudeHeadlessLLM()
