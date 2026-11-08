import json
from collections.abc import Callable
from pathlib import Path

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
        self.envs: list[dict[str, str]] = []
        self.on_run: Callable[[dict[str, str]], None] = lambda _env: None
        self.system_files: list[str] = []

    def __call__(
        self, command: list[str], stdin: str, timeout: float, cwd: str, env: dict[str, str]
    ) -> tuple[int, str, str]:
        self.calls.append((command, stdin, timeout, cwd))
        self.envs.append(env)
        self.on_run(env)
        # the file only exists during the call
        with open(command[command.index("--system-prompt-file") + 1], encoding="utf-8") as f:
            self.system_files.append(f.read())
        return self.reply


def llm(runner: Recorder, model: str | None = None, home: Path | None = None) -> ClaudeHeadlessLLM:
    return ClaudeHeadlessLLM(model=model, executable="claude", runner=runner, home=home)


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


def make_login(home: Path, content: str) -> Path:
    path = home / ".claude" / ".credentials.json"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_runs_in_an_empty_home_that_only_holds_the_login(tmp_path: Path) -> None:
    real = tmp_path / "real"
    make_login(real, "LOGIN")
    (real / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
    seen: dict[str, object] = {}

    def look(env: dict[str, str]) -> None:
        home = Path(env["USERPROFILE"])
        seen["same"] = env["HOME"] == env["USERPROFILE"]
        seen["files"] = sorted(p.name for p in (home / ".claude").iterdir())

    runner = Recorder((0, OK, ""))
    runner.on_run = look
    llm(runner, home=real).complete("s", "u")
    assert seen == {"same": True, "files": [".credentials.json"]}
    assert not Path(runner.envs[0]["USERPROFILE"]).exists()  # removed after the call


def test_a_renewed_login_is_copied_back(tmp_path: Path) -> None:
    real = tmp_path / "real"
    original = make_login(real, "OLD")
    runner = Recorder((0, OK, ""))

    def renew(env: dict[str, str]) -> None:
        login = Path(env["USERPROFILE"]) / ".claude" / ".credentials.json"
        login.write_text("NEW", encoding="utf-8")

    runner.on_run = renew
    llm(runner, home=real).complete("s", "u")
    assert original.read_text(encoding="utf-8") == "NEW"


def test_without_a_login_file_nothing_is_copied(tmp_path: Path) -> None:
    runner = Recorder((0, OK, ""))
    llm(runner, home=tmp_path).complete("s", "u")  # an API key in the environment is enough
    assert (tmp_path / ".claude").exists() is False
