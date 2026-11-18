from pathlib import Path

import pytest

from devgates.lock import RunLockedError, run_lock


def alive(_pid: int) -> bool:
    return True


def dead(_pid: int) -> bool:
    return False


def test_lock_is_acquired_when_no_other_run_holds_it(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "gates.lock"

    # Act
    with run_lock(path, pid=4242, is_alive=alive):
        content = path.read_text(encoding="utf-8")

    # Assert
    assert content == "4242"


def test_lock_is_refused_with_a_clear_message_when_another_run_holds_it(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "gates.lock"
    path.write_text("999", encoding="utf-8")

    # Act
    with pytest.raises(RunLockedError) as raised, run_lock(path, pid=1, is_alive=alive):
        pass

    # Assert
    assert str(raised.value) == "gates déjà en cours dans ce dossier (processus 999)"
    assert path.read_text(encoding="utf-8") == "999"


def test_lock_is_released_after_the_run(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "gates.lock"

    # Act
    with run_lock(path, pid=1, is_alive=alive):
        pass

    # Assert
    assert not path.exists()


def test_lock_is_released_when_the_run_fails(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "gates.lock"

    # Act
    with pytest.raises(ValueError, match="boom"), run_lock(path, pid=1, is_alive=alive):
        raise ValueError("boom")

    # Assert
    assert not path.exists()


@pytest.mark.parametrize(
    "leftover", ["999", "pas un nombre", ""], ids=["dead-pid", "garbage", "empty"]
)
def test_lock_left_by_a_dead_process_is_taken_over(tmp_path: Path, leftover: str) -> None:
    # Arrange
    path = tmp_path / "gates.lock"
    path.write_text(leftover, encoding="utf-8")

    # Act
    with run_lock(path, pid=7, is_alive=dead):
        content = path.read_text(encoding="utf-8")

    # Assert
    assert content == "7"
