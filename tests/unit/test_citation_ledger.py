from pathlib import Path

from graphrag.adapters.citation_ledger import FileCitationLedger


def test_missing_file_means_nothing_checked(tmp_path: Path) -> None:
    assert FileCitationLedger(tmp_path / "absent.txt").checked() == set()


def test_marks_are_kept_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "checked.txt"
    FileCitationLedger(path).mark(["2501.00001", "2501.00002"])
    FileCitationLedger(path).mark(["2501.00003"])
    assert FileCitationLedger(path).checked() == {"2501.00001", "2501.00002", "2501.00003"}


def test_file_is_one_id_per_line(tmp_path: Path) -> None:
    path = tmp_path / "checked.txt"
    FileCitationLedger(path).mark(["2501.00001", "2501.00002"])
    assert path.read_bytes() == b"2501.00001\n2501.00002\n"


def test_blank_lines_and_crlf_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "checked.txt"
    path.write_bytes(b"2501.00001\r\n\r\n  2501.00002  \n")
    assert FileCitationLedger(path).checked() == {"2501.00001", "2501.00002"}


def test_marking_nothing_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "checked.txt"
    FileCitationLedger(path).mark([])
    assert not path.exists()


def test_forget_removes_only_the_given_ids(tmp_path: Path) -> None:
    path = tmp_path / "checked.txt"
    ledger = FileCitationLedger(path)
    ledger.mark(["2501.00001", "2501.00002", "2501.00003"])
    ledger.forget(["2501.00002", "2509.99999"])
    assert ledger.checked() == {"2501.00001", "2501.00003"}


def test_forget_on_a_missing_file_does_nothing(tmp_path: Path) -> None:
    path = tmp_path / "absent.txt"
    FileCitationLedger(path).forget(["2501.00001"])
    assert not path.exists()


def test_forget_nothing_leaves_the_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "checked.txt"
    ledger = FileCitationLedger(path)
    ledger.mark(["2501.00001"])
    ledger.forget([])
    assert ledger.checked() == {"2501.00001"}
