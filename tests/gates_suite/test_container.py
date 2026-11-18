import pytest

from devgates.container import container_name


def test_container_name_keeps_a_valid_folder_name() -> None:
    # Arrange
    folder = "outillage-2.v1_x"

    # Act
    name = container_name(folder)

    # Assert
    assert name == "devgates-mutation-outillage-2.v1_x"


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        pytest.param("mon dossier", "devgates-mutation-mon_dossier", id="space"),
        pytest.param("a/b:c", "devgates-mutation-a_b_c", id="slash-and-colon"),
        pytest.param("équipe", "devgates-mutation-_quipe", id="accent"),
    ],
)
def test_container_name_replaces_characters_docker_refuses(folder: str, expected: str) -> None:
    # Arrange / Act
    name = container_name(folder)

    # Assert
    assert name == expected


def test_container_name_differs_for_two_folders() -> None:
    # Arrange / Act
    first = container_name("orchestrateur")
    second = container_name("outillage")

    # Assert
    assert first != second
