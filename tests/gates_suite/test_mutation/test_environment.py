import pytest

from devgates.mutation import select_environment


def test_select_environment_returns_docker_when_platform_is_windows() -> None:
    # Arrange
    platform = "win32"

    # Act
    environment = select_environment(platform)

    # Assert
    assert environment == "docker"


@pytest.mark.parametrize(
    "platform", [pytest.param("linux", id="linux"), pytest.param("darwin", id="macos")]
)
def test_select_environment_returns_native_when_platform_is_not_windows(platform: str) -> None:
    # Arrange: platform fournie par le paramètre

    # Act
    environment = select_environment(platform)

    # Assert
    assert environment == "native"
