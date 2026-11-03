import re

_NEW_STYLE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_STYLE = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")
_VERSION = re.compile(r"v\d+$")
_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://doi.org/10.48550/arxiv.",
    "arxiv:",
)
_DOI_PREFIX = "10.48550/arXiv."
_OPENALEX_URL = "https://openalex.org/"


def normalize_arxiv_id(raw: str) -> str:
    """Return the version-less arXiv identifier (e.g. `2005.11401`)."""
    value = raw.strip()
    for prefix in _PREFIXES:
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
            break
    if not (_NEW_STYLE.match(value) or _OLD_STYLE.match(value)):
        raise ValueError(f"not an arXiv identifier: {raw!r}")
    return _VERSION.sub("", value)


def arxiv_doi_url(arxiv_id: str) -> str:
    return f"https://doi.org/{_DOI_PREFIX}{normalize_arxiv_id(arxiv_id)}"


def arxiv_id_from_doi(doi: str) -> str | None:
    """Inverse of `arxiv_doi_url`; None if the DOI is not an arXiv DOI."""
    lowered = doi.strip().lower()
    prefix = f"https://doi.org/{_DOI_PREFIX}".lower()
    if not lowered.startswith(prefix):
        return None
    try:
        return normalize_arxiv_id(lowered[len(prefix) :])
    except ValueError:
        return None


def short_openalex_id(raw: str) -> str:
    """`https://openalex.org/W123` -> `W123`."""
    return raw.removeprefix(_OPENALEX_URL)
