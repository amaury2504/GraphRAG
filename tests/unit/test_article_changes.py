from graphrag.domain.models import ArticleChanges


def test_modified_ids_are_concatenated_when_changes_are_added() -> None:
    total = ArticleChanges(modified=1, modified_ids=("a",)) + ArticleChanges(
        new=2, modified=1, modified_ids=("b",)
    )
    assert total.modified_ids == ("a", "b")
    assert (total.new, total.modified) == (2, 2)


def test_counts_alone_decide_equality() -> None:
    assert ArticleChanges(modified=1, modified_ids=("a",)) == ArticleChanges(modified=1)
