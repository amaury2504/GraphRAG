"""Candidate evaluation questions: pick articles by topic, have the LLM write each question.

Articles are grouped by *topic proximity only* (vector and keyword neighbours, mixed), never
through citations or authors, so the set does not favour the graph configuration. The
neighbours come from the retrieval indexes themselves, which is a known limit: questions
about neighbouring articles are somewhat easier for those same indexes. Candidates are
written with `validated: false`; a person reads, fixes and validates each one.
"""

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, zip_longest
from typing import Any, cast

from graphrag.domain.chunking import search_text
from graphrag.domain.evaluation import QUESTION_TYPES, QuestionStyle, QuestionType
from graphrag.domain.models import Article, SearchHit
from graphrag.ports.embedding import Embedder
from graphrag.ports.llm import LLM
from graphrag.ports.retrieval import ChunkSearch

MIN_WORDS = 80  # shorter abstracts do not carry enough to ask about
MIN_SPAN_DAYS = 90  # an "evolution" question needs articles months apart
NEIGHBOURS = 12  # fetched from each of the two searches
POOL = 6  # the closest neighbours a relational or comparative group is drawn from
EVOLUTIVE_POOL = 8  # same for an evolution group: far in time but still on topic
MAX_ATTEMPTS = 60

GROUP_SIZE: Mapping[str, int] = {"factual": 1, "comparative": 2, "relational": 3, "evolutive": 3}

_NAMED_RULES = """\
- The question must be self-contained and NAME what it asks about: use the exact names of the
  systems, methods, datasets or topics, as written in the titles and abstracts.
- Never use vague references such as "these works", "the two approaches", "the first article",
  "the paper" or "both systems": a reader who has not seen the articles must know what is meant.
- It must need the content of ALL the given articles, and be answerable from them alone.
- Do not copy long phrases from the abstracts; keep the proper names."""

_DESCRIPTIVE_RULES = """\
- The question must be self-contained and DESCRIBE what it asks about the way a reader who has
  never seen these articles would: the problem, the mechanism or the setting, in your own words.
- Do NOT use the name of any system, method or framework introduced by the articles, and do not
  reuse the distinctive words of their titles. Well-known public datasets and benchmarks may
  be named.
- Never use vague references such as "these works", "the two approaches", "the first article",
  "the paper" or "both systems": say what is meant by describing it.
- It must need the content of ALL the given articles, and be answerable from them alone.
- Do not copy long phrases from the abstracts."""

_COMMON_RULES = """\
- The expected answer is 2 to 4 sentences, using only facts stated in the abstracts. If it
  links the articles in a way the abstracts do not state, it says so ("the abstracts do not
  say..."). If it quotes figures from different articles, it says whether they measure the
  same thing.
- Write in English.
- If the articles share no specific subject that one precise question could be about, do not
  force a link: reply {"skip": "<one sentence saying why>"} instead.
Reply with a JSON object only: {"question": ..., "expected_answer": ...} or {"skip": ...}."""


def _system_prompt(rules: str) -> str:
    return (
        "You write evaluation questions for a search system over scientific articles "
        "(AI and RAG).\n"
        "You are given numbered articles (title, date, abstract). "
        "Write ONE question and its expected answer.\n"
        f"Rules:\n{rules}\n{_COMMON_RULES}"
    )


SYSTEM_PROMPT = _system_prompt(_NAMED_RULES)
DESCRIPTIVE_SYSTEM_PROMPT = _system_prompt(_DESCRIPTIVE_RULES)
SYSTEM_PROMPTS: Mapping[str, str] = {
    "named": SYSTEM_PROMPT,
    "descriptive": DESCRIPTIVE_SYSTEM_PROMPT,
}

TYPE_INSTRUCTIONS: Mapping[str, str] = {
    "factual": "Ask a precise factual question about the method, result or setting of the article.",
    "relational": (
        "Ask how the approaches of these articles relate: what they share, how one builds on "
        "or addresses a limitation of another, or how they fit together."
    ),
    "comparative": (
        "Ask for a comparison of the two approaches on a concrete dimension "
        "(goal, mechanism, assumptions, strengths or limits)."
    ),
    "evolutive": (
        "Ask how the idea or approach evolved over time across these articles "
        "(the articles are listed from the earliest to the latest)."
    ),
}


@dataclass(frozen=True, slots=True)
class QuestionPlan:
    type: QuestionType
    articles: tuple[Article, ...]  # the seed first, except for "evolutive" (oldest first)
    style: QuestionStyle = "named"


def normalized_title(article: Article) -> str:
    return " ".join(article.title.lower().split())


def title_groups(articles: Sequence[Article]) -> dict[str, list[str]]:
    """Ids of articles sharing a title: the same work deposited twice on arXiv."""
    groups: dict[str, list[str]] = defaultdict(list)
    for article in articles:
        groups[normalized_title(article)].append(article.arxiv_id)
    return dict(groups)


def topic_neighbours(
    article: Article, search: ChunkSearch, embedder: Embedder, k: int = NEIGHBOURS
) -> list[str]:
    """Article ids close in topic, alternating the semantic and the keyword ranking."""
    vector = embedder.embed_documents([search_text(article.title, article.abstract)])[0]
    by_meaning = _article_ids(search.vector_search(vector, k * 2))
    by_keywords = _article_ids(search.keyword_search(article.title, k * 2))
    merged: list[str] = []
    for candidate in chain.from_iterable(zip_longest(by_meaning, by_keywords)):
        if candidate is not None and candidate != article.arxiv_id and candidate not in merged:
            merged.append(candidate)
    return merged


def _article_ids(hits: Sequence[SearchHit]) -> list[str]:
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.arxiv_id)
    return list(seen)


def plan_questions(
    articles: Sequence[Article],
    search: ChunkSearch,
    embedder: Embedder,
    *,
    per_type: int,
    seed: int,
    types: Sequence[QuestionType] = QUESTION_TYPES,
    exclude: Collection[str] = (),
    min_words: int = MIN_WORDS,
    min_span_days: int = MIN_SPAN_DAYS,
    style: QuestionStyle = "named",
) -> list[QuestionPlan]:
    """Choose `per_type` groups of articles for each of `types`, never reusing an article.

    Articles in `exclude` (already asked about), and their twin deposits, are never chosen.
    """
    rng = random.Random(seed)
    by_id = {a.arxiv_id: a for a in articles}
    same_title = title_groups(articles)
    eligible = [a for a in articles if len(a.abstract.split()) >= min_words]
    eligible_set = set(eligible)
    used: set[str] = set()
    for arxiv_id in exclude:  # the article and its twin deposits
        if arxiv_id in by_id:
            used.update(same_title[normalized_title(by_id[arxiv_id])])

    def usable(article: Article) -> bool:
        return article.arxiv_id not in used and article in eligible_set

    plans: list[QuestionPlan] = []
    for kind in types:
        for _ in range(per_type):
            for _attempt in range(MAX_ATTEMPTS):
                free = [a for a in eligible if usable(a)]
                if not free:
                    raise RuntimeError(f"no unused article left for a {kind} question")
                seed_article = rng.choice(free)
                neighbours = (
                    []
                    if kind == "factual"
                    else [
                        by_id[i]
                        for i in topic_neighbours(seed_article, search, embedder)
                        if i in by_id and usable(by_id[i])
                    ]
                )
                group = choose_group(kind, seed_article, neighbours, rng, min_span_days)
                if group is not None:
                    for article in group:
                        used.update(same_title[normalized_title(article)])
                    plans.append(QuestionPlan(kind, group, style))
                    break
            else:
                raise RuntimeError(f"could not find a group of articles for a {kind} question")
    return plans


def choose_group(
    kind: str,
    seed_article: Article,
    neighbours: list[Article],
    rng: random.Random,
    min_span_days: int,
) -> tuple[Article, ...] | None:
    if kind == "factual":
        return (seed_article,)
    # keep one article per distinct title among the closest neighbours
    pool: list[Article] = []
    titles = {normalized_title(seed_article)}
    for candidate in neighbours:
        if normalized_title(candidate) not in titles:
            titles.add(normalized_title(candidate))
            pool.append(candidate)
    if kind == "evolutive":
        everyone = sorted(
            [seed_article, *pool[:EVOLUTIVE_POOL]], key=lambda a: (a.published, a.arxiv_id)
        )
        if len(everyone) < GROUP_SIZE[kind]:
            return None
        oldest, newest = everyone[0], everyone[-1]
        if (newest.published - oldest.published).days < min_span_days:
            return None
        return (oldest, everyone[len(everyone) // 2], newest)
    needed = GROUP_SIZE[kind] - 1
    close = pool[:POOL]
    if len(close) < needed:
        return None
    return (seed_article, *rng.sample(close, needed))


def build_prompt(plan: QuestionPlan) -> str:
    blocks = [
        f"[{n}] {a.title} ({a.published.isoformat()})\n{' '.join(a.abstract.split())}"
        for n, a in enumerate(plan.articles, start=1)
    ]
    return (
        f"Question type: {plan.type}. {TYPE_INSTRUCTIONS[plan.type]}\n\n"
        "Articles:\n\n" + "\n\n".join(blocks)
    )


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class UnrelatedArticlesError(ValueError):
    """The model declined: the articles share no subject a precise question could be about."""


def parse_generation(text: str) -> tuple[str, str]:
    """Extract (question, expected_answer) from a model reply, tolerating code fences."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise ValueError("no JSON object in the reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in the reply: {exc.msg}") from None
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    fields = cast("dict[str, Any]", data)
    if "skip" in fields:
        raise UnrelatedArticlesError(f"no specific shared subject: {fields['skip']}")
    question, answer = fields.get("question"), fields.get("expected_answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("the reply has no question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("the reply has no expected_answer")
    return question.strip(), answer.strip()


_VAGUE = re.compile(
    r"\b(?:these|those)\s+(?:\w+\s+)?(?:works?|articles?|papers?|approaches|systems|methods"
    r"|studies|frameworks|techniques|models)\b"
    r"|\bthe\s+(?:two|three|four)\b"
    r"|\b(?:the\s+)?(?:first|second|third)\s+(?:work|article|paper|approach|system|method"
    r"|study|framework)\b"
    r"|\b(?:this|the)\s+(?:article|paper|study|work)s?\b"
    r"|\bboth\s+(?:works?|articles?|papers?|approaches|systems|methods)\b",
    re.IGNORECASE,
)
MAX_REWRITES = 1  # one more attempt, told what was wrong


def find_vague_wording(question: str) -> str | None:
    """The first vague reference ("these works", "the two approaches"...) or None.

    Such a question does not say what it is about, so no search system could find its articles.
    """
    match = _VAGUE.search(question)
    return match.group(0) if match else None


MAX_LEAK_FREQUENCY = 3  # a title word found in at most this many titles identifies its article

_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    re.findall(
        r"\w+",
        """a an and are as at be by can do does for from has have how in into is it its more most
        of on or over that the their this to under using via was were what when which who why with
        within without than then them these those through""",
    )
)


def content_words(text: str) -> set[str]:
    """Lower-case words of at least 3 characters, without the most common function words."""
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOP_WORDS}


def title_word_frequencies(articles: Sequence[Article]) -> Counter[str]:
    """For each word, the number of distinct titles that contain it."""
    distinct = {normalized_title(a) for a in articles}
    return Counter(word for title in distinct for word in content_words(title))


def find_title_leak(
    question: str,
    titles: Sequence[str],
    frequencies: Mapping[str, int],
    max_frequency: int = MAX_LEAK_FREQUENCY,
) -> str | None:
    """A rare word of the expected articles' titles that the question repeats, or None.

    A word found in at most `max_frequency` titles (a system name such as "ARIA") lets the
    keyword search jump straight to the article, so a descriptive question must not use it.
    """
    title_words = set().union(*(content_words(t) for t in titles)) if titles else set()
    leaked = sorted(
        word
        for word in content_words(question) & title_words
        if frequencies.get(word, 0) <= max_frequency
    )
    return leaked[0] if leaked else None


def title_overlap(question: str, titles: Sequence[str]) -> float:
    """Share of the question's content words that also appear in the expected titles."""
    words = content_words(question)
    if not words:
        return 0.0
    title_words = set().union(*(content_words(t) for t in titles)) if titles else set()
    return round(len(words & title_words) / len(words), 3)


def write_question(
    plan: QuestionPlan, llm: LLM, frequencies: Mapping[str, int] | None = None
) -> tuple[str, str]:
    """Ask the model, and once more if the question is vague (or, if descriptive, leaks a name).

    A descriptive plan needs the title word `frequencies`. Raises ValueError on failure.
    """
    if plan.style == "descriptive" and frequencies is None:
        raise ValueError("a descriptive question needs the title word frequencies to be checked")
    system = SYSTEM_PROMPTS[plan.style]
    titles = [a.title for a in plan.articles]
    prompt = build_prompt(plan)
    attempt_prompt = prompt
    problem = ""
    for _ in range(MAX_REWRITES + 1):
        question, answer = parse_generation(llm.complete(system, attempt_prompt).text)
        vague = find_vague_wording(question)
        leak = (
            find_title_leak(question, titles, frequencies)
            if plan.style == "descriptive" and frequencies is not None
            else None
        )
        if vague is not None:
            problem = f"still vague after a rewrite: {vague!r}"
            advice = f"it says {vague!r}, which does not say what it is about. Rewrite it " + (
                "describing the specific problem or mechanism."
                if plan.style == "descriptive"
                else "naming the specific systems, methods or topics."
            )
        elif leak is not None:
            problem = f"still names the article after a rewrite: {leak!r}"
            advice = (
                f"it uses {leak!r}, a distinctive word of an article title. Rewrite it "
                "describing the idea without that word or any name of the systems."
            )
        else:
            return question, answer
        attempt_prompt = f"{prompt}\n\nYour previous question was rejected: {advice}"
    raise ValueError(problem)


def generate_candidates(
    plans: Sequence[QuestionPlan],
    llm: LLM,
    same_title: Mapping[str, Sequence[str]],
    *,
    first_index: int = 1,
    log: Callable[[str], None] = print,
    frequencies: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One LLM call per plan. Returns the candidate records and a message per failed plan."""
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    counters: dict[str, int] = defaultdict(lambda: first_index)
    for plan in plans:
        label = f"{plan.type} on {', '.join(a.arxiv_id for a in plan.articles)}"
        try:
            question, answer = write_question(plan, llm, frequencies)
        except ValueError as exc:
            failures.append(f"{label}: {exc}")
            log(f"failed: {label}: {exc}")
            continue
        number = counters[plan.type]
        counters[plan.type] += 1
        ids = [a.arxiv_id for a in plan.articles]
        records.append(
            {
                "id": f"{plan.type}-{number:02d}",
                "type": plan.type,
                "style": plan.style,
                "question": question,
                "expected_answer": answer,
                "expected_articles": ids,
                "equivalents": {
                    a.arxiv_id: others
                    for a in plan.articles
                    if (others := [i for i in same_title[normalized_title(a)] if i != a.arxiv_id])
                },
                "validated": False,
                # computed on the generated text: stale if you rewrite the question by hand
                "title_overlap": title_overlap(question, [a.title for a in plan.articles]),
                "source_titles": [a.title for a in plan.articles],
                "source_dates": [a.published.isoformat() for a in plan.articles],
            }
        )
        log(f"generated {records[-1]['id']}")
    return records, failures
