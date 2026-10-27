"""The evidence boundary (P3-5): numbered full passages for the verified
tenant, and an answer kept only if it cites them."""

from app.services.rag.evidence import NOT_COVERED, Evidence, Passage, check_grounded, gather

LONG = (
    "Hess's law states that the total enthalpy change of a reaction is the same "
    "whichever route is taken from reactants to products. " * 4
)


def passage(number, page_start=3, page_end=3):
    return Passage(
        number, f"text {number}", f"chunk-{number}", "asset-chm101", page_start, page_end
    )


async def test_gather_returns_the_tenants_passages_in_full_and_numbered(in_memory_retrieval):
    await in_memory_retrieval.index_document(
        tenant_id="tenant-a", document_id="asset-chm101", document_title="CHM 101", text=LONG
    )

    evidence = await gather("Hess's law", "tenant-a")
    other = await gather("Hess's law", "tenant-b")

    [first] = evidence.passages
    assert (first.number, first.document_id, first.text) == (1, "asset-chm101", LONG.strip())
    assert len(first.text) > 200  # the whole chunk, not a 200-character snippet
    assert other.passages == ()


def test_the_prompt_names_each_passage_with_its_document_and_pages():
    evidence = Evidence("q", (passage(1), passage(2, 4, 5)))

    assert evidence.prompt_block() == (
        "[1] (asset-chm101, p. 3) text 1\n\n[2] (asset-chm101, pp. 4-5) text 2"
    )


def test_citations_are_kept_once_each_in_order():
    answer = check_grounded(
        "Route-independent [2], see also [1] and [2].", Evidence("q", (passage(1), passage(2)))
    )

    assert answer.grounded is True
    assert [c["number"] for c in answer.citations] == [1, 2]


def test_an_empty_answer_or_no_evidence_is_not_covered():
    assert check_grounded("", Evidence("q", (passage(1),))).text == NOT_COVERED
    assert check_grounded("Something [1].", Evidence("q", ())).text == NOT_COVERED
