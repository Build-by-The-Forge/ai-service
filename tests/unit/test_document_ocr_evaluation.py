from app.core.documents.evaluation import (
    OCRCandidateEvaluation,
    OCRModelCandidate,
    OCRModelEvaluator,
)


def test_ocr_candidate_metadata_is_model_replaceable():
    candidate = OCRModelCandidate(
        name="PP-OCRv3",
        family="paddleocr",
        openvino_supported=True,
        document_oriented=True,
        multilingual_support="limited",
        licensing="apache-2.0",
        status="baseline",
    )

    assert candidate.name == "PP-OCRv3"
    assert candidate.family == "paddleocr"
    assert candidate.openvino_supported is True
    assert candidate.status == "baseline"


def test_ocr_model_evaluator_ranks_candidates_by_score():
    evaluator = OCRModelEvaluator()
    candidates = [
        OCRModelCandidate(
            name="PP-OCRv3",
            family="paddleocr",
            openvino_supported=True,
            document_oriented=True,
            multilingual_support="limited",
            licensing="apache-2.0",
            status="baseline",
        ),
        OCRModelCandidate(
            name="Newer PaddleOCR",
            family="paddleocr",
            openvino_supported=True,
            document_oriented=True,
            multilingual_support="broad",
            licensing="apache-2.0",
            status="candidate",
        ),
        OCRModelCandidate(
            name="Surya",
            family="surya",
            openvino_supported=False,
            document_oriented=True,
            multilingual_support="broad",
            licensing="non-commercial",
            status="candidate",
        ),
    ]

    results = evaluator.evaluate_candidates(
        candidates,
        {
            "PP-OCRv3": {"accuracy": 72, "latency": 80, "memory": 70, "languages": 30},
            "Newer PaddleOCR": {"accuracy": 88, "latency": 74, "memory": 72, "languages": 80},
            "Surya": {"accuracy": 92, "latency": 51, "memory": 46, "languages": 85},
        },
    )

    assert [result.candidate.name for result in results] == [
        "Newer PaddleOCR",
        "Surya",
        "PP-OCRv3",
    ]
    assert isinstance(results[0], OCRCandidateEvaluation)


def test_ocr_model_evaluator_prefers_replaceable_provider_graph():
    evaluator = OCRModelEvaluator()
    candidate = evaluator.select_primary_candidate(
        [
            OCRModelCandidate(
                name="PP-OCRv3",
                family="paddleocr",
                openvino_supported=True,
                document_oriented=True,
                multilingual_support="limited",
                licensing="apache-2.0",
                status="baseline",
            ),
            OCRModelCandidate(
                name="Newer PaddleOCR",
                family="paddleocr",
                openvino_supported=True,
                document_oriented=True,
                multilingual_support="broad",
                licensing="apache-2.0",
                status="candidate",
            ),
        ],
        {
            "PP-OCRv3": {"accuracy": 74, "latency": 80, "memory": 72, "languages": 35},
            "Newer PaddleOCR": {"accuracy": 89, "latency": 76, "memory": 78, "languages": 82},
        },
    )

    assert candidate.name == "Newer PaddleOCR"
    assert candidate.status == "candidate"
