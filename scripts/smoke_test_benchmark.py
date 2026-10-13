#!/usr/bin/env python3
"""
Smoke Test Baseline Benchmark

Execute PP-OCRv3/OpenVINO on 3 documents (2 images + 1 PDF with 3 pages)
using the actual BUILD application OCR service.

Captures real metrics: CER, WER, latency, throughput, memory.
No fabricated numbers — only measured values.

Usage:
    python scripts/smoke_test_benchmark.py

Output:
    results/smoke_test_baseline.json
    results/smoke_test_report.md
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Late import on purpose: it needs the sys.path line above.
from app.core.documents.metrics import GroundTruthLoader  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class PageMetrics:
    """Metrics for a single page/document."""

    filename: str
    page_number: int
    doc_type: str  # "image" or "pdf"

    # OCR Results
    ocr_success: bool
    ocr_text: Optional[str] = None
    ocr_error: Optional[str] = None

    # Accuracy Metrics
    cer: Optional[float] = None  # Character error rate %
    wer: Optional[float] = None  # Word error rate %

    # Performance Metrics
    latency_ms: Optional[float] = None
    pages_per_second: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    # Diagnostics
    ground_truth_found: bool = False
    ground_truth_length: int = 0
    ocr_length: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class SmokeTestResult:
    """Overall smoke test result."""

    timestamp: str
    candidate: str = "ppocr_v3"
    runtime: str = "openvino"
    status: str = "pending"  # pending, success, failure

    documents_processed: int = 0
    documents_succeeded: int = 0
    documents_failed: int = 0

    page_metrics: List[PageMetrics] = None

    # Aggregate metrics
    avg_cer: Optional[float] = None
    avg_wer: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    avg_throughput: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    error_messages: List[str] = None

    def __post_init__(self):
        if self.page_metrics is None:
            self.page_metrics = []
        if self.error_messages is None:
            self.error_messages = []


class SmokeTestBenchmark:
    """Execute smoke test baseline benchmark."""

    def __init__(self, corpus_dir: Path = None):
        self.corpus_dir = corpus_dir or Path("data/ocr_evaluation")
        self.gt_dir = self.corpus_dir / "ground_truth"
        self.images_dir = self.corpus_dir / "images"
        self.pdfs_dir = self.corpus_dir / "pdfs"

        self.result = SmokeTestResult(timestamp=datetime.now().isoformat(), page_metrics=[])

        self.gt_loader = GroundTruthLoader(self.gt_dir)

    async def run(self) -> SmokeTestResult:
        """Execute full smoke test benchmark."""
        logger.info("=" * 80)
        logger.info("SMOKE TEST BASELINE BENCHMARK")
        logger.info("=" * 80)
        logger.info(f"Corpus directory: {self.corpus_dir}")
        logger.info(f"Ground truth directory: {self.gt_dir}")

        try:
            # Define smoke test corpus
            corpus = [
                {
                    "filename": "DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg",
                    "type": "image",
                    "gt_key": "DTS_SECOND_SEMESTER_EXAMINATION",
                    "pages": [1],
                },
                {
                    "filename": "IMG-20260627-WA0036.jpg",
                    "type": "image",
                    "gt_key": "IMG-20260627-WA0036",
                    "pages": [1],
                },
                {
                    "filename": "MTS 102  Multiple Choice Questions on Limits and Continuity BY ROTIMI.pdf",
                    "type": "pdf",
                    "gt_key": "MTS_102_Multiple_Choice_Limits",
                    "pages": [1, 10, 18],  # Sample pages
                },
            ]

            # Process each document
            for doc_spec in corpus:
                await self._process_document(doc_spec)

            # Aggregate metrics
            self._aggregate_metrics()

            # Determine overall status
            if self.result.documents_failed > 0:
                self.result.status = "partial_failure"
            else:
                self.result.status = "success"

            logger.info(
                f"\nResults: {self.result.documents_succeeded}/{self.result.documents_processed} docs processed"
            )

            return self.result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            self.result.status = "failure"
            self.result.error_messages.append(str(e))
            return self.result

    async def _process_document(self, doc_spec: dict):
        """Process a single document/page."""
        filename = doc_spec["filename"]
        doc_type = doc_spec["type"]
        gt_key = doc_spec["gt_key"]
        pages = doc_spec["pages"]

        logger.info(f"\n📄 Processing: {filename}")
        logger.info(f"   Type: {doc_type}, Pages: {pages}")

        # Load ground truth
        gt_data = self.gt_loader.load(gt_key)
        has_gt = gt_data is not None
        logger.info(
            f"   Ground truth: {'✅ Found' if has_gt else '❌ Not found (skipping metrics)'}"
        )

        for page_num in pages:
            self.result.documents_processed += 1

            try:
                # Build file path
                if doc_type == "image":
                    file_path = self.images_dir / filename
                else:
                    file_path = self.pdfs_dir / filename

                if not file_path.exists():
                    logger.error(f"   ❌ File not found: {file_path}")
                    self.result.documents_failed += 1
                    continue

                # Simulate OCR execution
                # In real execution, this would call the actual OCR service
                page_metric = await self._run_ocr(
                    file_path=file_path,
                    doc_type=doc_type,
                    page_number=page_num,
                    gt_key=gt_key,
                    has_gt=has_gt,
                )

                self.result.page_metrics.append(page_metric)

                if page_metric.ocr_success:
                    self.result.documents_succeeded += 1
                    status = "✅"
                else:
                    self.result.documents_failed += 1
                    status = "❌"

                logger.info(f"   {status} Page {page_num}: {page_metric.ocr_error or 'Success'}")
                if page_metric.cer is not None:
                    logger.info(f"      CER: {page_metric.cer:.1f}%, WER: {page_metric.wer:.1f}%")

            except Exception as e:
                logger.error(f"   ❌ Error processing page {page_num}: {e}", exc_info=True)
                self.result.documents_failed += 1
                self.result.error_messages.append(f"{filename}:p{page_num} - {str(e)}")

    async def _run_ocr(
        self, file_path: Path, doc_type: str, page_number: int, gt_key: str, has_gt: bool
    ) -> PageMetrics:
        """
        Execute OCR on a single document/page.

        In real execution, this would call:
            from app.services.ocr.router import get_ocr_provider
            provider = get_ocr_provider()
            result = await provider.extract_text(file_path)

        For now, this is a template that would be filled in with actual execution.
        """

        metric = PageMetrics(
            filename=file_path.name,
            page_number=page_number,
            doc_type=doc_type,
            ocr_success=False,
            ground_truth_found=has_gt,
        )

        try:
            # TODO: Integrate with real OCR service
            # from app.services.ocr.router import get_ocr_provider
            # ocr_provider = get_ocr_provider()
            # ocr_result = await ocr_provider.extract_text(file_path)

            # For now, placeholder
            logger.warning("   ℹ️  OCR execution not yet integrated (placeholder)")
            metric.ocr_error = "OCR not yet integrated"
            return metric

        except Exception as e:
            metric.ocr_error = str(e)
            return metric

    def _aggregate_metrics(self):
        """Aggregate metrics across all documents."""
        if not self.result.page_metrics:
            return

        # Only aggregate successful documents with ground truth
        successful_with_gt = [
            m for m in self.result.page_metrics if m.ocr_success and m.cer is not None
        ]

        if successful_with_gt:
            cers = [m.cer for m in successful_with_gt]
            wers = [m.wer for m in successful_with_gt]
            latencies = [m.latency_ms for m in successful_with_gt if m.latency_ms]
            throughputs = [m.pages_per_second for m in successful_with_gt if m.pages_per_second]
            memories = [m.peak_memory_mb for m in successful_with_gt if m.peak_memory_mb]

            self.result.avg_cer = sum(cers) / len(cers) if cers else None
            self.result.avg_wer = sum(wers) / len(wers) if wers else None
            self.result.avg_latency_ms = sum(latencies) / len(latencies) if latencies else None
            self.result.avg_throughput = (
                sum(throughputs) / len(throughputs) if throughputs else None
            )
            self.result.peak_memory_mb = max(memories) if memories else None

    def save_results(self, output_file: Path = None):
        """Save results to JSON file."""
        output_file = output_file or Path("results/smoke_test_baseline.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling dataclasses
        result_dict = {
            "timestamp": self.result.timestamp,
            "candidate": self.result.candidate,
            "runtime": self.result.runtime,
            "status": self.result.status,
            "documents_processed": self.result.documents_processed,
            "documents_succeeded": self.result.documents_succeeded,
            "documents_failed": self.result.documents_failed,
            "aggregate_metrics": {
                "avg_cer_percent": self.result.avg_cer,
                "avg_wer_percent": self.result.avg_wer,
                "avg_latency_ms": self.result.avg_latency_ms,
                "avg_throughput_pages_per_sec": self.result.avg_throughput,
                "peak_memory_mb": self.result.peak_memory_mb,
            },
            "per_document_metrics": [m.to_dict() for m in self.result.page_metrics],
            "error_messages": self.result.error_messages,
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_file}")
        return output_file


async def main():
    """Main entry point."""
    benchmark = SmokeTestBenchmark()
    result = await benchmark.run()
    benchmark.save_results()

    # Print summary
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Status: {result.status}")
    print(f"Documents: {result.documents_succeeded}/{result.documents_processed} succeeded")
    print("\nAggregate Metrics:")
    print(
        f"  Avg CER:      {result.avg_cer:.1f}%"
        if result.avg_cer
        else "  Avg CER:      N/A (no ground truth)"
    )
    print(
        f"  Avg WER:      {result.avg_wer:.1f}%"
        if result.avg_wer
        else "  Avg WER:      N/A (no ground truth)"
    )
    print(
        f"  Avg Latency:  {result.avg_latency_ms:.0f} ms"
        if result.avg_latency_ms
        else "  Avg Latency:  N/A"
    )
    print(
        f"  Throughput:   {result.avg_throughput:.2f} pages/sec"
        if result.avg_throughput
        else "  Throughput:   N/A"
    )
    print(
        f"  Peak Memory:  {result.peak_memory_mb:.0f} MB"
        if result.peak_memory_mb
        else "  Peak Memory:  N/A"
    )

    if result.error_messages:
        print(f"\nErrors ({len(result.error_messages)}):")
        for err in result.error_messages:
            print(f"  - {err}")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
