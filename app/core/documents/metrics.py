"""
OCR Metrics Calculation

Implements Character Error Rate (CER) and Word Error Rate (WER) calculation,
along with comparison logic for OCR output vs ground truth.

CER (Character Error Rate):
    - Ratio of character-level edits (insertions, deletions, substitutions)
    - Formula: (I + D + S) / N * 100 (percent)
    - Range: 0-100%, lower is better

WER (Word Error Rate):
    - Ratio of word-level edits
    - Formula: (I + D + S) / N * 100 (percent)
    - Range: 0-100%, lower is better

Ground Truth Format:
    JSON file with expected OCR output text and regional annotations.

Example:
    {
        "full_text": "Expected transcription...",
        "text_regions": [
            {"text": "Question 1", "confidence_expected": 0.90},
            ...
        ]
    }
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EditDistanceResult:
    """Result of Levenshtein distance calculation."""

    insertions: int
    deletions: int
    substitutions: int
    total_edits: int
    reference_length: int
    error_rate: float  # percent


def levenshtein_distance(s1: str, s2: str) -> EditDistanceResult:
    """
    Calculate Levenshtein distance between two strings.

    Args:
        s1: Reference (ground truth) string
        s2: Hypothesis (OCR output) string

    Returns:
        EditDistanceResult with breakdown of edit operations
    """
    len1, len2 = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrack to count operations
    insertions = deletions = substitutions = 0
    i, j = len1, len2

    while i > 0 or j > 0:
        if i == 0:
            insertions += 1
            j -= 1
        elif j == 0:
            deletions += 1
            i -= 1
        elif s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        else:
            # Determine which operation was used
            if dp[i - 1][j - 1] <= dp[i - 1][j] and dp[i - 1][j - 1] <= dp[i][j - 1]:
                substitutions += 1
            elif dp[i - 1][j] < dp[i][j - 1]:
                deletions += 1
            else:
                insertions += 1
            i -= 1
            j -= 1

    total_edits = insertions + deletions + substitutions
    reference_length = max(len1, 1)  # Avoid division by zero
    error_rate = (total_edits / reference_length) * 100.0

    return EditDistanceResult(
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        total_edits=total_edits,
        reference_length=reference_length,
        error_rate=error_rate,
    )


def calculate_cer(reference: str, hypothesis: str) -> tuple[float, EditDistanceResult]:
    """
    Calculate Character Error Rate (CER).

    CER = (I + D + S) / N * 100
    where I = insertions, D = deletions, S = substitutions, N = reference length

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        Tuple of (CER percent, EditDistanceResult with breakdown)

    Example:
        >>> cer, result = calculate_cer("hello", "helo")
        >>> print(f"CER: {cer:.2f}%")
        CER: 20.00%
    """
    result = levenshtein_distance(reference, hypothesis)
    return result.error_rate, result


def calculate_wer(reference: str, hypothesis: str) -> tuple[float, EditDistanceResult]:
    """
    Calculate Word Error Rate (WER).

    WER = (I + D + S) / N * 100
    where I, D, S are word-level operations

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        Tuple of (WER percent, EditDistanceResult with word-level breakdown)

    Example:
        >>> wer, result = calculate_wer("hello world", "helo world")
        >>> print(f"WER: {wer:.2f}%")
        WER: 50.00%  # One word wrong (helo vs hello)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    result = levenshtein_distance(" ".join(ref_words), " ".join(hyp_words))
    return result.error_rate, result


class GroundTruthLoader:
    """Load and parse ground truth JSON files."""

    @staticmethod
    def load_ground_truth(gt_path: Path) -> Optional[dict]:
        """
        Load ground truth from JSON file.

        Args:
            gt_path: Path to ground truth JSON file

        Returns:
            Dictionary with ground truth data, or None if file not found

        Raises:
            json.JSONDecodeError: If JSON is malformed
        """
        if not gt_path.exists():
            logger.warning(f"Ground truth file not found: {gt_path}")
            return None

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded ground truth: {gt_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ground truth JSON {gt_path}: {e}")
            raise

    @staticmethod
    def extract_full_text(ground_truth: dict) -> str:
        """
        Extract full expected text from ground truth data.

        Args:
            ground_truth: Parsed ground truth dictionary

        Returns:
            Full text string (empty if not found)
        """
        if "ground_truth" in ground_truth:
            gt_section = ground_truth["ground_truth"]
            if isinstance(gt_section, dict) and "full_text" in gt_section:
                return gt_section["full_text"]

        return ""

    @staticmethod
    def extract_text_regions(ground_truth: dict) -> list[dict]:
        """
        Extract text regions from ground truth data.

        Useful for fine-grained accuracy analysis (e.g., heading accuracy vs body text).

        Args:
            ground_truth: Parsed ground truth dictionary

        Returns:
            List of text region dictionaries
        """
        regions = []

        if "ground_truth" in ground_truth:
            gt_section = ground_truth["ground_truth"]

            # Handle single-page and multi-page formats
            if "text_regions" in gt_section and isinstance(gt_section["text_regions"], list):
                regions = gt_section["text_regions"]
            elif "pages" in gt_section and isinstance(gt_section["pages"], list):
                for page in gt_section["pages"]:
                    if "text_regions" in page:
                        regions.extend(page["text_regions"])

        return regions


@dataclass
class ComparisonResult:
    """Result of comparing OCR output to ground truth."""

    document_name: str
    character_error_rate: float
    word_error_rate: float
    total_characters: int
    total_words: int
    cer_breakdown: EditDistanceResult
    wer_breakdown: EditDistanceResult
    ground_truth_available: bool = True


def compare_ocr_to_ground_truth(
    ocr_text: str, gt_path: Path, document_name: str = ""
) -> ComparisonResult:
    """
    Compare OCR output to ground truth and calculate metrics.

    Args:
        ocr_text: Text output from OCR engine
        gt_path: Path to ground truth JSON file
        document_name: Name of document (for logging)

    Returns:
        ComparisonResult with CER, WER, and breakdown

    Raises:
        FileNotFoundError: If ground truth file not found
        json.JSONDecodeError: If ground truth JSON is malformed
    """
    gt_data = GroundTruthLoader.load_ground_truth(gt_path)

    if gt_data is None:
        logger.warning(f"No ground truth for {document_name}")
        return ComparisonResult(
            document_name=document_name,
            character_error_rate=-1.0,
            word_error_rate=-1.0,
            total_characters=0,
            total_words=0,
            cer_breakdown=EditDistanceResult(0, 0, 0, 0, 0, -1.0),
            wer_breakdown=EditDistanceResult(0, 0, 0, 0, 0, -1.0),
            ground_truth_available=False,
        )

    gt_text = GroundTruthLoader.extract_full_text(gt_data)

    if not gt_text:
        logger.warning(f"Empty ground truth text for {document_name}")
        return ComparisonResult(
            document_name=document_name,
            character_error_rate=-1.0,
            word_error_rate=-1.0,
            total_characters=0,
            total_words=0,
            cer_breakdown=EditDistanceResult(0, 0, 0, 0, 0, -1.0),
            wer_breakdown=EditDistanceResult(0, 0, 0, 0, 0, -1.0),
            ground_truth_available=False,
        )

    cer, cer_breakdown = calculate_cer(gt_text, ocr_text)
    wer, wer_breakdown = calculate_wer(gt_text, ocr_text)

    return ComparisonResult(
        document_name=document_name,
        character_error_rate=cer,
        word_error_rate=wer,
        total_characters=len(gt_text),
        total_words=len(gt_text.split()),
        cer_breakdown=cer_breakdown,
        wer_breakdown=wer_breakdown,
        ground_truth_available=True,
    )


# Example usage and tests
if __name__ == "__main__":
    # Test CER
    print("Testing CER calculation...")
    cer, result = calculate_cer("hello world", "helo world")
    print(f"CER: {cer:.2f}%")
    print(
        f"  Insertions: {result.insertions}, Deletions: {result.deletions}, Substitutions: {result.substitutions}"
    )

    # Test WER
    print("\nTesting WER calculation...")
    wer, result = calculate_wer("The quick brown fox", "The quik brown fox")
    print(f"WER: {wer:.2f}%")
    print(
        f"  Insertions: {result.insertions}, Deletions: {result.deletions}, Substitutions: {result.substitutions}"
    )
