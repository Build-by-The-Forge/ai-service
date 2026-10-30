# The extraction-deadline race (Phase 3 closeout)

**Status: reproduced, root-caused, corrected, regression-tested and stress-tested.** A concurrency race cannot be proven absent by repetition. The evidence and its limits are below. Reopen this if the condition at the end occurs.

## The failure

`tests/unit/test_extraction.py::test_extraction_has_one_deadline` failed once on 2026-10-26, in a full run (1 failed, 966 passed, 10 skipped) with an `OSError` instead of `EXTRACTION_TIMEOUT`. That run's output was filtered, so its full traceback was not kept. The test passed 5 of 5 when run alone.

## The theory that was wrong

The first guess was a temporary-directory cleanup error hiding the timeout. A probe replaced `rmtree` with a function that reported a failure, and `EXTRACTION_TIMEOUT` still surfaced, so the guess was set aside and the issue stayed open as unexplained.

The guess was right and the probe was wrong. It called the cleanup's error handler outside an `except` block, so the handler's bare `raise` re-raised the cancellation already in flight rather than an `OSError`. A real failure raises from inside `rmtree`.

## Reproduction

A loop over the test's body, keeping the full traceback of anything other than `EXTRACTION_TIMEOUT`:

| Run, before the fix | Iterations | `OSError` |
|---|---|---|
| first | 300 | 1 |
| three more | 900 | 0 |
| larger | 3,000 | 2 |
| **total** | **4,200** | **3** (about 1 in 1,400) |

## Root cause

1. The extraction deadline cancels `_pdf` while `asyncio.to_thread(self._read_page, ...)` is rendering a page image into the temporary directory. Cancelling the coroutine does not stop the thread.
2. `TemporaryDirectory.__exit__` runs at once. If the thread creates its file after `rmtree` has listed the directory, `rmdir` fails with `ENOTEMPTY`.
3. That `OSError` replaces the `CancelledError`, so `asyncio.timeout` has nothing to convert to `TimeoutError`.

## Impact before the fix

- A timed-out job reported an unclassified `OSError`, not `EXTRACTION_TIMEOUT`.
- The temporary directory and its image were left behind.
- `with pdf:` could close the MuPDF document while the thread was still rendering from it, a use-after-close in native code. No crash from this was observed.

Only timeouts that land during a page read were affected. The chance per timeout is low, and it is not the chance per document.

## The fix

- **The race (`_pdf_page`):** on cancellation, wait for the in-flight page read before leaving the document and directory contexts. The configured deadline is unchanged; the error can arrive later by at most one page read or render.
- **Exception precedence (`_pdf`):** a cleanup `OSError` during cancellation is logged and the cancellation continues. This still matters if a second cancellation interrupts the wait. Outside cancellation, the `OSError` still propagates.
- **OCR calls (`PaddleOCRService`):** see below.

## Evidence

Deterministic tests in `tests/unit/test_extraction.py`:

- a page read still running at the deadline: the extraction ends with `EXTRACTION_TIMEOUT` only after the read finished, and no directory remains. This test fails without the fix;
- the real failure ordering, with a file created during `rmtree` so that `rmdir` genuinely fails: `EXTRACTION_TIMEOUT` is reported and the failure is logged;
- the same failure without cancellation still raises `OSError`;
- an outside cancellation (a fenced task) also waits for the read.

Stress with the fix: **6,000 iterations, 6,000 `EXTRACTION_TIMEOUT`**. The final 3,000 ran with the cleanup guard in place and logged **zero** cleanup failures, so the guard was not hiding the race.

The limit of that evidence: at the measured rate (3 in 4,200), 6,000 clean iterations would occur by chance about 1.4% of the time with no fix, assuming independent runs. The stress runs use a fake OCR engine, so they exercise the extraction path, not the OCR changes below.

## Abandoned OCR calls

Python cannot stop a running thread, so an OCR call cancelled by the deadline runs until it ends. `tests/unit/test_paddle_abandoned_call.py` covers each guarantee, using a fake engine:

| Guarantee | How |
|---|---|
| The image stays available while OCR may read it | Each call works on its own link or copy, removed when the call ends |
| No shared state changes after the deadline | `_recognize` only returns a value; the model is used by one thread at a time |
| The late result is discarded | Nothing awaits the cancelled call |
| Repeated timeouts cannot accumulate threads | OCR has its own one-thread executor; queued calls are cancelled before they start |
| A late end is observable | Abandonment, late completion and late failure are logged |

**Remaining limitation:** a native OCR call that hangs forever keeps the one OCR thread until the process ends. Later OCR then waits and reaches its own deadline (`EXTRACTION_TIMEOUT`), and fetching, rendering, embedding and native-text extraction carry on. `EXTRACTION_TIMEOUT` is a permanent failure, so scanned documents caught behind a hung call fail and are not retried automatically; they need a reprocess once the worker has been restarted, which clears the hang. No metric or alert covers this yet; only the logs show it. Stronger isolation (OCR in a separate process that can be killed) is a follow-up, not part of this fix.

## When to reopen

Reopen if any of these occurs:

- `test_extraction_has_one_deadline` or its neighbours fail again;
- a job fails with an unclassified `OSError` from extraction;
- the log line `Page directory not removed after cancellation` appears, since the wait should prevent that outside a second cancellation;
- `pages-` or `ocr-` directories accumulate in a worker's temporary directory.

To investigate, run the test's body in a loop, keep every full traceback, and capture the evidence before changing anything.
