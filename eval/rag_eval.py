from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class EvalSample:
    sample_id: str
    query: str
    expected_answer: str | None
    expected_keywords: list[str]
    expected_files: list[str]
    expected_chunk_ids: list[str]


@dataclass
class SampleResult:
    sample_id: str
    query: str
    status: str
    error: str | None
    latencies_ms: list[float]
    api_execution_time_ms: float | None
    retrieved_count: int

    # Retrieval — file granularity (rank-aware).
    file_hit_at_k: int | None = None
    file_recall_at_k: float | None = None
    file_precision_at_k: float | None = None
    file_mrr: float | None = None

    # Retrieval — chunk granularity (rank-aware).
    chunk_hit_at_k: int | None = None
    chunk_recall_at_k: float | None = None
    chunk_precision_at_k: float | None = None
    chunk_mrr: float | None = None

    # Answer — lexical baselines.
    answer_exact_match: int | None = None
    answer_token_f1: float | None = None
    keyword_coverage_answer: float | None = None
    keyword_coverage_context: float | None = None

    # Answer — LLM-as-judge (optional, RAGAS-style 0..1 scores).
    judge_correctness: float | None = None
    judge_faithfulness: float | None = None
    judge_relevance: float | None = None
    judge_rationale: str | None = None


# ── Text utilities ────────────────────────────────────────────────────────────


def normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_present(keyword: str, text: str) -> bool:
    """Word-boundary match so ``api`` does not match inside ``rapid``.

    Multi-word keywords are matched as a phrase with boundaries on each end.
    """
    normalized = normalize_text(keyword)
    if not normalized:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in normalized.split(" ")) + r"\b"
    return re.search(pattern, normalize_text(text)) is not None


# ── Dataset loading ─────────────────────────────────────────────────────────────


def parse_sample(raw: dict[str, Any], index: int) -> EvalSample:
    sample_id = str(raw.get("id") or raw.get("sample_id") or f"sample_{index + 1}")
    query = str(raw.get("query", "")).strip()
    if not query:
        raise ValueError(f"Sample {sample_id} has empty query")

    expected_answer = raw.get("expected_answer")
    if expected_answer is not None:
        expected_answer = str(expected_answer)

    expected_keywords = [str(item).strip() for item in raw.get("expected_keywords", []) if str(item).strip()]
    expected_files = [str(item).strip().lower() for item in raw.get("expected_files", []) if str(item).strip()]
    expected_chunk_ids = [str(item).strip() for item in raw.get("expected_chunk_ids", []) if str(item).strip()]

    return EvalSample(
        sample_id=sample_id,
        query=query,
        expected_answer=expected_answer,
        expected_keywords=expected_keywords,
        expected_files=expected_files,
        expected_chunk_ids=expected_chunk_ids,
    )


def load_samples(dataset_path: Path) -> list[EvalSample]:
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    raw_items: list[dict[str, Any]] = []

    if suffix == ".jsonl":
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_no}: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_no} must be a JSON object")
            raw_items.append(item)
    elif suffix == ".json":
        parsed = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise ValueError("JSON dataset must contain a top-level array")
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"JSON dataset item at index {idx} must be an object")
            raw_items.append(item)
    else:
        raise ValueError("Dataset must be .json or .jsonl")

    if not raw_items:
        raise ValueError("Dataset is empty")

    return [parse_sample(item, idx) for idx, item in enumerate(raw_items)]


# ── Retrieval metrics (rank-aware) ───────────────────────────────────────────────


def _rank_metrics(relevant: set[str], ranked: list[str]) -> dict[str, float | int]:
    """Compute hit@k, recall@k, precision@k and reciprocal rank for one ranked list.

    ``ranked`` is the ordered list of identifiers (file names or chunk ids), one
    entry per retrieved chunk, in the order the API returned them.
    """
    retrieved_relevant = [item for item in ranked if item in relevant]
    found_unique = set(retrieved_relevant)

    mrr = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            mrr = 1.0 / position
            break

    return {
        "hit": 1 if found_unique else 0,
        "recall": len(found_unique) / len(relevant) if relevant else 0.0,
        # Precision over retrieved chunks: fraction of returned chunks that are relevant.
        "precision": len(retrieved_relevant) / len(ranked) if ranked else 0.0,
        "mrr": mrr,
    }


# ── LLM-as-judge (optional) ──────────────────────────────────────────────────────


_JUDGE_SYSTEM = (
    "You are a strict RAG evaluation judge. You are given a user question, the "
    "context that was retrieved for it, the assistant's answer, and an optional "
    "reference. Score the answer on three axes, each a float in [0,1]:\n"
    "- correctness: factually correct with respect to the reference (or, if no "
    "reference, with respect to the context). 1 = fully correct, 0 = wrong.\n"
    "- faithfulness: every claim in the answer is supported by the context "
    "(no hallucination). 1 = fully grounded, 0 = fabricated.\n"
    "- relevance: the answer directly addresses the question. 1 = on point, "
    "0 = off topic.\n"
    'Respond with ONLY a JSON object: {"correctness": <float>, '
    '"faithfulness": <float>, "relevance": <float>, "rationale": "<one sentence>"}.'
)


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _clamp01(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def make_judge(model: str, max_context_chars: int):
    """Return a judge callable, or None if the judge cannot be initialised."""
    try:
        import anthropic
    except ImportError:
        print("  [judge] 'anthropic' not installed — skipping LLM-as-judge.")
        return None

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        print("  [judge] ANTHROPIC_API_KEY not set — skipping LLM-as-judge.")
        return None

    client = anthropic.Anthropic()

    def judge(query: str, answer: str, context: str, reference: str | None) -> dict[str, Any]:
        context = context[:max_context_chars]
        reference_block = f"\n\nReference answer:\n{reference}" if reference else ""
        user = (
            f"Question:\n{query}\n\nRetrieved context:\n{context}{reference_block}"
            f"\n\nAssistant answer:\n{answer}"
        )
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as error:  # noqa: BLE001 - judge failures must not abort the run
            return {"error": str(error)}

        text = next(
            (block.text for block in response.content if getattr(block, "type", None) == "text"),
            "",
        )
        parsed = _extract_json(text)
        if not parsed:
            return {"error": "judge returned unparseable output"}
        return {
            "correctness": _clamp01(parsed.get("correctness")),
            "faithfulness": _clamp01(parsed.get("faithfulness")),
            "relevance": _clamp01(parsed.get("relevance")),
            "rationale": str(parsed.get("rationale", "")).strip() or None,
        }

    return judge


# ── Per-sample evaluation ────────────────────────────────────────────────────────


def _query_once(
    api_base_url: str, query: str, hybrid_alpha: float, timeout_seconds: float
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(
        f"{api_base_url.rstrip('/')}/api/v1/retrieval/query",
        json={"query": query, "hybrid_alpha": hybrid_alpha},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return response.json(), elapsed_ms


def evaluate_sample(
    sample: EvalSample,
    api_base_url: str,
    hybrid_alpha: float,
    timeout_seconds: float,
    repeat: int,
    judge,
    max_context_chars: int,
) -> SampleResult:
    latencies: list[float] = []
    payload: dict[str, Any] = {}
    try:
        for _ in range(max(repeat, 1)):
            payload, elapsed_ms = _query_once(api_base_url, sample.query, hybrid_alpha, timeout_seconds)
            latencies.append(elapsed_ms)
    except Exception as error:  # noqa: BLE001
        return SampleResult(
            sample_id=sample.sample_id,
            query=sample.query,
            status="ERROR",
            error=str(error),
            latencies_ms=latencies,
            api_execution_time_ms=None,
            retrieved_count=0,
        )

    answer = str(payload.get("response", ""))
    chunks = payload.get("retrieved_chunks", [])
    if not isinstance(chunks, list):
        chunks = []

    result = SampleResult(
        sample_id=sample.sample_id,
        query=sample.query,
        status="OK",
        error=None,
        latencies_ms=latencies,
        api_execution_time_ms=(
            float(payload["execution_time_ms"])
            if isinstance(payload.get("execution_time_ms"), (int, float))
            else None
        ),
        retrieved_count=len(chunks),
    )

    ranked_files = [str(c.get("file_name", "")).strip().lower() for c in chunks]
    ranked_chunk_ids = [str(c.get("chunk_id", "")) for c in chunks]
    context = "\n\n".join(str(c.get("chunk_text", "")) for c in chunks)

    # Retrieval metrics, reported separately per granularity.
    if sample.expected_files:
        m = _rank_metrics(set(sample.expected_files), ranked_files)
        result.file_hit_at_k = m["hit"]
        result.file_recall_at_k = m["recall"]
        result.file_precision_at_k = m["precision"]
        result.file_mrr = m["mrr"]

    if sample.expected_chunk_ids:
        m = _rank_metrics(set(sample.expected_chunk_ids), ranked_chunk_ids)
        result.chunk_hit_at_k = m["hit"]
        result.chunk_recall_at_k = m["recall"]
        result.chunk_precision_at_k = m["precision"]
        result.chunk_mrr = m["mrr"]

    # Lexical answer metrics.
    if sample.expected_answer is not None:
        result.answer_exact_match = 1 if normalize_text(answer) == normalize_text(sample.expected_answer) else 0
        result.answer_token_f1 = token_f1_score(answer, sample.expected_answer)

    if sample.expected_keywords:
        total = len(sample.expected_keywords)
        result.keyword_coverage_answer = sum(keyword_present(k, answer) for k in sample.expected_keywords) / total
        result.keyword_coverage_context = sum(keyword_present(k, context) for k in sample.expected_keywords) / total

    # LLM-as-judge (semantic quality).
    if judge is not None:
        reference = sample.expected_answer
        if reference is None and sample.expected_keywords:
            reference = "Expected key facts: " + ", ".join(sample.expected_keywords)
        verdict = judge(sample.query, answer, context, reference)
        if "error" in verdict:
            print(f"  [judge] {sample.sample_id}: {verdict['error']}")
        else:
            result.judge_correctness = verdict.get("correctness")
            result.judge_faithfulness = verdict.get("faithfulness")
            result.judge_relevance = verdict.get("relevance")
            result.judge_rationale = verdict.get("rationale")

    return result


# ── Aggregation ──────────────────────────────────────────────────────────────────


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return sorted_values[low]
    fraction = rank - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def _avg_of(results: list[SampleResult], attr: str) -> float | None:
    values = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
    return average([float(v) for v in values])


def aggregate_results(results: list[SampleResult], wall_clock_s: float) -> dict[str, Any]:
    ok_results = [item for item in results if item.status == "OK"]
    error_results = [item for item in results if item.status == "ERROR"]

    latencies = [ms for item in ok_results for ms in item.latencies_ms]
    total_queries = sum(len(item.latencies_ms) for item in results)

    summary: dict[str, Any] = {
        "total_samples": len(results),
        "ok_samples": len(ok_results),
        "error_samples": len(error_results),
        # Performance.
        "total_queries": total_queries,
        "wall_clock_s": round(wall_clock_s, 3),
        "throughput_qps": round(total_queries / wall_clock_s, 3) if wall_clock_s > 0 else None,
        "latency_ms_avg": average(latencies),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "latency_ms_stdev": statistics.pstdev(latencies) if len(latencies) > 1 else (0.0 if latencies else None),
        "api_execution_time_ms_avg": _avg_of(ok_results, "api_execution_time_ms"),
        # Retrieval — files.
        "file_hit_at_k": _avg_of(ok_results, "file_hit_at_k"),
        "file_recall_at_k": _avg_of(ok_results, "file_recall_at_k"),
        "file_precision_at_k": _avg_of(ok_results, "file_precision_at_k"),
        "file_mrr": _avg_of(ok_results, "file_mrr"),
        # Retrieval — chunks.
        "chunk_hit_at_k": _avg_of(ok_results, "chunk_hit_at_k"),
        "chunk_recall_at_k": _avg_of(ok_results, "chunk_recall_at_k"),
        "chunk_precision_at_k": _avg_of(ok_results, "chunk_precision_at_k"),
        "chunk_mrr": _avg_of(ok_results, "chunk_mrr"),
        # Answer — lexical.
        "answer_exact_match": _avg_of(ok_results, "answer_exact_match"),
        "answer_token_f1": _avg_of(ok_results, "answer_token_f1"),
        "keyword_coverage_answer": _avg_of(ok_results, "keyword_coverage_answer"),
        "keyword_coverage_context": _avg_of(ok_results, "keyword_coverage_context"),
        # Answer — LLM-as-judge.
        "judge_correctness": _avg_of(ok_results, "judge_correctness"),
        "judge_faithfulness": _avg_of(ok_results, "judge_faithfulness"),
        "judge_relevance": _avg_of(ok_results, "judge_relevance"),
    }
    return summary


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_summary(summary: dict[str, Any]) -> None:
    def line(label: str, key: str) -> None:
        print(f"{label:<26}{format_metric(summary[key])}")

    print("\nRAG Evaluation Summary")
    print("=" * 80)
    print(f"{'Total / OK / Error':<26}{summary['total_samples']} / {summary['ok_samples']} / {summary['error_samples']}")

    print("\n-- Performance --")
    print(f"{'Total queries':<26}{summary['total_queries']}")
    print(f"{'Wall clock (s)':<26}{summary['wall_clock_s']}")
    line("Throughput (q/s)", "throughput_qps")
    line("Latency avg (ms)", "latency_ms_avg")
    line("Latency p50 (ms)", "latency_ms_p50")
    line("Latency p95 (ms)", "latency_ms_p95")
    line("Latency p99 (ms)", "latency_ms_p99")
    line("Latency min (ms)", "latency_ms_min")
    line("Latency max (ms)", "latency_ms_max")
    line("API exec avg (ms)", "api_execution_time_ms_avg")

    print("\n-- Retrieval (files) --")
    line("Hit@k", "file_hit_at_k")
    line("Recall@k", "file_recall_at_k")
    line("Precision@k", "file_precision_at_k")
    line("MRR", "file_mrr")

    print("\n-- Retrieval (chunks) --")
    line("Hit@k", "chunk_hit_at_k")
    line("Recall@k", "chunk_recall_at_k")
    line("Precision@k", "chunk_precision_at_k")
    line("MRR", "chunk_mrr")

    print("\n-- Answer quality (lexical) --")
    line("Exact match", "answer_exact_match")
    line("Token F1", "answer_token_f1")
    line("Keyword cov. (answer)", "keyword_coverage_answer")
    line("Keyword cov. (context)", "keyword_coverage_context")

    print("\n-- Answer quality (LLM judge) --")
    line("Correctness", "judge_correctness")
    line("Faithfulness", "judge_faithfulness")
    line("Relevance", "judge_relevance")


def build_report(
    dataset_path: Path,
    args: argparse.Namespace,
    results: list[SampleResult],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset_path.resolve()),
        "config": {
            "api_base_url": args.api_base_url,
            "hybrid_alpha": args.hybrid_alpha,
            "timeout_seconds": args.timeout_seconds,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "concurrency": args.concurrency,
            "judge": bool(args.judge),
            "judge_model": args.judge_model if args.judge else None,
        },
        "summary": summary,
        "results": [asdict(item) for item in results],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality against a dataset.")
    parser.add_argument("--dataset", required=True, help="Path to .json or .jsonl dataset file.")
    parser.add_argument("--api-base-url", default="http://localhost:8080", help="Base URL of the RAG API.")
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Hybrid retrieval alpha (0.0 keyword only, 1.0 vector only).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Per-query request timeout in seconds.")
    parser.add_argument("--output", default="eval/eval_report.json", help="Where to write the JSON report.")
    # Performance controls.
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warmup queries before measuring (excludes cold start: model load + BM25 index build).",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Timed queries per sample (more = stabler latency).")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel workers across samples (throughput test).")
    # LLM-as-judge controls.
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge for correctness/faithfulness/relevance (needs ANTHROPIC_API_KEY).",
    )
    parser.add_argument("--judge-model", default="claude-sonnet-4-6", help="Model id used for LLM-as-judge.")
    parser.add_argument("--judge-context-chars", type=int, default=12000, help="Max context chars sent to the judge.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.hybrid_alpha <= 1.0:
        raise ValueError("--hybrid-alpha must be between 0.0 and 1.0")
    if args.concurrency < 1 or args.repeat < 1 or args.warmup < 0:
        raise ValueError("--concurrency and --repeat must be >= 1; --warmup must be >= 0")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    samples = load_samples(dataset_path)

    print(f"Loaded {len(samples)} samples from {dataset_path}")
    print(f"API: {args.api_base_url} · alpha={args.hybrid_alpha} · "
          f"warmup={args.warmup} · repeat={args.repeat} · concurrency={args.concurrency} · judge={bool(args.judge)}")

    judge = make_judge(args.judge_model, args.judge_context_chars) if args.judge else None

    # Warmup — prime the server so cold-start cost is excluded from latency stats.
    for i in range(args.warmup):
        try:
            _query_once(args.api_base_url, samples[0].query, args.hybrid_alpha, args.timeout_seconds)
            print(f"Warmup {i + 1}/{args.warmup} done")
        except Exception as error:  # noqa: BLE001
            print(f"Warmup {i + 1}/{args.warmup} failed: {error}")

    def run(sample: EvalSample) -> SampleResult:
        return evaluate_sample(
            sample=sample,
            api_base_url=args.api_base_url,
            hybrid_alpha=args.hybrid_alpha,
            timeout_seconds=args.timeout_seconds,
            repeat=args.repeat,
            judge=judge,
            max_context_chars=args.judge_context_chars,
        )

    started = time.perf_counter()
    results: list[SampleResult] = []
    if args.concurrency == 1:
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.sample_id}")
            result = run(sample)
            if result.status == "ERROR":
                print(f"  ERROR: {result.error}")
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run, sample): sample for sample in samples}
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                marker = "ERROR" if result.status == "ERROR" else "ok"
                print(f"[{done}/{len(samples)}] {result.sample_id}: {marker}")
                results.append(result)
        # Restore dataset order for a stable report.
        order = {sample.sample_id: i for i, sample in enumerate(samples)}
        results.sort(key=lambda r: order.get(r.sample_id, 0))
    wall_clock_s = time.perf_counter() - started

    summary = aggregate_results(results, wall_clock_s)
    print_summary(summary)

    report = build_report(dataset_path, args, results, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {output_path}")

    return 0 if summary["error_samples"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
