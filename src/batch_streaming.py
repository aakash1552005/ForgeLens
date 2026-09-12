"""
ForgeLens-X — Milestone 9: Mass-Scale Batch Streaming & Ingestion Engine
========================================================================
High-throughput bulk ingestion pipeline designed to process 100 to 1,000+
documents with bounded memory utilization, zero crash vulnerability,
dynamic progress telemetry, and compliance audit ledger export.
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from src.concurrent_engine import run_concurrent_screening


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def stream_document_directory(
    input_dir: str,
    chunk_size: int = 32,
    recursive: bool = False,
) -> Generator[List[str], None, None]:
    """
    Generator yielding bounded chunks of document file paths.
    Avoids loading thousands of paths into a single unbounded list.
    """
    p_dir = Path(input_dir)
    if not p_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pattern = "**/*" if recursive else "*"
    current_chunk = []

    for file_path in p_dir.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            current_chunk.append(str(file_path))
            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []

    if current_chunk:
        yield current_chunk


def process_single_document_safe(
    doc_path: str,
    doc_id: Optional[str] = None,
    face_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Screens an individual document with full fault isolation.
    Guarantees zero crashes on corrupted, truncated, or unreadable files.
    """
    t0 = time.perf_counter()
    document_id = doc_id or f"DOC-{Path(doc_path).stem}"

    try:
        if not os.path.exists(doc_path) or os.path.getsize(doc_path) == 0:
            return {
                "document_id": document_id,
                "filename": os.path.basename(doc_path),
                "filepath": doc_path,
                "status": "FAIL",
                "decision": "CORRUPTED_FILE",
                "risk_score": 100.0,
                "fraud_probability": 1.0,
                "attack_type_guess": "unreadable_zero_byte",
                "suspicious_count": 0,
                "pipeline_latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "error": "Zero-byte or non-existent file",
            }

        rep = run_concurrent_screening(
            doc_input=doc_path,
            face_input=face_path,
            document_id=document_id,
            doc_path_hint=doc_path,
        )

        dec = rep.get("decision", "MANUAL_REVIEW")
        status = "PASS" if dec in ["VERIFIED", "CLEAR_AUTHENTIC"] else "FAIL"

        return {
            "document_id": document_id,
            "filename": os.path.basename(doc_path),
            "filepath": doc_path,
            "status": status,
            "decision": dec,
            "risk_score": float(rep.get("risk_score", 0.0)),
            "fraud_probability": float(rep.get("fraud_probability", 0.0)),
            "attack_type_guess": rep.get("attack_type_guess", "none"),
            "suspicious_count": len(rep.get("suspicious_regions", [])),
            "fields_extracted": len(rep.get("fields", {})),
            "pipeline_latency_ms": float(rep.get("pipeline_latency_ms", 0.0)),
            "full_report": rep,
            "error": None,
        }

    except Exception as e:
        return {
            "document_id": document_id,
            "filename": os.path.basename(doc_path),
            "filepath": doc_path,
            "status": "FAIL",
            "decision": "ENGINE_ERROR",
            "risk_score": 100.0,
            "fraud_probability": 1.0,
            "attack_type_guess": f"error: {str(e)[:30]}",
            "suspicious_count": 0,
            "pipeline_latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": str(e),
        }


def process_bulk_batch(
    file_paths: List[str],
    max_workers: int = 4,
    ledger_csv: Optional[str] = None,
    ledger_jsonl: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    """
    Executes high-throughput concurrent batch screening across a list of file paths.
    Streams progress events and writes dual audit ledgers in real time.
    Supports checkpoint resumption if resume_existing is True.
    """
    import concurrent.futures

    screened_filenames = set()
    csv_mode = "w"

    if resume_existing and ledger_csv and os.path.exists(ledger_csv):
        try:
            with open(ledger_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get("filename")
                    if fn:
                        screened_filenames.add(fn)
            if screened_filenames:
                csv_mode = "a"
        except Exception:
            pass

    # Filter out already screened documents if resuming
    pending_files = [f for f in file_paths if os.path.basename(f) not in screened_filenames]
    total_files = len(pending_files)
    results = []
    t_start = time.perf_counter()

    verified_count = 0
    flagged_count = 0
    total_latency = 0.0

    csv_writer = None
    csv_file = None
    jsonl_file = None

    # Initialize CSV ledger
    if ledger_csv:
        os.makedirs(os.path.dirname(os.path.abspath(ledger_csv)), exist_ok=True)
        csv_file = open(ledger_csv, csv_mode, newline="", encoding="utf-8")
        fieldnames = [
            "document_id", "filename", "status", "decision", "risk_score",
            "fraud_probability", "attack_type_guess", "suspicious_count",
            "pipeline_latency_ms", "error"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_mode == "w":
            csv_writer.writeheader()

    # Initialize JSONL ledger
    if ledger_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(ledger_jsonl)), exist_ok=True)
        jsonl_mode = "a" if resume_existing and os.path.exists(ledger_jsonl) else "w"
        jsonl_file = open(ledger_jsonl, jsonl_mode, encoding="utf-8")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_document_safe, f, f"BATCH-{i+1+len(screened_filenames):04d}"): (i, f)
                for i, f in enumerate(pending_files)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx, fpath = future_to_idx[future]
                res = future.result()
                results.append(res)

                lat = res.get("pipeline_latency_ms", 0.0)
                total_latency += lat

                if res.get("status") == "PASS":
                    verified_count += 1
                else:
                    flagged_count += 1

                # Write to ledgers immediately
                if csv_writer and csv_file:
                    row = {}
                    for k in [
                        "document_id", "filename", "status", "decision", "risk_score",
                        "fraud_probability", "attack_type_guess", "suspicious_count",
                        "pipeline_latency_ms", "error"
                    ]:
                        v = res.get(k)
                        if isinstance(v, str):
                            v = v.replace("\r", " ").replace("\n", " ").strip()
                        row[k] = v
                    csv_writer.writerow(row)
                    csv_file.flush()

                if jsonl_file:
                    record = {
                        "document_id": res.get("document_id"),
                        "filename": res.get("filename"),
                        "timestamp": time.time(),
                        "status": res.get("status"),
                        "decision": res.get("decision"),
                        "risk_score": res.get("risk_score"),
                        "report": res.get("full_report"),
                    }
                    jsonl_file.write(json.dumps(record) + "\n")
                    jsonl_file.flush()

                # Emit progress telemetry event
                if progress_callback:
                    elapsed = time.perf_counter() - t_start
                    processed = len(results)
                    fps = processed / max(0.01, elapsed)
                    remaining = total_files - processed
                    eta_sec = remaining / max(0.01, fps)

                    progress_callback({
                        "processed": processed,
                        "total": total_files,
                        "verified": verified_count,
                        "flagged": flagged_count,
                        "current_fps": round(fps, 1),
                        "elapsed_sec": round(elapsed, 1),
                        "eta_sec": round(eta_sec, 1),
                        "last_filename": res.get("filename"),
                        "last_decision": res.get("decision"),
                    })

    finally:
        if csv_file:
            csv_file.close()
        if jsonl_file:
            jsonl_file.close()

    total_time = time.perf_counter() - t_start
    mean_lat = total_latency / max(1, len(results))
    mean_risk = sum(r.get("risk_score", 0.0) for r in results) / max(1, len(results))

    return {
        "total_screened": len(results),
        "total_verified": verified_count,
        "total_flagged": flagged_count,
        "batch_elapsed_sec": round(total_time, 2),
        "throughput_docs_per_sec": round(len(results) / max(0.01, total_time), 2),
        "mean_latency_ms": round(mean_lat, 1),
        "mean_risk_score": round(mean_risk, 1),
        "ledger_csv": ledger_csv,
        "ledger_jsonl": ledger_jsonl,
        "results": results,
    }


def export_batch_summary_reports(batch_result: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """
    Saves aggregated executive summaries to disk.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    summary_json_path = str(out_p / "batch_summary.json")
    clean_summary = {
        "total_screened": batch_result["total_screened"],
        "total_verified": batch_result["total_verified"],
        "total_flagged": batch_result["total_flagged"],
        "batch_elapsed_sec": batch_result["batch_elapsed_sec"],
        "throughput_docs_per_sec": batch_result["throughput_docs_per_sec"],
        "mean_latency_ms": batch_result["mean_latency_ms"],
        "mean_risk_score": batch_result["mean_risk_score"],
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    summary_csv_path = str(out_p / "batch_ledger.csv")
    if not os.path.exists(summary_csv_path) and "results" in batch_result:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "document_id", "filename", "status", "decision", "risk_score",
                "fraud_probability", "attack_type_guess", "suspicious_count",
                "pipeline_latency_ms"
            ])
            writer.writeheader()
            for r in batch_result["results"]:
                writer.writerow({
                    "document_id": r.get("document_id"),
                    "filename": r.get("filename"),
                    "status": r.get("status"),
                    "decision": r.get("decision"),
                    "risk_score": r.get("risk_score"),
                    "fraud_probability": r.get("fraud_probability"),
                    "attack_type_guess": r.get("attack_type_guess"),
                    "suspicious_count": r.get("suspicious_count"),
                    "pipeline_latency_ms": r.get("pipeline_latency_ms"),
                })

    return summary_csv_path, summary_json_path


def generate_batch_analytics_report(
    batch_result: Dict[str, Any],
    output_md: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive analytics across batch screening results:
    - P50, P90, P99 latency percentiles
    - Risk band distributions (LOW, ELEVATED, HIGH, CRITICAL)
    - Diagnosed attack taxonomy breakdown
    - Formatted Markdown executive audit summary
    """
    results = batch_result.get("results", [])
    total = len(results)

    if total == 0:
        return {"error": "no_results", "total": 0}

    # Latency percentiles
    latencies = [float(r.get("pipeline_latency_ms", 0.0)) for r in results]
    lat_p50 = float(np.percentile(latencies, 50))
    lat_p90 = float(np.percentile(latencies, 90))
    lat_p99 = float(np.percentile(latencies, 99))
    lat_mean = float(np.mean(latencies))

    # Risk score distribution
    risks = [float(r.get("risk_score", 0.0)) for r in results]
    low_count = sum(1 for r in risks if r < 25.0)
    elev_count = sum(1 for r in risks if 25.0 <= r < 50.0)
    high_count = sum(1 for r in risks if 50.0 <= r < 75.0)
    crit_count = sum(1 for r in risks if r >= 75.0)

    # Attack distribution
    attack_counts: Dict[str, int] = {}
    for r in results:
        atk = r.get("attack_type_guess") or "none"
        attack_counts[atk] = attack_counts.get(atk, 0) + 1

    # Format Markdown
    verified = batch_result.get("total_verified", sum(1 for r in results if r.get("status") == "PASS"))
    flagged = batch_result.get("total_flagged", total - verified)
    pass_pct = (verified / max(1, total)) * 100.0
    throughput = batch_result.get("throughput_docs_per_sec", 0.0)

    md_lines = [
        "# ForgeLens-X | High-Volume Batch Screening Executive Audit Report",
        "",
        "## 1. Screening Performance & Throughput",
        f"- **Total Documents Screened**: {total:,}",
        f"- **Authentic Verified**: {verified:,} ({pass_pct:.1f}%)",
        f"- **Fraudulent Flagged**: {flagged:,} ({100.0 - pass_pct:.1f}%)",
        f"- **Processing Throughput**: {throughput:.2f} documents/second",
        "",
        "### Latency Benchmarks (ms)",
        "| Metric | Latency (ms) |",
        "| :--- | :--- |",
        f"| Mean Latency | {lat_mean:.1f} ms |",
        f"| Median (P50) | {lat_p50:.1f} ms |",
        f"| 90th Percentile (P90) | {lat_p90:.1f} ms |",
        f"| 99th Percentile (P99) | {lat_p99:.1f} ms |",
        "",
        "## 2. Risk Distribution Bands",
        "| Risk Tier | Risk Score Range | Document Count | Share (%) |",
        "| :--- | :--- | :--- | :--- |",
        f"| **LOW** | [0.0 - 25.0) | {low_count:,} | {(low_count / total) * 100.0:.1f}% |",
        f"| **ELEVATED** | [25.0 - 50.0) | {elev_count:,} | {(elev_count / total) * 100.0:.1f}% |",
        f"| **HIGH** | [50.0 - 75.0) | {high_count:,} | {(high_count / total) * 100.0:.1f}% |",
        f"| **CRITICAL** | [75.0 - 100.0] | {crit_count:,} | {(crit_count / total) * 100.0:.1f}% |",
        "",
        "## 3. Attack Taxonomy Diagnosed",
        "| Attack Classification | Intercept Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]
    for atk, cnt in sorted(attack_counts.items(), key=lambda x: -x[1]):
        md_lines.append(f"| `{atk}` | {cnt:,} | {(cnt / total) * 100.0:.1f}% |")

    md_text = "\n".join(md_lines) + "\n"

    analytics = {
        "total_screened": total,
        "total_verified": verified,
        "total_flagged": flagged,
        "pass_rate_pct": round(pass_pct, 2),
        "throughput_docs_per_sec": throughput,
        "latency": {
            "mean_ms": round(lat_mean, 1),
            "p50_ms": round(lat_p50, 1),
            "p90_ms": round(lat_p90, 1),
            "p99_ms": round(lat_p99, 1),
        },
        "risk_bands": {
            "low": low_count,
            "elevated": elev_count,
            "high": high_count,
            "critical": crit_count,
        },
        "attack_taxonomy": attack_counts,
        "markdown_report": md_text,
    }

    if output_md:
        os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_text)

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2)

    return analytics
