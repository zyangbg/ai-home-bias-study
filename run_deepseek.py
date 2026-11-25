#!/usr/bin/env python3
"""Run the Study 1 prompts through the official DeepSeek API, one prompt at a time."""

import argparse
import csv
from datetime import datetime, timezone
from getpass import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = "https://api.deepseek.com/chat/completions"
LANGUAGES = ("zh", "en", "sv")
MAX_TOKENS = 128
ORDER_SEED = 20261125
API_KEY_EDGES = " \t\r\n\v\f\u00a0\ufeff\u200b\x16\"'\u2018\u2019\u201c\u201d"
RESULT_FIELDS = (
    "item_id", "country", "template", "panel", "language", "repetition",
    "status", "answer", "finish_reason", "request_id", "requested_model",
    "returned_model", "temperature", "started_utc", "error",
    "reused_from",
)


class RetryableError(RuntimeError):
    pass


def load_questions(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    required = {"item_id", "country", "template", "panel"}
    required.update(f"prompt_{language}" for language in LANGUAGES)
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Question CSV is missing one or more required columns.")
    rows = list(reader)
    if not rows or any(not (row.get(key) or "").strip() for row in rows for key in required):
        raise ValueError("Question CSV is empty or has a blank required value.")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Question CSV contains duplicate item IDs.")
    return rows


def build_jobs(rows, repeats):
    for repetition in range(1, repeats + 1):
        jobs = [(row, language) for row in rows for language in LANGUAGES]
        random.Random(ORDER_SEED + repetition).shuffle(jobs)
        for row, language in jobs:
            yield row, language, repetition


def describe_http_error(exc, api_key, provider="DeepSeek"):
    content_type = exc.headers.get_content_type()
    request_id = exc.headers.get("x-request-id") or exc.headers.get("request-id")
    with exc:
        raw = exc.read(4096).decode("utf-8", errors="replace").strip()
    if content_type == "text/html":
        title = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
        detail = f"HTML error page ({title.group(1)})" if title else "HTML error page"
    else:
        detail = raw
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error.get("code") or detail
            elif isinstance(error, str):
                detail = error
            else:
                detail = body.get("message") or body.get("detail") or detail

    detail = " ".join(str(detail).replace(api_key, "[redacted]").split())[:400]
    detail = detail or "Empty response body"
    request_ref = f", request ID {request_id}" if request_id else ""
    return f"{provider} HTTP {exc.code} ({content_type}{request_ref}). {detail}"


def normalize_api_key(value, provider="DeepSeek"):
    key = value.strip(API_KEY_EDGES)
    # Some Windows terminals pass paste markers through to a masked prompt.
    key = key.removeprefix("\x1b[200~").removesuffix("\x1b[201~").strip(API_KEY_EDGES)
    if key.lower().startswith("bearer "):
        key = key[7:].strip(API_KEY_EDGES)
    if not key:
        raise ValueError(f"A {provider} API key is required.")
    if not key.isascii() or any(ch.isspace() or not ch.isprintable() for ch in key):
        raise ValueError(f"The input has extra text or characters inside it. Copy the key directly from {provider}.")
    return key


def ask_deepseek(api_key, model, temperature, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                body = json.load(response)
            if not body.get("choices"):
                raise RetryableError("API returned an empty choices list.")
            choice = body["choices"][0]
            message = choice["message"]
            if message.get("tool_calls") or message.get("server_tool_calls"):
                raise RuntimeError("Unexpected tool use. The run was stopped.")
            reason = choice.get("finish_reason") or ""
            refusal = message.get("refusal")
            answer = message.get("content") or refusal or ""
            status = "ok"
            if reason in {"content_filter", "refusal"} or refusal:
                status = "refusal"
            elif reason == "tool_calls" or body.get("server_tool_use"):
                raise RuntimeError("Unexpected tool use. The run was stopped.")
            elif reason == "length":
                status = "truncated"
            elif not answer.strip():
                status = "empty"
            return {
                "status": status, "answer": answer, "finish_reason": reason,
                "request_id": body.get("id", ""), "returned_model": body.get("model", ""),
            }
        except HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(describe_http_error(exc, api_key)) from exc
            error = f"DeepSeek HTTP {exc.code}"
        except (URLError, TimeoutError, json.JSONDecodeError, RetryableError) as exc:
            error = str(exc)
        if attempt < 2:
            time.sleep(2 ** attempt)
    return {"status": "api_error", "error": error}


def result_key(record):
    return record["item_id"], record["language"], record["repetition"]


def read_log(log, settings):
    if not log.exists():
        with log.open("x", encoding="utf-8") as f:
            f.write(json.dumps({"settings": settings}) + "\n")
        return {}
    with log.open(encoding="utf-8") as f:
        if json.loads(f.readline()).get("settings") != settings:
            raise ValueError("This output directory uses different prompts or model settings. Choose a new --output-dir.")
        results = {}
        for line in f:
            record = json.loads(line)
            results[result_key(record)] = record
        return results


def csv_cell(value):
    # Keep model output unchanged in JSONL and neutralize Excel formulas in CSV.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def save_results(path, results):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for key in sorted(results):
            writer.writerow({k: csv_cell(v) for k, v in results[key].items()})
    temporary.replace(path)


def plan_run(args):
    data = args.questions.read_bytes()
    rows = load_questions(data)
    if args.country:
        rows = [row for row in rows if row["country"] == args.country]
        if not rows:
            raise ValueError(f"No questions for country {args.country}.")
    if args.limit:
        rows = rows[:args.limit]
    jobs = list(build_jobs(rows, args.repeats))
    settings = {
        "questions_sha256": hashlib.sha256(data).hexdigest(),
        "requested_model": args.model,
        "temperature": args.temperature,
        "max_tokens": MAX_TOKENS,
        "thinking": "disabled",
        "tool_choice": "none",
        "endpoint": ENDPOINT,
    }
    return rows, jobs, settings


def reusable_records(directory, jobs, settings):
    data = (directory / "questions.csv").read_bytes()
    old_questions = {row["item_id"]: row for row in load_questions(data)}
    old_settings = {**settings, "questions_sha256": hashlib.sha256(data).hexdigest()}
    old_log = directory / "deepseek_output" / "deepseek_log.jsonl"
    if not old_log.is_file():
        raise ValueError(f"Source log not found: {old_log}")
    old_results = read_log(old_log, old_settings)
    reusable = []
    for row, language, repetition in jobs:
        item_id = row["item_id"]
        previous_question = old_questions.get(item_id)
        previous = old_results.get((item_id, language, repetition))
        fields = ("country", "template", f"prompt_{language}")
        if previous_question is None or any(row[field] != previous_question[field] for field in fields):
            continue
        if previous is None or previous["status"] == "api_error":
            continue
        reusable.append({**previous, "panel": row["panel"], "reused_from": "previous_50_item_run"})
    return reusable


def run(args):
    _, jobs, settings = plan_run(args)
    reusable = reusable_records(args.reuse_from, jobs, settings) if args.reuse_from else []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "deepseek_log.jsonl"
    output = args.output_dir / "deepseek_results.csv"
    results = read_log(log, settings)
    imported = 0
    if reusable:
        with log.open("a", encoding="utf-8") as f:
            for record in reusable:
                key = result_key(record)
                previous = results.get(key)
                if previous is not None and previous["status"] != "api_error":
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                results[key] = record
                imported += 1
    pending = []
    for row, language, repetition in jobs:
        key = row["item_id"], language, repetition
        previous = results.get(key)
        if previous is None or previous["status"] == "api_error":
            pending.append((row, language, repetition))

    print(f"{len(jobs)} planned, {imported} imported, {len(pending)} pending.")
    try:
        if pending:
            api_key = normalize_api_key(os.environ.get("DEEPSEEK_API_KEY") or getpass("DeepSeek API key: "))
            with log.open("a", encoding="utf-8") as f:
                for index, (row, language, repetition) in enumerate(pending, 1):
                    record = {key: row[key] for key in ("item_id", "country", "template", "panel")}
                    record.update(
                        language=language, repetition=repetition, requested_model=args.model,
                        temperature=args.temperature, started_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    record.update(ask_deepseek(api_key, args.model, args.temperature, row[f"prompt_{language}"]))
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    results[result_key(record)] = record
                    print(f"[{index}/{len(pending)}] {row['item_id']} {language} r{repetition}: {record['status']}", flush=True)
    finally:
        save_results(output, results)
        print(f"Results: {output}")
    errors = sum(results[(row["item_id"], lang, rep)]["status"] == "api_error" for row, lang, rep in jobs)
    if errors:
        print(f"{errors} API errors remain. Rerun the same command to retry them.")
    return int(errors > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("deepseek_output_full"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--country", choices=("CN", "US", "SE"), help="Run only questions about this country.")
    parser.add_argument("--limit", type=int, help="Run only the first N items, each in all three languages.")
    parser.add_argument("--reuse-from", type=Path, help="Previous run folder containing questions.csv and deepseek_output/deepseek_log.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the input and print the request count.")
    args = parser.parse_args()
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeats and --limit must be positive.")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be in [0, 2].")
    try:
        if args.dry_run:
            rows, jobs, settings = plan_run(args)
            reusable = reusable_records(args.reuse_from, jobs, settings) if args.reuse_from else []
            print(f"{len(rows)} items x {len(LANGUAGES)} languages x {args.repeats} repetitions = {len(jobs)} requests.")
            if args.reuse_from:
                print(f"{len(reusable)} eligible from the previous run. {len(jobs) - len(reusable)} new requests before accounting for any completed requests in the new output folder.")
            return 0
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to resume.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
