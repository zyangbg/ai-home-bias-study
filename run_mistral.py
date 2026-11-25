#!/usr/bin/env python3
"""Run Study 1 on Mistral with one independent request per prompt."""

import argparse
from datetime import datetime, timezone
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from run_deepseek import (
    LANGUAGES, RetryableError, build_jobs, describe_http_error, load_questions,
    normalize_api_key, read_log, result_key, save_results,
)


ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MODELS_ENDPOINT = "https://api.mistral.ai/v1/models"
DEFAULT_MODEL = "mistral-large-4-0"
TEMPERATURE = 0.0
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
MODEL_SETTINGS = {
    DEFAULT_MODEL: ("low", 1024, "mistral_output"),
    "mistral-medium-3-5": ("none", 128, "mistral_output_medium_3_5"),
    "mistral-medium-3-5-26-04": ("none", 128, "mistral_output_medium_3_5_26_04"),
    "mistral-small-2603": ("none", 128, "mistral_output_small_4"),
    "mistral-small-4-0-26-03": ("none", 128, "mistral_output_small_4_26_03"),
}


def get_api_key():
    return normalize_api_key(
        os.environ.get("MISTRAL_API_KEY") or getpass("Mistral API key: "),
        provider="Mistral",
    )


def list_models(api_key):
    request = Request(MODELS_ENDPOINT, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=30) as response:
            body = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(describe_http_error(exc, api_key, provider="Mistral")) from exc
    models = body.get("data") if isinstance(body, dict) else body
    if not isinstance(models, list):
        raise RuntimeError("Mistral returned an unexpected model list.")
    supported = set()
    print("Chat model IDs and aliases returned for this API key:")
    for card in sorted(models, key=lambda item: item.get("id", "")):
        if card.get("archived") or (card.get("capabilities") or {}).get("completion_chat") is False:
            continue
        model_id = card.get("id")
        if not model_id:
            continue
        aliases = card.get("aliases") or []
        supported.update(MODEL_SETTINGS.keys() & {model_id, *aliases})
        print("  " + ", ".join([model_id, *aliases]))
    print("Configured for Study 1: " + (", ".join(sorted(supported)) or "(none in the returned list)"))


def parse_completion(body):
    if not body.get("choices"):
        raise RetryableError("Mistral returned an empty choices list.")
    choice = body["choices"][0]
    message = choice["message"]
    reason = choice.get("finish_reason") or ""
    if message.get("tool_calls") or reason == "tool_calls":
        raise RuntimeError("Unexpected Mistral tool call. The run was stopped.")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            chunk.get("text", "") for chunk in content if chunk.get("type") == "text"
        )
    refusal = message.get("refusal")
    answer = content or refusal or ""
    status = "ok"
    if reason in {"content_filter", "refusal", "model_content_filter"} or refusal:
        status = "refusal"
    elif reason in {"length", "model_length"}:
        status = "truncated"
    elif reason == "error":
        raise RetryableError("Mistral returned an error finish reason.")
    elif not answer.strip():
        status = "empty"
    return {
        "status": status,
        "answer": answer,
        "finish_reason": reason,
        "request_id": body.get("id", ""),
        "returned_model": body.get("model", ""),
    }


def ask_mistral(api_key, prompt, settings):
    payload = {
        "model": settings["requested_model"],
        "messages": [{"role": "user", "content": prompt}],
        "tool_choice": "none",
        "reasoning_effort": settings["reasoning_effort"],
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
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
    for attempt in range(4):
        delay = 2 ** attempt
        try:
            with urlopen(request, timeout=90) as response:
                return parse_completion(json.load(response))
        except HTTPError as exc:
            retry_after = exc.headers.get("Retry-After", "")
            error = describe_http_error(exc, api_key, provider="Mistral")
            if exc.code == 403 and "subscription tier" in error.lower():
                raise RuntimeError(f"{error} Run with --list-models to check your workspace.") from exc
            if exc.code not in RETRYABLE_HTTP:
                raise RuntimeError(error) from exc
            if retry_after.isdigit():
                delay = max(delay, int(retry_after))
        except (URLError, TimeoutError, json.JSONDecodeError, RetryableError) as exc:
            error = str(exc).replace(api_key, "[redacted]")
        if attempt < 3:
            time.sleep(delay)
    return {"status": "api_error", "error": error}


def run(out, jobs, settings):
    out.mkdir(parents=True, exist_ok=True)
    log = out / "mistral_log.jsonl"
    csv_path = out / "mistral_results.csv"
    results = read_log(log, settings)
    pending = []
    for question, language, repetition in jobs:
        key = question["item_id"], language, repetition
        previous = results.get(key)
        if previous is None or previous["status"] == "api_error":
            pending.append((question, language, repetition))

    print(f"{len(jobs)} planned, {len(pending)} pending.", flush=True)
    try:
        if pending:
            api_key = get_api_key()
            with log.open("a", encoding="utf-8") as f:
                for index, (question, language, repetition) in enumerate(pending, 1):
                    record = {field: question[field] for field in ("item_id", "country", "template", "panel")}
                    record.update(
                        language=language, repetition=repetition, requested_model=settings["requested_model"],
                        temperature=settings["temperature"], started_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    record.update(ask_mistral(api_key, question[f"prompt_{language}"], settings))
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    results[result_key(record)] = record
                    print(f"[{index}/{len(pending)}] {question['item_id']} {language} r{repetition}: {record['status']}", flush=True)
    finally:
        save_results(csv_path, results)
        print(f"Results: {csv_path}")
    errors = sum(results[(q["item_id"], lang, rep)]["status"] == "api_error" for q, lang, rep in jobs)
    if errors:
        print(f"{errors} API errors remain. Rerun the same command to retry them.")
    return int(errors > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.csv"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to a separate directory for each model.")
    parser.add_argument("--model", choices=MODEL_SETTINGS, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="Run the first N questions in all three languages.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate the input without an API request.")
    mode.add_argument("--list-models", action="store_true", help="Query the models available to your workspace.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    try:
        if args.list_models:
            list_models(get_api_key())
            return 0
        data = args.questions.read_bytes()
        rows = load_questions(data)[:args.limit]
        jobs = list(build_jobs(rows, repeats=1))
        effort, max_tokens, default_output = MODEL_SETTINGS[args.model]
        settings = {
            "questions_sha256": hashlib.sha256(data).hexdigest(),
            "requested_model": args.model,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "reasoning_effort": effort,
            "tool_choice": "none",
            "endpoint": ENDPOINT,
        }
        if args.dry_run:
            print(f"{len(rows)} items x {len(LANGUAGES)} languages = {len(jobs)} requests for {args.model}.")
            return 0
        return run(args.output_dir or Path(default_output), jobs, settings)
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to resume.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
