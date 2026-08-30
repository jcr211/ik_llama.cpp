#!/usr/bin/env python3
"""Drive a single-slot server from a repetitive 64K-token seed past 100K tokens.

This is a coordinator-run GPU soak. It is intentionally not part of the automated test suite.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


STANZA = (
    "Archive cycle alpha beta gamma delta epsilon zeta eta theta records the same stable sequence. "
    "Archive cycle alpha beta gamma delta epsilon zeta eta theta records the same stable sequence.\n"
)


def post_json(base_url, path, payload, timeout):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(base_url, text, timeout):
    response = post_json(base_url, "/tokenize", {"content": text, "add_special": True}, timeout)
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {response!r}")
    return len(tokens)


def seed_prompt(base_url, wanted_tokens, timeout):
    low, high = 1, 1
    while token_count(base_url, STANZA * high, timeout) < wanted_tokens:
        low = high + 1
        high *= 2

    while low < high:
        middle = (low + high) // 2
        if token_count(base_url, STANZA * middle, timeout) < wanted_tokens:
            low = middle + 1
        else:
            high = middle

    prompt = STANZA * low
    return prompt, token_count(base_url, prompt, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--seed-tokens", type=int, default=66000)
    parser.add_argument("--target-tokens", type=int, default=105000)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.target_tokens <= args.seed_tokens:
        parser.error("--target-tokens must be greater than --seed-tokens")
    if args.chunk <= 0:
        parser.error("--chunk must be positive")

    try:
        prompt, prompt_tokens = seed_prompt(args.base_url, args.seed_tokens, args.timeout)
        print(json.dumps({"event": "seed", "prompt_tokens": prompt_tokens}), flush=True)

        iteration = 0
        while prompt_tokens < args.target_tokens:
            iteration += 1
            result = post_json(
                args.base_url,
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": min(args.chunk, args.target_tokens - prompt_tokens),
                    "temperature": 0.0,
                    "top_k": 1,
                    "ignore_eos": True,
                    "cache_prompt": True,
                    "id_slot": 0,
                    "stream": False,
                },
                args.timeout,
            )
            content = result.get("content")
            predicted = int(result.get("tokens_predicted", 0))
            if not isinstance(content, str) or predicted <= 0:
                raise RuntimeError(f"completion made no progress: {result!r}")

            prompt += content
            prompt_tokens = token_count(args.base_url, prompt, args.timeout)
            print(
                json.dumps(
                    {
                        "event": "completion",
                        "iteration": iteration,
                        "tokens_evaluated": result.get("tokens_evaluated"),
                        "tokens_predicted": predicted,
                        "retokenized_prompt_tokens": prompt_tokens,
                        "stopped_limit": result.get("stopped_limit"),
                    }
                ),
                flush=True,
            )

        print(json.dumps({"event": "complete", "prompt_tokens": prompt_tokens}), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as error:
        print(f"soak failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
