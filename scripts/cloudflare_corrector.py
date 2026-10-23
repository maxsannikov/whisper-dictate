#!/usr/bin/env python3
"""Private, safe Cloudflare Workers AI text correction client."""
import json
import math
import urllib.error
import urllib.request

from cloudflare_asr import CloudflareError, read_credentials

DEFAULT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
TIMEOUT_SECONDS = 90
MAX_RESPONSE_BYTES = 1_000_000
# Cloudflare model card: $0.0509/M input, $0.335/M output; pricing page:
# $0.011/1000 neurons. Used only for a clearly labelled local estimate.
MODEL_NEURON_RATES = {
    DEFAULT_MODEL: (4627.272727, 30454.545455),
    "@cf/zai-org/glm-4.7-flash": (5500.0, 36400.0),
}


def parse_response(payload):
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise CloudflareError("Некорректный ответ Cloudflare-корректора")
    result = payload.get("result")
    text = None
    if isinstance(result, dict):
        text = result.get("response")
        if not isinstance(text, str):
            choices = result.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise CloudflareError("Cloudflare-корректор вернул пустой или некорректный ответ")
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    return text.strip(), input_tokens, output_tokens


def estimate_neurons(input_tokens, output_tokens, model=DEFAULT_MODEL):
    input_rate, output_rate = MODEL_NEURON_RATES.get(model, MODEL_NEURON_RATES[DEFAULT_MODEL])
    return ((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000)


def correct(text, model=DEFAULT_MODEL, temperature=0.0, max_tokens=2048,
            system_prompt="Исправь только ошибки распознавания, грамматику и пунктуацию. Сохрани смысл, язык и стиль. Верни только исправленный текст."):
    account_id, token = read_credentials()
    payload = {"messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": text}],
               "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    request = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CloudflareError("Ответ Cloudflare-корректора слишком велик")
        return parse_response(json.loads(raw.decode("utf-8")))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            message = "Cloudflare-корректор отклонил запрос (401/403)"
        elif exc.code == 429:
            message = "Cloudflare-корректор достиг лимита (429)"
        else:
            message = f"Ошибка Cloudflare-корректора (HTTP {exc.code})"
        raise CloudflareError(message, "http_401_403" if exc.code in (401, 403) else "http_429" if exc.code == 429 else "http_error") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CloudflareError("Сетевая ошибка или тайм-аут Cloudflare-корректора", "timeout" if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError) else "network_error") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise CloudflareError("Cloudflare-корректор вернул некорректный JSON") from None


def token_counts(source, corrected, input_tokens, output_tokens):
    estimated = input_tokens is None or output_tokens is None
    if estimated:
        input_tokens = max(1, math.ceil(len(source) / 4))
        output_tokens = max(1, math.ceil(len(corrected) / 4))
    if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
        estimated = True
        input_tokens = max(1, math.ceil(len(source) / 4))
        output_tokens = max(1, math.ceil(len(corrected) / 4))
    return input_tokens, output_tokens, estimated
