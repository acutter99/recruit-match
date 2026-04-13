"""LLM integration layer - supports OpenAI and Anthropic APIs."""
import os
import json
import httpx


async def call_llm(prompt: str) -> dict | str | None:
    """Call configured LLM provider. Returns parsed JSON dict, raw string, or None."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        return await _call_openai(prompt, openai_key)
    elif anthropic_key:
        return await _call_anthropic(prompt, anthropic_key)
    else:
        return None  # No LLM configured - baseline scoring only


async def _call_openai(prompt: str, api_key: str) -> dict | str | None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _try_parse_json(content)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None


async def _call_anthropic(prompt: str, api_key: str) -> dict | str | None:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"]
            return _try_parse_json(content)
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None


def _try_parse_json(text: str) -> dict | str:
    """Attempt to parse JSON from LLM response, return raw text if it fails."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
