"""CLI for the supply-chain assistant.

    python -m assistant "What is the on-time rate for shipments from Shanghai to Rotterdam?"
    python -m assistant            # interactive session (follow-up questions keep context)

Needs the API running (ASSISTANT_API_URL, default http://localhost:8000) and GEMINI_API_KEY
(read from the environment or a local, git-ignored .env file).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from assistant.agent import Answer, Assistant, AssistantConfig, make_gemini_chat_model
from assistant.tools import ToolBox

log = logging.getLogger("assistant")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no extra dependency). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def model_note(card_path: Path) -> str:
    try:
        card = json.loads(card_path.read_text())
        test = card["results"][card["selected_model"]]["test"]
        ci = card["results"][card["selected_model"]].get("test_auc_95ci")
        return (
            f"The delay model ({card['model_version']}) has test ROC AUC {test['roc_auc']} "
            f"(95% CI {ci}), i.e. no better than chance on held-out data. When you report a "
            "prediction, give the probability and risk band from the tool, and say plainly that "
            "the model has very little predictive power, so the figure is close to the base rate."
        )
    except (OSError, KeyError, ValueError):
        return "Report the probability and risk band from the tool; do not over-interpret them."


def build_assistant(config: AssistantConfig, http: httpx.Client | None = None) -> Assistant:
    http = http or httpx.Client(base_url=config.api_url, timeout=30)
    health = http.get("/health").json()
    if health.get("status") != "ok":
        raise RuntimeError(f"API at {config.api_url} is not healthy: {health}")
    ports = http.get("/ports").json()
    chat_model = make_gemini_chat_model(config, os.getenv("GEMINI_API_KEY"))
    return Assistant(
        chat_model,
        ToolBox(http),
        config,
        data_as_of=health["checks"]["data_store"].get("data_as_of"),
        ports=ports,
        model_note=model_note(config.model_card_path),
    )


def render(ans: Answer) -> str:
    tools = ", ".join(f"{t.get('id', '-')}:{t['name']}" for t in ans.tool_calls) or "none"
    footer = (
        f"[tools: {tools} | llm calls: {ans.llm_calls} | tokens in/out: "
        f"{ans.input_tokens}/{ans.output_tokens} | cost: ${ans.cost_usd:.5f} | "
        f"grounded: {ans.grounding.ok} | {ans.latency_s}s]"
    )
    return f"{ans.text}\n\n{footer}"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Ask the supply-chain assistant a question.")
    parser.add_argument("question", nargs="*", help="Question (omit for interactive mode)")
    parser.add_argument("--model", help="Gemini model id (default: $GEMINI_MODEL)")
    parser.add_argument("--api-url", help="API base URL (default: $ASSISTANT_API_URL)")
    args = parser.parse_args(argv)

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 2
    base = AssistantConfig.from_env()
    config = AssistantConfig(
        **{
            **base.__dict__,
            **{k: v for k, v in {"model": args.model, "api_url": args.api_url}.items() if v},
        }
    )
    try:
        assistant = build_assistant(config)
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"Cannot reach a healthy API at {config.api_url}: {exc}", file=sys.stderr)
        return 3
    print(
        f"[session {assistant.session_id} | model {config.model} | log {assistant.logger.path}]",
        file=sys.stderr,
    )

    questions = [" ".join(args.question)] if args.question else None
    try:
        if questions:
            print(render(assistant.ask(questions[0])))
            return 0
        while True:
            try:
                q = input("\nyou> ").strip()
            except EOFError:
                return 0
            if q.lower() in {"", "exit", "quit"}:
                return 0
            print("\n" + render(assistant.ask(q)))
    except ChatGoogleGenerativeAIError as exc:
        print(f"Gemini API error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
