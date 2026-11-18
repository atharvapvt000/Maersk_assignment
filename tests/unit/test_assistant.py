"""LangGraph assistant, tools and guardrails with a scripted fake chat model (no network, no key).

The compiled graph runs for real. Tools call the FastAPI app (fake repository) through
TestClient, which is an httpx.Client, so the assistant-to-API HTTP contract is exercised too.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from assistant.agent import GEMINI_TOOLS, Assistant, AssistantConfig, build_system_prompt
from assistant.guardrails import OFF_DOMAIN_REPLY, check_grounding
from assistant.tools import ToolArgumentError, ToolBox, validate_args

# ----------------------------------------------------------------------------- fakes

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def call(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"c-{name}", "type": "tool_call"}],
        usage_metadata=USAGE,
    )


def text(t: str) -> AIMessage:
    return AIMessage(content=t, usage_metadata=USAGE)


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AI messages in order and records what it was sent."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@pytest.fixture
def config(tmp_path: Path) -> AssistantConfig:
    return AssistantConfig(log_dir=tmp_path / "logs")


def make(client, config, script) -> tuple[Assistant, ScriptedChatModel]:
    model = ScriptedChatModel(script=list(script), seen=[])
    a = Assistant(
        model,
        ToolBox(client),
        config,
        data_as_of="2025-10-14T23:55:00",
        ports=[{"port_code": "CNSHA", "port_name": "Shanghai", "country": "China"}],
        model_note="Model test ROC AUC 0.5063 (95% CI [0.4705, 0.5434]).",
        session_id="test-session",
    )
    return a, model


def records(a: Assistant) -> list[dict]:
    return [json.loads(line) for line in a.logger.path.read_text().splitlines()]


# ----------------------------------------------------------------------------- happy path + logging


def test_tool_call_then_grounded_answer(client, config):
    a, model = make(
        client,
        config,
        [
            call("get_route_stats", origin="CNSHA", destination="NLRTM"),
            text("On-time rate is 66.7% over 3 arrived shipments [T1]."),
        ],
    )
    ans = a.ask("What is the on-time rate from Shanghai to Rotterdam?")
    assert ans.grounding.ok and ans.grounding.cited_ids == ["T1"]
    assert ans.llm_calls == 2 and [t["name"] for t in ans.tool_calls] == ["get_route_stats"]
    # the tool result went back to the model as a ToolMessage carrying our citation id
    tool_msg = model.seen[1][-1]
    assert tool_msg.type == "tool" and tool_msg.tool_call_id == "c-get_route_stats"
    payload = json.loads(tool_msg.content)
    assert payload["tool_result_id"] == "T1"
    assert payload["data"]["on_time_rate"] == pytest.approx(0.6667)


def test_system_prompt_is_sent_first(client, config):
    a, model = make(client, config, [text(OFF_DOMAIN_REPLY)])
    a.ask("hi")
    assert model.seen[0][0].type == "system" and "SCOPE" in model.seen[0][0].content


def test_every_llm_call_is_logged_with_prompt_tools_and_output(client, config):
    a, _ = make(
        client,
        config,
        [
            call("get_route_stats", origin="CNSHA", destination="NLRTM"),
            text("66.7% [T1]"),
        ],
    )
    a.ask("on-time CNSHA to NLRTM?")
    recs = records(a)
    assert [r["type"] for r in recs] == ["question", "llm_call", "tool_call", "llm_call", "final"]
    first = recs[1]
    sent = first["request"]["messages"]
    assert [m["type"] for m in sent] == ["system", "human"]
    assert sent[1]["data"]["content"] == "on-time CNSHA to NLRTM?"
    assert "get_route_stats" in first["request"]["tools_offered"]
    assert first["response"]["data"]["tool_calls"][0]["name"] == "get_route_stats"
    assert first["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert first["cost_usd"] == pytest.approx((100 * 0.25 + 20 * 1.5) / 1e6)
    assert first["latency_ms"] is not None
    tool = recs[2]
    assert tool["ok"] is True and tool["http_status"] == 200 and tool["response"]["data"]
    # second call includes the tool result in its prompt
    assert [m["type"] for m in recs[3]["request"]["messages"]][-1] == "tool"
    final = recs[-1]
    assert final["answer"] == "66.7% [T1]" and final["guardrails"]["grounding"]["ok"]


def test_gemini_tool_schemas_drop_unsupported_key():
    assert all("additionalProperties" not in t["parameters"] for t in GEMINI_TOOLS)
    assert {t["name"] for t in GEMINI_TOOLS} >= {
        "query_shipments",
        "get_route_stats",
        "predict_delay",
    }


def test_follow_up_question_keeps_context(client, config):
    a, model = make(client, config, [text(OFF_DOMAIN_REPLY), text(OFF_DOMAIN_REPLY)])
    a.ask("first")
    a.ask("second")
    contents = [m.content for m in model.seen[1] if m.type == "human"]
    assert contents == ["first", "second"]


# ----------------------------------------------------------------------------- grounding guardrail


def test_ungrounded_answer_triggers_one_repair(client, config):
    a, model = make(
        client,
        config,
        [
            call("get_route_stats", origin="CNSHA", destination="NLRTM"),
            text("On-time rate is 91% [T1]."),  # invented
            text("On-time rate is 66.67% [T1]."),
        ],
    )
    ans = a.ask("on-time rate CNSHA to NLRTM?")
    assert ans.grounding.ok and ans.text == "On-time rate is 66.67% [T1]."
    assert "GUARDRAIL" in model.seen[2][-1].content
    assert any(r.get("guardrail") == "grounding_repair" for r in records(a))


def test_still_ungrounded_after_repair_is_flagged_not_hidden(client, config):
    a, _ = make(client, config, [text("It is 91% [T1]."), text("It is 91% [T1].")])
    ans = a.ask("on-time rate CNSHA to NLRTM?")
    assert not ans.grounding.ok
    assert ans.stopped_reason == "answered_with_unverified_figures"
    assert "could not verify" in ans.text and "91" in ans.text


def test_off_domain_refusal_needs_no_tools(client, config):
    a, model = make(client, config, [text(OFF_DOMAIN_REPLY)])
    ans = a.ask("What's Apple's share price?")
    assert ans.text == OFF_DOMAIN_REPLY and ans.grounding.ok and not ans.tool_calls
    assert len(model.seen) == 1


def test_gemini_style_content_blocks_are_read(client, config):
    blocks = AIMessage(content=[{"type": "text", "text": OFF_DOMAIN_REPLY}], usage_metadata=USAGE)
    a, _ = make(client, config, [blocks])
    assert a.ask("weather?").text == OFF_DOMAIN_REPLY


# ----------------------------------------------------------------------------- budgets


def test_per_question_tool_cap_blocks_and_unbinds_tools(client, config):
    cfg = replace(config, max_tool_calls_per_question=2)
    a, _ = make(
        client,
        cfg,
        [
            call("get_route_stats", origin="CNSHA", destination="NLRTM"),
            call("get_shipment", shipment_id="SHP-00421"),
            call("predict_delay", shipment_id="SHP-00421"),
            text("Done."),
        ],
    )
    ans = a.ask("q")
    assert [t.get("blocked", False) for t in ans.tool_calls] == [False, False, True]
    llm_calls = [r for r in records(a) if r["type"] == "llm_call"]
    assert llm_calls[3]["request"]["tools_offered"] == []  # tools no longer offered
    last_tool_msg = llm_calls[3]["request"]["messages"][-1]["data"]["content"]
    assert "TOOL_BUDGET_EXHAUSTED" in last_tool_msg
    assert a.session_tool_calls == 2


def test_session_tool_cap_refuses_without_llm_call(client, config):
    a, model = make(client, replace(config, max_tool_calls_per_session=0), [])
    ans = a.ask("anything")
    assert ans.stopped_reason == "session_tool_budget_exhausted" and model.seen == []


def test_llm_call_budget(client, config):
    cfg = replace(config, max_llm_calls_per_question=2)
    a, _ = make(
        client,
        cfg,
        [
            call("get_shipment", shipment_id="SHP-00421"),
            call("get_shipment", shipment_id="SHP-00421"),
        ],
    )
    assert a.ask("q").stopped_reason == "llm_call_budget_exhausted"


def test_empty_model_response(client, config):
    a, _ = make(client, config, [AIMessage(content="", usage_metadata=USAGE)])
    assert a.ask("q").stopped_reason == "empty_response"


# ----------------------------------------------------------------------------- tools


def test_invalid_model_arguments_never_reach_the_api(client):
    box = ToolBox(client)
    r = box.call("get_route_stats", {"origin": "Shanghai", "destination": "NLRTM"})
    assert not r.ok and r.data["code"] == "INVALID_ARGUMENTS" and r.http_status is None
    assert box.call("drop_tables", {}).data["code"] == "INVALID_ARGUMENTS"


def test_api_errors_are_unwrapped_for_the_model(client):
    r = ToolBox(client).call("get_shipment", {"shipment_id": "SHP-99999"})
    assert not r.ok and r.http_status == 404 and r.data["code"] == "SHIPMENT_NOT_FOUND"
    assert r.for_model() == {"tool_result_id": "T1", "error": r.data}


def test_query_shipments_trims_fields(client):
    r = ToolBox(client).call("query_shipments", {"origin": "cnsha", "limit": 5})
    assert r.ok and r.data["total"] == 1
    assert set(r.data["items"][0]) <= {
        "shipment_id",
        "route_key",
        "status",
        "cargo_type",
        "planned_departure",
        "planned_arrival",
        "actual_arrival",
        "actual_delay_hours",
        "on_time_flag",
    }


def test_rank_routes_tool(client):
    r = ToolBox(client).call(
        "rank_routes", {"metric": "avg_delay_hours", "order": "desc", "quarter": "2025-Q3"}
    )
    assert r.ok and r.data["routes"][0]["route_key"] == "CNSHA → NLRTM"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_shipment", {}),
        ("get_shipment", {"shipment_id": "x"}),
        ("query_shipments", {"limit": 500}),
        ("query_shipments", {"status": "LOST"}),
        ("query_shipments", {"departure_from": "01/02/2025"}),
        ("rank_routes", {"metric": "avg_delay_hours", "order": "desc", "evil": 1}),
    ],
)
def test_validate_args_rejects(name, args):
    with pytest.raises(ToolArgumentError):
        validate_args(name, args)


def test_validate_args_normalises():
    assert validate_args("get_route_stats", {"origin": " cnsha", "destination": "nlrtm"}) == {
        "origin": "CNSHA",
        "destination": "NLRTM",
    }
    assert validate_args("query_shipments", {"limit": 5.0}) == {"limit": 5}


# ----------------------------------------------------------------------------- grounding unit


@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        ("Rate is 40.4% [T1].", True),  # 0.404 -> percent
        ("Avg delay 36 hours [T1].", True),  # rounding of 36.04
        ("Avg delay 36.0 hours [T1].", True),
        ("1,234 shipments [T1].", True),  # thousands separator
        ("Avg delay 37 hours [T1].", False),  # wrong
        ("Rate is 40.4%.", False),  # figure without citation
        ("Late means more than 24 hours.", True),  # domain constant, no tool needed
        ("Top 5 routes [T1].", True),  # number from the question
        ("AUC 0.5063.", True),  # trusted operator fact, no citation needed
    ],
)
def test_grounding(answer, ok):
    tool = {"on_time_rate": 0.404, "avg_delay_hours": 36.04, "total": 1234}
    v = check_grounding(answer, "show the top 5", [tool], trusted={"note": "AUC 0.5063"})
    assert v.ok is ok, v.as_dict()


def test_system_prompt_anchors_this_quarter_to_snapshot():
    p = build_system_prompt("2025-10-14T23:55:00", [], "note")
    assert "'This quarter' means 2025-Q4" in p and OFF_DOMAIN_REPLY in p
