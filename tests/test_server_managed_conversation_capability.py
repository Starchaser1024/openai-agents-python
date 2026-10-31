"""Server-managed conversation settings must not drop the history for adapters that
cannot keep it server-side."""

from __future__ import annotations

import pytest
from openai.types.responses import ResponseFunctionToolCall

from agents import Agent, RunConfig, Runner, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run_internal.oai_conversation import model_keeps_conversation_server_side
from agents.testing import ModelStep, ScriptedModel

from .test_responses import get_text_message

RUN_CONFIG = RunConfig(tracing_disabled=True)


class ClientSideHistoryModel(ScriptedModel):
    """A model that needs the caller to resend the conversation, like Chat Completions."""

    supports_server_managed_conversation_state = False


@function_tool
def lookup() -> str:
    """Look something up."""
    return "42"


def _tool_call() -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall.model_validate(
        {
            "id": "fc_1",
            "call_id": "call_1",
            "type": "function_call",
            "name": "lookup",
            "arguments": "{}",
        }
    )


def _labels(items: object) -> list[str]:
    assert isinstance(items, list)
    return [str(item.get("role") or item.get("type")) for item in items if isinstance(item, dict)]


@pytest.mark.asyncio
async def test_client_side_model_resends_history_when_a_conversation_id_is_given() -> None:
    model = ClientSideHistoryModel(
        [ModelStep(output=[_tool_call()]), ModelStep(output=[get_text_message("done")])]
    )
    agent = Agent(name="client-side", model=model, tools=[lookup])

    await Runner.run(agent, "hello", conversation_id="conv_1", run_config=RUN_CONFIG)

    assert _labels(model.calls[0].input) == ["user"]
    assert _labels(model.calls[1].input) == ["user", "function_call", "function_call_output"]


@pytest.mark.asyncio
async def test_server_side_model_keeps_sending_only_the_new_turn() -> None:
    model = ScriptedModel(
        [ModelStep(output=[_tool_call()]), ModelStep(output=[get_text_message("done")])]
    )
    agent = Agent(name="server-side", model=model, tools=[lookup])

    await Runner.run(agent, "hello", conversation_id="conv_1", run_config=RUN_CONFIG)

    assert _labels(model.calls[1].input) == ["function_call_output"]


def test_chat_completions_declares_it_cannot_keep_conversation_server_side() -> None:
    assert OpenAIChatCompletionsModel.supports_server_managed_conversation_state is False
    assert model_keeps_conversation_server_side(ScriptedModel([])) is True
    assert model_keeps_conversation_server_side(ClientSideHistoryModel([])) is False
