from __future__ import annotations

import pytest

from agents import Agent, RunContextWrapper, Runner
from agents.decorators import tool
from agents.testing import ScriptedModel, assistant_message, function_call
from agents.tool import default_tool_error_function


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_default_failure_does_not_request_a_retry_after_a_lost_response(
    streamed: bool,
) -> None:
    # The tool commits its effect and then loses the response, as when a write succeeds
    # but the connection drops before the reply arrives. The default output must not ask
    # the model to call the tool again, because the effect has already happened.
    committed: list[str] = []

    @tool
    def create_order(sku: str) -> str:
        """Create an order."""
        committed.append(sku)
        raise ConnectionError("connection closed before the response arrived")

    model = ScriptedModel(
        [
            [function_call("create_order", {"sku": "A1"}, call_id="order")],
            [assistant_message("checked")],
        ]
    )
    agent = Agent(name="orders", model=model, tools=[create_order])
    if streamed:
        result = Runner.run_streamed(agent, "Order one A1.")
        async for _ in result.stream_events():
            pass
    else:
        result = await Runner.run(agent, "Order one A1.")

    assert result.final_output == "checked"
    assert committed == ["A1"]
    next_input = model.calls[-1].input
    assert isinstance(next_input, list)
    outputs = [item for item in next_input if item.get("type") == "function_call_output"]
    assert len(outputs) == 1
    output = outputs[0]["output"]
    assert output == default_tool_error_function(RunContextWrapper(context=None), ConnectionError())
    assert "try again" not in output.lower()
    assert "may or may not have taken effect" in output
    assert "before calling it again" in output
