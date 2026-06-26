"""Tests for the core agent loop."""
import pytest
from aegis.core.agent_loop import AgentLoop, StepResult, ToolCall, AgentState


class TestParseResponse:
    
    def test_parses_final_answer(self):
        loop = AgentLoop(llm_client=None, tools={})
        result = loop.parse_response("Final Answer: It is sunny in London.")
        assert result.is_final is True
        assert result.thought == "It is sunny in London."
    
    def test_handles_malformed_action_gracefully(self):
        """Agent should not crash when LLM outputs unparseable action."""
        loop = AgentLoop(llm_client=None, tools={})
        result = loop.parse_response("Thought: Let me try.\nAction: badformat[[[unparseable]]]")
        assert result.tool_call is None
        assert "badformat" in result.thought
        
    def test_coerces_numeric_arguments(self):
        """Arguments should be strongly typed numbers or booleans, not flat strings."""
        loop = AgentLoop(llm_client=None, tools={})
        result = loop.parse_response('Action: calculate(expression="2+2", precision=5, active=true)')
        assert result.tool_call.arguments["expression"] == "2+2"
        assert result.tool_call.arguments["precision"] == 5
        assert result.tool_call.arguments["active"] is True

    def test_handles_multiline_action(self):
        """Real LLMs output multi-line string payloads; parser must handle newlines cleanly."""
        loop = AgentLoop(llm_client=None, tools={})
        result = loop.parse_response('Thought: Searching.\nAction: query(\n    table="customers",\n    limit=10\n)')
        assert result.tool_call.name == "query"
        assert result.tool_call.arguments["table"] == "customers"
        assert result.tool_call.arguments["limit"] == 10


class TestExecuteTool:

    def test_safety_gate_blocks_dangerous_tool(self, sample_tools):
        """The safety gate tuple feedback must block dangerous execution parameters."""
        def policy_gate(tool_call: ToolCall):
            if tool_call.name == "delete_database":
                return False, "Unauthorized execution target payload detected"
            return True, "Passed basic operations policy"

        loop = AgentLoop(llm_client=None, tools=sample_tools, safety_gate=policy_gate)
        call = ToolCall(name="delete_database", arguments={"db": "production"})
        
        observation = loop.execute_tool(call)
        assert "Action blocked" in observation
        assert "Unauthorized execution target" in observation


class TestAgentLoopIntegration:

    def test_completes_simple_task(self, mock_llm, sample_tools):
        """The core run loop should gracefully transition through states to a final output."""
        loop = AgentLoop(llm_client=mock_llm, tools=sample_tools)
        final_answer = loop.run("Check weather conditions.")
        
        assert "sunny" in final_answer.lower()
        assert loop.state == AgentState.COMPLETED