"""Tests for the core agent loop."""
import pytest
from aegis.core.agent_loop import AgentLoop, StepResult, ToolCall, AgentState

class TestParseResponse:
    
    def test_parses_final_answer(self):
        loop = AgentLoop(llm_client=None, tools={})
        result = loop.parse_response("Final Answer: It is sunny in London.")
        assert result.is_final == True
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