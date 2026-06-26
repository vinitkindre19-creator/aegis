"""Core ReAct agent loop implementation.

This module provides the fundamental agent loop that all Aegis agents use.
It implements the Reason+Act (ReAct) pattern with pluggable tools,
safety gates, and memory.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel


class AgentState(str, Enum):
    """Possible states in the agent lifecycle."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class Message(BaseModel):
    """A single message in the agent's history."""
    role: str  # "user", "assistant", "tool", "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A parsed tool call extracted from the LLM output."""
    name: str
    arguments: dict[str, Any]


@dataclass
class StepResult:
    """The result of one loop iteration."""
    thought: str
    tool_call: Optional[ToolCall] = None
    observation: Optional[str] = None
    is_final: bool = False


class AgentLoop:
    """A ReAct agent loop that Reason+Act until completion."""

    def __init__(
        self,
        llm_client: Any,
        tools: dict[str, Callable],
        max_steps: int = 10,
        safety_gate: Optional[Callable] = None,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.safety_gate = safety_gate
        self.history: list[Message] = []
        self.state = AgentState.IDLE

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Append a message to the conversation history."""
        self.history.append(Message(role=role, content=content, metadata=metadata))

    def parse_response(self, text: str) -> StepResult:
        """Parse LLM output into a structured StepResult."""
        if "Final Answer:" in text:
            parts = text.split("Final Answer:", 1)
            thought_part = parts[0].replace("Thought:", "").strip()
            final_answer = parts[1].strip()
            return StepResult(
                thought=thought_part,
                observation=final_answer,
                is_final=True
            )
        
        try:
            if "Action:" in text:
                parts = text.split("Action:", 1)
                thought = parts[0].replace("Thought:", "").strip()
                action_text = parts[1].strip()
                
                if "(" in action_text and action_text.endswith(")"):
                    tool_name, args_str = action_text.split("(", 1)
                    tool_name = tool_name.strip()
                    args_str = args_str.rstrip(")").strip()
                    
                    arguments = {}
                    if args_str:
                        pairs = args_str.split(",")
                        for pair in pairs:
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                val = v.strip().strip("'\"")
                                if val.isdigit():
                                    val = int(val)
                                arguments[k.strip()] = val
                                
                    return StepResult(
                        thought=thought,
                        tool_call=ToolCall(name=tool_name, arguments=arguments),
                        is_final=False
                    )
        except Exception:
            pass

        return StepResult(thought=text, tool_call=None, is_final=False)

    def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call safely with error handling and safety checks."""
        if self.safety_gate:
            if not self.safety_gate(tool_call):
                return f"Error: Tool execution denied by safety gate for '{tool_call.name}'."

        if tool_call.name not in self.tools:
            return f"Error: Tool '{tool_call.name}' is not recognized or available."
        
        try:
            tool_fn = self.tools[tool_call.name]
            result = tool_fn(**tool_call.arguments)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_call.name}': {str(e)}"

    def run(self, task: str) -> str:
        """Execute the full agent loop for a given task."""
        self.state = AgentState.THINKING
        self.add_message(role="user", content=task)
        
        current_step = 0
        final_output = "Error: Agent failed to reach a conclusion."

        while current_step < self.max_steps:
            raw_response = self.llm_client.generate(self.history)
            self.add_message(role="assistant", content=raw_response)
            
            step_res = self.parse_response(raw_response)
            
            if step_res.is_final:
                self.state = AgentState.COMPLETED
                final_output = step_res.observation if step_res.observation else step_res.thought
                break
                
            if step_res.tool_call:
                self.state = AgentState.ACTING
                observation = self.execute_tool(step_res.tool_call)
                self.add_message(
                    role="tool", 
                    content=observation, 
                    metadata={"tool_name": step_res.tool_call.name}
                )
                
            current_step += 1
        
        if self.state != AgentState.COMPLETED:
            self.state = AgentState.FAILED
            final_output = f"Error: Max execution limit of {self.max_steps} steps exceeded."
            
        return final_output