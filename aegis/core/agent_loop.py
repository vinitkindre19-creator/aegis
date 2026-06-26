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
        text = text.strip()

        # 1. Handle Final Answer
        if "Final Answer:" in text:
            content = text.split("Final Answer:", 1)[1].strip()
            return StepResult(thought=content, tool_call=None, is_final=True)

        # 2. Extract Thought block cleanly
        thought = ""
        if "Thought:" in text:
            if "Action:" in text:
                thought = text.split("Thought:", 1)[1].split("Action:", 1)[0].strip()
            else:
                thought = text.split("Thought:", 1)[1].strip()

        # 3. Extract Action block safely
        if "Action:" in text:
            action_part = text.split("Action:", 1)[1].strip()
            
            # Find the first '(' and matching last ')'
            if "(" in action_part and ")" in action_part:
                tool_name = action_part.split("(", 1)[0].strip()
                args_str = action_part.split("(", 1)[1].rsplit(")", 1)[0].strip()
                
                args = self._parse_key_value_args(args_str)
                return StepResult(
                    thought=thought,
                    tool_call=ToolCall(name=tool_name, arguments=args),
                    is_final=False
                )

        # 4. Fallback for unparseable input
        return StepResult(thought=text, tool_call=None, is_final=False)

    def _parse_key_value_args(self, args_str: str) -> dict[str, Any]:
        """Parse key=value pairs handling quotes, commas, numbers, and newlines."""
        args = {}
        if not args_str.strip():
            return args

        args_str = args_str.replace("\n", " ")
        
        pairs = []
        current = []
        in_quote = False
        quote_char = None
        
        for char in args_str:
            if char in ('"', "'"):
                if not in_quote:
                    in_quote = True
                    quote_char = char
                elif char == quote_char:
                    in_quote = False
                    quote_char = None
                current.append(char)
            elif char == ',' and not in_quote:
                pairs.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            pairs.append("".join(current).strip())

        for pair in pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                key = key.strip().strip('"').strip("'").strip()
                value = value.strip().strip('"').strip("'").strip()
                if key:
                    args[key] = self._coerce_type(value)
                    
        return args

    def _coerce_type(self, value: str) -> Any:
        """Convert string value to appropriate Python type."""
        if value.lower() == "true": return True
        if value.lower() == "false": return False
        if value.lower() == "null" or value.lower() == "none": return None
        try: return int(value)
        except ValueError: pass
        try: return float(value)
        except ValueError: pass
        return value

    def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the observation."""
        # Temporary mock implementation to support future testing
        pass

    def run(self, task: str) -> str:
        """Execute the full agent loop for a given task."""
        # Temporary mock implementation to support future testing
        pass