"""Core ReAct agent loop implementation.

This module provides the fundamental agent loop that all Aegis agents use.
It implements the Reason+Act (ReAct) pattern with pluggable tools,
safety gates, and memory.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Tuple

from pydantic import BaseModel

# Initialize structured logger
logger = logging.getLogger(__name__)


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
        safety_gate: Optional[Callable[[ToolCall], Tuple[bool, str]]] = None,
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
            
            if "(" in action_part and ")" in action_part:
                tool_name = action_part.split("(", 1)[0].strip()
                args_str = action_part.split("(", 1)[1].rsplit(")", 1)[0].strip()
                
                args = self._parse_key_value_args(args_str)
                return StepResult(
                    thought=thought,
                    tool_call=ToolCall(name=tool_name, arguments=args),
                    is_final=False
                )

        return StepResult(thought=text, tool_call=None, is_final=False)

    def _parse_key_value_args(self, args_str: str) -> dict[str, Any]:
        """Parse key=value pairs handling quotes, commas, numbers, and newlines."""
        args = {}
        if not args_str.strip():
            return args

        # NOTE: Key-value string parsing is best-effort. Complex nested structures
        # should strictly use JSON formatting schemas in down-stream tasks.
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
        """Execute a tool call safely with validation check gates."""
        logger.debug(f"Intercepting execution request for tool: {tool_call.name}")
        
        # 1. Advanced Evaluation Safety Check
        if self.safety_gate:
            allowed, reason = self.safety_gate(tool_call)
            if not allowed:
                logger.warning(f"Safety Gate Intervention! Denied execution of '{tool_call.name}'. Reason: {reason}")
                return f"Action blocked: {reason}. Please try an alternative approach."

        # 2. Scope verification
        if tool_call.name not in self.tools:
            logger.error(f"Execution Target Misalignment: Unknown tool '{tool_call.name}' called.")
            return f"Error: Unknown tool '{tool_call.name}' requested."

        # 3. Dynamic payload execution
        try:
            result = self.tools[tool_call.name](**tool_call.arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Runtime Exception in tool '{tool_call.name}': {str(e)}")
            return f"Error executing tool '{tool_call.name}': {str(e)}"

    def run(self, task: str) -> str:
        """Execute the full agent orchestration execution loop."""
        logger.info(f"Starting execution run lifecycle for task payload: {task[:100]}")
        self.add_message("user", task)
        
        for step in range(self.max_steps):
            logger.info(f"Orchestrating Step {step + 1}/{self.max_steps}")
            self.state = AgentState.THINKING
            
            # Fetch response generation from client engine
            raw_response = self.llm_client.generate(self.history)
            result = self.parse_response(raw_response)
            
            # Record state trajectory history
            self.add_message("assistant", raw_response)
            
            if result.is_final:
                logger.info(f"Terminal node reached successfully. Task completed.")
                self.state = AgentState.COMPLETED
                return result.thought
                
            if result.tool_call:
                self.state = AgentState.ACTING
                observation = self.execute_tool(result.tool_call)
                logger.debug(f"Tool Observation captured: {observation[:100]}")
                self.add_message("tool", observation)
            else:
                # Loop recovery sequence for formatting friction
                logger.warning("Agent encountered loop parsing friction. Prompting corrective recovery.")
                self.add_message("system", "Format error: Please supply an Action or a Final Answer block.")

        logger.error("System Failure: Orchestration termination criteria breached via step ceiling bounds.")
        self.state = AgentState.FAILED
        return "Max steps reached without resolution."