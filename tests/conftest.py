"""Shared test fixtures for Aegis."""
import pytest

class MockLLMClient:
    """A mock LLM client for testing agent loops without API calls.
    
    You configure it with a sequence of responses it should return.
    This lets you test exact scenarios deterministically.
    """
    
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.history: list[dict] = []
    
    def generate(self, messages: list[dict]) -> str:
        """Return the next pre-configured response."""
        self.history.append({"messages": messages, "call": self.call_count})
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Final Answer: I've run out of pre-configured responses."

@pytest.fixture
def mock_llm():
    """Create a mock LLM with default responses."""
    return MockLLMClient(responses=[
        'Thought: I need to check the weather. Action: get_weather(city="London")',
        'Final Answer: The weather in London is sunny.',
    ])

@pytest.fixture
def sample_tools():
    """Create a sample tool registry for testing."""
    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny."
    
    return {"get_weather": get_weather}