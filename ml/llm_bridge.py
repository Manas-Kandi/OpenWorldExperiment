"""
LLM bridge utilities that let language models interpret Open-Ended Learning Framework goals,
evaluate plans, and issue tool-based commands into the environment.

This module is intentionally lightweight: it does not assume a specific backend
implementation of the game environment, only that a set of callable controls
are provided (move, pickup, drop, interact, etc.).  The bridge exposes those
controls as "tools" that can be surfaced to an LLM via the OpenAI-compatible
chat/completions API (including NVIDIA's integrate endpoint).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


logger = logging.getLogger(__name__)


class MiniQuestController(Protocol):
    """Protocol describing the minimal control surface for the Open-Ended Learning Framework."""

    def move(self, direction: str) -> Dict[str, Any]:
        ...

    def pickup(self) -> Dict[str, Any]:
        ...

    def drop(self) -> Dict[str, Any]:
        ...

    def interact(self) -> Dict[str, Any]:
        ...

    def summarize_state(self) -> str:
        ...


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolActionResult:
    success: bool
    observation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMPlan:
    thought: str = ""
    actions: List[ToolCall] = field(default_factory=list)


class GameToolbox:
    """Registry of tool specs and handlers that the LLM can invoke."""

    def __init__(self):
        self._tools: Dict[str, Tuple[ToolSpec, Callable[..., ToolActionResult]]] = {}

    def register_tool(
        self,
        spec: ToolSpec,
        handler: Callable[..., ToolActionResult],
    ):
        self._tools[spec.name] = (spec, handler)
        logger.debug("Registered tool %s", spec.name)

    def describe_for_prompt(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec, _ in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolActionResult:
        if call.tool not in self._tools:
            raise ValueError(f"Unknown tool: {call.tool}")
        spec, handler = self._tools[call.tool]
        logger.info("Executing tool %s with args %s", spec.name, call.arguments)
        return handler(**call.arguments)


class MiniQuestToolbox(GameToolbox):
    """
    Convenience wrapper that exposes the canonical Open-Ended Learning Framework controls
    (move, pickup, drop, interact) as LLM tools.
    """

    def __init__(self, controller: MiniQuestController):
        super().__init__()
        self.controller = controller
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(
            ToolSpec(
                name="move",
                description="Move the avatar one tile in the specified direction.",
                parameters={
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to move.",
                    }
                },
            ),
            lambda direction: self._wrap_result(self.controller.move(direction)),
        )
        self.register_tool(
            ToolSpec(
                name="pickup",
                description="Pick up an object located on the same tile.",
            ),
            lambda: self._wrap_result(self.controller.pickup()),
        )
        self.register_tool(
            ToolSpec(
                name="drop",
                description="Drop the most recently picked object.",
            ),
            lambda: self._wrap_result(self.controller.drop()),
        )
        self.register_tool(
            ToolSpec(
                name="interact",
                description="Interact with the current tile (e.g., goal zone buttons).",
            ),
            lambda: self._wrap_result(self.controller.interact()),
        )

    @staticmethod
    def _wrap_result(payload: Dict[str, Any]) -> ToolActionResult:
        return ToolActionResult(
            success=payload.get("success", True),
            observation=payload.get("observation", ""),
            metadata=payload,
        )


class SimpleMiniQuestController:
    """
    Minimal in-memory controller used for demos/tests. Replace this with a real
    bridge to the Open-Ended Learning Framework (for example via WebSockets or a Python wrapper
    around the JavaScript game state) when running live.
    """

    def __init__(self):
        self.position = [0, 0]
        self.inventory: List[str] = []

    def move(self, direction: str) -> Dict[str, Any]:
        delta = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }.get(direction, (0, 0))
        self.position[0] += delta[0]
        self.position[1] += delta[1]
        return {
            "success": True,
            "observation": f"Moved {direction} to {tuple(self.position)}",
            "position": tuple(self.position),
        }

    def pickup(self) -> Dict[str, Any]:
        item = f"mock-item-{len(self.inventory)+1}"
        self.inventory.append(item)
        return {
            "success": True,
            "observation": f"Picked up {item}",
            "inventory": list(self.inventory),
        }

    def drop(self) -> Dict[str, Any]:
        if not self.inventory:
            return {"success": False, "observation": "Inventory empty."}
        item = self.inventory.pop()
        return {"success": True, "observation": f"Dropped {item}"}

    def interact(self) -> Dict[str, Any]:
        return {"success": True, "observation": "Interacted with current tile."}

    def summarize_state(self) -> str:
        return json.dumps(
            {
                "position": tuple(self.position),
                "inventory": list(self.inventory),
            }
        )


class LLMGoalCoach:
    """
    Wraps an OpenAI-compatible LLM so it can interpret goals and propose tool
    invocations.
    """

    def __init__(
        self,
        toolbox: GameToolbox,
        model: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 512,
    ):
        self.toolbox = toolbox
        self.model = model or os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct-v0.3")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.base_url = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.api_key = os.getenv("LLM_API_KEY")
        timeout_env = os.getenv("LLM_REQUEST_TIMEOUT")
        self.request_timeout: Optional[int] = int(timeout_env) if timeout_env else None
        self._client: Optional[OpenAI] = None

    @property
    def client(self):
        if self._client:
            return self._client
        if OpenAI is None:
            raise RuntimeError(
                "openai package is not installed. `pip install openai` to enable LLM access."
            )
        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Populate it via .env or environment variables."
            )
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def propose_plan(
        self,
        goal_text: str,
        state_summary: str,
        max_tool_calls: int = 4,
        stream: bool = False,
    ) -> LLMPlan:
        """
        Ask the LLM for a JSON plan that references toolbox tools.
        Falls back to a trivial heuristic plan when no API key is provided.
        """
        if not self.api_key or OpenAI is None:
            logger.warning(
                "LLM_API_KEY missing or openai unavailable; returning heuristic plan."
            )
            return self._fallback_plan(goal_text)

        tools_description = json.dumps(self.toolbox.describe_for_prompt(), indent=2)
        user_content = (
            "You control the Open-Ended Learning Framework via the provided tools.\n"
            f"Goal: {goal_text}\n"
            f"Current State: {state_summary}\n"
            f"You may call up to {max_tool_calls} tools. "
            "Respond strictly as JSON with keys `thought` and `actions`."
        )
        system_prompt = (
            "You are a helpful embodied agent planner. "
            "Reason about the goal, decide if it is already satisfied, "
            "and output a plan that uses the available tools in sequence."
            f"\nAvailable tools:\n{tools_description}"
        )

        request_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=stream,
        )

        if stream:
            chunks = self.client.chat.completions.create(**request_kwargs)
            collected = []
            for chunk in chunks:
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
            raw_response = "".join(collected)
        else:
            completion = self.client.chat.completions.create(**request_kwargs)
            raw_response = completion.choices[0].message.content or ""

        return self._parse_plan(raw_response)

    def _fallback_plan(self, goal_text: str) -> LLMPlan:
        """Simple heuristic when an LLM is not available."""
        actions = []
        if "collect" in goal_text.lower():
            actions.append(ToolCall(tool="pickup"))
        elif "drop" in goal_text.lower():
            actions.append(ToolCall(tool="drop"))
        else:
            actions.append(ToolCall(tool="move", arguments={"direction": "right"}))
        return LLMPlan(thought="Heuristic fallback plan", actions=actions)

    @staticmethod
    def _parse_plan(raw_text: str) -> LLMPlan:
        """Extract JSON from the LLM response."""
        try:
            json_str = LLMGoalCoach._extract_json(raw_text)
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM output: %s", raw_text)
            raise RuntimeError("LLM response was not valid JSON.") from exc

        actions: List[ToolCall] = []
        for entry in payload.get("actions", []):
            tool_name = entry.get("tool") or entry.get("name")
            if not tool_name:
                logger.warning("LLM response missing tool name: %s", entry)
                continue
            arguments = entry.get("arguments") or {}
            if not isinstance(arguments, dict):
                logger.warning(
                    "LLM response provided non-dict arguments for tool %s: %s",
                    tool_name,
                    arguments,
                )
                continue
            actions.append(ToolCall(tool=tool_name, arguments=arguments))

        if not actions:
            logger.warning("LLM returned no valid actions. Raw response: %s", raw_text.strip())

        return LLMPlan(thought=payload.get("thought", ""), actions=actions)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object embedded in text."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise json.JSONDecodeError("JSON object not found", text, 0)
        return text[start : end + 1]


class HybridLLMAgent:
    """
    Wraps a baseline RL agent (e.g., PPOAgent) and lets an LLM periodically
    intervene with tool calls.
    """

    def __init__(
        self,
        rl_agent: Any,
        llm_coach: LLMGoalCoach,
        toolbox: GameToolbox,
        intervention_interval: int = 10,
    ):
        self.rl_agent = rl_agent
        self.llm_coach = llm_coach
        self.toolbox = toolbox
        self.intervention_interval = intervention_interval
        self._step = 0

    def act(self, state: Dict[str, Any], goal_text: str) -> Dict[str, Any]:
        """
        Either execute the next LLM tool call or fall back to the RL policy.
        Returns metadata describing which policy produced the action.
        """
        self._step += 1
        if self._should_query_llm():
            state_summary = json.dumps(state)
            plan = self.llm_coach.propose_plan(goal_text, state_summary)
            if plan.actions:
                call = plan.actions[0]
                result = self.toolbox.execute(call)
                return {
                    "source": "llm",
                    "action": call.tool,
                    "result": result.observation,
                    "thought": plan.thought,
                }

        # Default to RL policy
        action, log_prob, value = self.rl_agent.act(state, goal_text)
        return {
            "source": "rl",
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }

    def _should_query_llm(self) -> bool:
        return self.intervention_interval > 0 and self._step % self.intervention_interval == 0


__all__ = [
    "ToolSpec",
    "ToolCall",
    "ToolActionResult",
    "LLMPlan",
    "GameToolbox",
    "MiniQuestToolbox",
    "SimpleMiniQuestController",
    "LLMGoalCoach",
    "HybridLLMAgent",
]
