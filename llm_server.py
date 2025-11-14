"""
FastAPI server that bridges the Mini-Quest frontend with the LLM planning loop.

How to run:
    source .venv/bin/activate
    python -m pip install fastapi uvicorn
    uvicorn llm_server:app --reload --port 8001

The frontend (game.js) will call POST /llm/step with the current goal and state
and receive a list of tool actions plus the LLM's reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.llm_bridge import (
    LLMGoalCoach,
    MiniQuestToolbox,
    SimpleMiniQuestController,
    ToolCall,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Mini-Quest LLM Bridge", version="1.0")

# Allow local development (frontend served via http://localhost:* or file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_controller = SimpleMiniQuestController()
_toolbox = MiniQuestToolbox(_controller)
_coach = LLMGoalCoach(_toolbox)


class LLMRequest(BaseModel):
    goal: str = Field(..., description="Current textual goal")
    state: Dict[str, Any] = Field(..., description="Serialized game state snapshot")
    max_tool_calls: int = Field(4, ge=1, le=8, description="Max actions per plan request")
    stream: bool = Field(False, description="Whether to stream tokens (off by default)")


class LLMAction(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    thought: str
    actions: List[LLMAction] = Field(default_factory=list)


def _tool_call_to_dict(call: ToolCall) -> Dict[str, Any]:
    return {
        "tool": call.tool,
        "arguments": call.arguments or {},
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/llm/step", response_model=LLMResponse)
def llm_step(payload: LLMRequest) -> LLMResponse:
    """
    Generate a short plan from the LLM given the current goal and state snapshot.
    Returns the thought plus the ordered tool calls for the frontend to execute.
    """
    state_summary = json.dumps(payload.state)
    plan = _coach.propose_plan(
        goal_text=payload.goal,
        state_summary=state_summary,
        max_tool_calls=payload.max_tool_calls,
        stream=payload.stream,
    )

    actions = [_tool_call_to_dict(call) for call in plan.actions]
    logger.info("Generated plan with %d actions", len(actions))
    return LLMResponse(thought=plan.thought, actions=actions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm_server:app", host="0.0.0.0", port=8001, reload=False)
