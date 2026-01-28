from fastapi import FastAPI, HTTPException
from openai_service import generate_action_items

from pydantic import BaseModel, Field, computed_field

app = FastAPI()

class ActionItem(BaseModel):
    title: str =  Field(..., description="Actionable Item Title")
    owner: str = Field(...,description="Assigned Owner")
    evidence: str | None = Field(default=None, description = "Short quote from tscript supporting A Item")
    confidence: float | None = Field(
        default = None, description= "0-1 confidence in the assignment"
    )


class TranscriptTurn(BaseModel):
    speaker: str = Field(..., description="Speaker name: ")
    text: str = Field(..., description="Spoken text: ") 



class ActionItemsRequest(BaseModel):
    transcript: str | list[TranscriptTurn] = Field(
        ..., description="Full transcript text or ordered speaker turns."
    )
    participants: list[str] = Field(
        default_factory=list, description="Participant names to assign tasks to."
    )
    max_items: int = Field(
        default=12, ge=1, le=50, description="Maximum number of action items."
    )


class ActionItemsResponse(BaseModel):
    action_items: list[ActionItem]

    @computed_field
    @property
    def number_of_action_items(self) -> int:
        return len(self.action_items)


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/action-items", response_model=ActionItemsResponse)
def create_action_items(payload: ActionItemsRequest) -> ActionItemsResponse:
    if not payload.transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        items = generate_action_items(
            transcript=payload.transcript,
            participants=payload.participants,
            max_items=payload.max_items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ActionItemsResponse(action_items=items)
