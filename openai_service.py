import json
import os
from typing import Any

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

SYSTEM_PROMPT = """
You are an expert meeting analyst that extracts follow-up action items from transcripts and assigns them to owners.

Your job is to return ONLY high-quality, concrete action items that can be tracked in Monday.com.

Definitions:
- An action item is a specific, future-oriented task or commitment that someone agreed (explicitly or implicitly) to do after the meeting.
- Status updates, general discussion, ideas without commitment, and background explanations are NOT action items.

Strict rules:
- Only include clear next steps or explicit commitments (e.g., “Mark will follow up with Linda for folder IDs”, “Kyle to record Loom training for ICE Monday board”).
- Exclude:
  - Pure status updates.
  - Vague plans without an owner.
  - Long narrative sentences that don’t have a clear verb + owner + outcome.
- Each action must have:
  - A clear owner mapped to the closest matching participant name from the provided list.
  - A short, verb-first title (max ~12 words) that could be used as a Monday.com item name.
  - A brief evidence snippet directly from the transcript (one or two lines) that shows why you created the action.
- If there is no clear owner in the transcript, set owner to "Unassigned".
- If multiple people are involved, choose the primary responsible person; do NOT use teams like "Everyone" or "Team" unless explicitly said.
- Due dates:
  - Use ISO 8601 format (YYYY-MM-DD) ONLY when a specific date or unambiguous phrase like "by Friday, January 30" is mentioned.
  - If the timing is vague (e.g., "this week", "soon") or depends on scheduling, set due_date to null.
- Avoid duplication:
  - If multiple lines describe the same action, combine them into a single, clean item.
- Be conservative:
  - If you are not at least 0.6 confident that something is a follow-up task, do NOT create an item.

Output format (JSON only, no extra text, no markdown):
{
  "items": [
    {
      "title": "Follow up with Linda for Open Asset folder IDs",
      "owner": "Mark Lohr",
      "due_date": null,
      "evidence": "Mark: 'Working with Zaki on the open asset, getting that folder ID from Linda.'",
      "confidence": 0.9
    }
  ]
}

Constraints:
- Return AT MOST the requested number of items.
- Titles must be concise, action-oriented, and avoid filler words.
- Evidence must be a short quote or merged snippet from the transcript that justifies the action.
- The response MUST be valid JSON that can be parsed by json.loads, with no trailing commas and no text before or after the JSON.
"""


def _format_participants(participants: list[str]) -> str:
    if not participants:
        return "None provided."
    return "\n".join(f"- {name}" for name in participants)


def _normalize_transcript_input(
    transcript: str | list[dict[str, str]] | list[Any],
) -> str:
    if isinstance(transcript, str):
        return transcript

    lines: list[str] = []
    for turn in transcript:
        if isinstance(turn, dict):
            speaker_value = turn.get("speaker")
            text_value = turn.get("text")
        else:
            speaker_value = getattr(turn, "speaker", None)
            text_value = getattr(turn, "text", None)

        speaker = (speaker_value or "Speaker").strip() or "Speaker"
        text = (text_value or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines).strip()

def generate_action_items(
    transcript: str | list[dict[str, str]] | list[Any],
    participants: list[str] | None = None,
    max_items: int = 12,
) -> list[dict[str, Any]]:
    client = OpenAI()
    participants = participants or []

    transcript_text = _normalize_transcript_input(transcript)
    if not transcript_text:
        raise ValueError("Transcript was empty after normalization.")

    # Embed max_items in the system prompt so the model “feels” the cap.
    system_prompt = SYSTEM_PROMPT.strip() + f"\n\nThe caller has requested at most {max_items} items."

    user_prompt = (
        "Participants:\n"
        f"{_format_participants(participants)}\n\n"
        f"Max items: {max_items}\n\n"
        "Transcript:\n"
        f"{transcript_text}\n"
    )

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=800,
    )

    output_text = response.output_text.strip()
    if not output_text:
        raise ValueError("OpenAI response was empty.")

    # If your model sometimes wraps JSON in ```json fences, strip them.
    if output_text.startswith("```"):
        output_text = output_text.strip("`")
        if output_text.lower().startswith("json"):
            output_text = output_text[4:].lstrip()

    data = json.loads(output_text)
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("OpenAI response JSON did not contain items list.")

    # Defensive cap
    if len(items) > max_items:
        items = items[:max_items]

    return items
