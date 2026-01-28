import json
import os
import re
import sys
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

API_URL = "https://api.fireflies.ai/graphql"
API_KEY_ENV_VAR = "FF_API_KEY"

_STATE = {"dotenv_loaded": False}


def _ensure_dotenv_loaded() -> None:
    if not _STATE["dotenv_loaded"]:
        load_dotenv()
        _STATE["dotenv_loaded"] = True

def _graphql_request(query: str, variables: dict) -> dict:
    _ensure_dotenv_loaded()

    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Missing {API_KEY_ENV_VAR}. Set it to your Fireflies API key."
        )

    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "firefliesrand/1.0",
    }
    req = request.Request(API_URL, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Fireflies API HTTP {exc.code}: {error_body}"
        ) from exc

    if "errors" in data:
        raise RuntimeError(f"Fireflies API error: {data['errors']}")

    return data["data"]


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "transcript"


def _clean_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _fetch_recent_transcripts(limit: int, title_filter: str | None = None) -> list[dict]:
    query = """
    query Transcripts($title: String, $limit: Int, $skip: Int) {
      transcripts(title: $title, limit: $limit, skip: $skip) {
        id
        title
        date
        transcript_url
      }
    }
    """
    data = _graphql_request(query, {"title": title_filter, "limit": limit, "skip": 0})
    transcripts = data.get("transcripts", [])
    transcripts.sort(key=lambda item: item.get("date") or 0, reverse=True)
    return transcripts[:limit]


def _fetch_transcript_text(transcript_id: str) -> dict:
    query = """
    query Transcript($transcriptId: String!) {
      transcript(id: $transcriptId) {
        id
        title
        dateString
        speakers {
          id
          name
        }
        sentences {
          index
          speaker_name
          text
        }
      }
    }
    """
    data = _graphql_request(query, {"transcriptId": transcript_id})
    return data.get("transcript", {})


def _format_transcript_text(transcript: dict) -> str:
    lines = []
    title = transcript.get("title") or "Untitled"
    date_string = _clean_datetime(transcript.get("dateString")) or "Unknown date"
    transcript_id = transcript.get("id") or "Unknown ID"
    lines.append(f"Title: {title}")
    lines.append(f"Date: {date_string}")
    lines.append(f"ID: {transcript_id}")
    participants = _extract_participants(transcript)
    if participants:
        lines.append("Participants: " + ", ".join(participants))
    lines.append("")

    sentences = transcript.get("sentences") or []
    sentences.sort(key=lambda item: item.get("index") or 0)
    for sentence in sentences:
        speaker = sentence.get("speaker_name") or "Speaker"
        text = sentence.get("text") or ""
        if text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines).strip() + "\n"


def _extract_participants(transcript: dict) -> list[str]:
    participants: list[str] = []
    seen: set[str] = set()

    for speaker in transcript.get("speakers") or []:
        name = (speaker.get("name") or "").strip()
        if name and name not in seen:
            participants.append(name)
            seen.add(name)

    for sentence in transcript.get("sentences") or []:
        name = (sentence.get("speaker_name") or "").strip()
        if name and name not in seen:
            participants.append(name)
            seen.add(name)

    return participants


def _build_transcript_turns(transcript: dict) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    sentences = transcript.get("sentences") or []
    sentences.sort(key=lambda item: item.get("index") or 0)
    for sentence in sentences:
        speaker = (sentence.get("speaker_name") or "Speaker").strip() or "Speaker"
        text = (sentence.get("text") or "").strip()
        if not text:
            continue
        turns.append({"speaker": speaker, "text": text})
    return turns


def download_recent_transcripts(limit: int = 6, title_filter: str | None = None) -> Path:
    output_dir = Path(__file__).resolve().parent / "downloaded_transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts = _fetch_recent_transcripts(limit, title_filter=title_filter)
    if not transcripts:
        print("No transcripts returned by the API.")
        return output_dir

    used_names: set[str] = set()
    for transcript_meta in transcripts:
        transcript_id = transcript_meta.get("id")
        if not transcript_id:
            continue

        transcript = _fetch_transcript_text(transcript_id)
        participants = _extract_participants(transcript)
        title = transcript.get("title") or transcript_meta.get("title") or transcript_id
        base_name = _sanitize_filename(title)
        filename = base_name
        counter = 1
        while filename in used_names:
            counter += 1
            filename = f"{base_name}_{counter}"
        used_names.add(filename)

        file_path = output_dir / f"{filename}.txt"
        file_path.write_text(_format_transcript_text(transcript), encoding="utf-8")

        json_path = output_dir / f"{filename}.json"
        json_payload = {
            "metadata": {
                "id": transcript.get("id"),
                "title": transcript.get("title"),
                "date": _clean_datetime(transcript.get("dateString")),
            },
            "participants": participants,
            "transcript": _build_transcript_turns(transcript),
            "max_items": 12,
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        print(f"Saved {file_path.name} and {json_path.name}")

    return output_dir


def main() -> int:
    try:
        download_recent_transcripts(title_filter="Weekly sync")
    except (RuntimeError, HTTPError, URLError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
