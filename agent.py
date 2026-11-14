import json
import re
from copy import deepcopy
from typing import Any

from ollama import Client

import config
from outcomes import requested_exit
from tools import TOOL_SCHEMAS, AgentSession, CandidateAPI, CandidateTools
from validators import is_candidate_complete


class CandidateAgent:
    def __init__(
        self,
        candidate_id: str,
        api: CandidateAPI,
        *,
        llm: Any = None,
        model: str | None = None,
        max_tool_rounds: int = 6,
    ) -> None:
        if not 1 <= max_tool_rounds <= 12:
            raise ValueError("max_tool_rounds must be between 1 and 12.")
        record = api.get_candidate(candidate_id)
        self.session = AgentSession(
            candidate_id, deepcopy(record), deepcopy(record), record["version"]
        )
        self.session.refresh(record)
        self.tools = CandidateTools(self.session, api)
        self.llm = llm if llm is not None else Client(host=config.OLLAMA_BASE_URL, timeout=120.0)
        self.model = model or config.OLLAMA_MODEL
        self.max_tool_rounds = max_tool_rounds
        self.started = False
        if is_candidate_complete(record):
            if record.get("collection_status") != "completed":
                # No attempted contact: only correct the status, with no contact metadata writes.
                result = self.tools._write({"collection_status": "completed"})
                if not result["ok"]:
                    self.session.outcome = "needs_human"
                    self.session.ended = True
                    return
            if is_candidate_complete(self.session.latest_candidate):
                self.session.outcome = "completed"
                self.session.outcome_persisted = True
                self.session.ended = True
                return
        prompt = (config.PROJECT_ROOT / "prompts/system_prompt.txt").read_text()
        self.session.history.append({"role": "system", "content": prompt})

    def start(self) -> str | None:
        if self.session.ended or self.started:
            return None
        self.started = True
        name = self.session.latest_candidate.get("full_name", "the applicant")
        # This gate is deterministic; application collection after it is model-directed.
        return self._say(
            f"Hi! I'm ACME Corp's internship application assistant. Am I speaking with {name}? "
            "Please reply yes or no."
        )

    def respond(self, message: str) -> str | None:
        if self.session.ended:
            return None
        if not self.started:
            raise ValueError("Call start() before submitting a candidate reply.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("A nonempty candidate reply is required.")
        session = self.session
        session.history.append({"role": "user", "content": message})
        intent = requested_exit(message)
        if (
            session.identity != "confirmed"
            and intent is None
            and re.match(
                r"^(?:no\b|nope\b|(?:i am|i'm) not\b|my name is not\b)",
                message.strip(),
                re.I,
            )
        ):
            session.identity = "denied"
            intent = {
                "collection_status": "needs_human",
                "notes": "Identity denied: " + message,
                "refused_fields": [],
            }
        if intent and (
            intent["collection_status"] == "needs_human" or session.identity != "confirmed"
        ):
            if re.search(
                r"wrong (?:person|number)|not me|not my (?:record|application)|someone else's",
                message,
                re.I,
            ):
                session.identity = "denied"
            if intent["collection_status"] == "refused":
                intent = {
                    "collection_status": "no_answer",
                    "notes": intent["notes"],
                    "refused_fields": [],
                }
            session.requested_outcome = intent
            self.stop()
            return self._say(self._outcome_message())
        if session.identity != "confirmed":
            answer = message.strip().casefold().rstrip(".! ")
            if answer in {"no", "nope", "wrong person", "wrong number", "not me"}:
                session.identity = "denied"
                self.stop(
                    "needs_human", "Recipient denied identity; do not collect applicant data."
                )
                return self._say("Sorry for the mix-up. I'll stop this conversation.")
            name = " ".join(str(session.latest_candidate.get("full_name", "")).casefold().split())
            confirmations = {
                "yes",
                "yes, that's me",
                "that's me",
                "yes that is me",
                "yes, that is me",
            }
            confirmations.update({f"i am {name}", f"i'm {name}", f"yes, i am {name}"})
            if answer not in confirmations:
                return self._say(
                    "Before we discuss the application, please confirm: is that your name? "
                    "Reply yes or no."
                )
            session.identity = "confirmed"
        else:
            session.candidate_messages.append(message)
            session.validated_values.clear()
            session.requested_outcome = intent
        return self._run_turn()

    def _say(self, message: str) -> str:
        self.session.history.append({"role": "assistant", "content": message})
        return message

    def _run_turn(self) -> str:
        session = self.session
        correction = ""
        for round_index in range(self.max_tool_rounds + 1):
            inspected = any(
                entry["name"] == "inspect_candidate_state" and entry["result"].get("ok")
                for entry in session.tool_calls
            )
            available_tools = TOOL_SCHEMAS
            action = (
                "Choose the next useful question or tool from actual state and the latest reply."
            )
            if not inspected:
                available_tools = TOOL_SCHEMAS[:1]
                action = "Call inspect_candidate_state now, before speaking."
            elif session.validated_values:
                available_tools = TOOL_SCHEMAS[2:3]
                action = "Call save_candidate_information now with updates=" + json.dumps(
                    session.validated_values
                )
            elif session.requested_outcome:
                action = (
                    "The candidate requested an exit. Validate and save any other explicit "
                    "answers in their latest reply, then finish with "
                    + json.dumps(session.requested_outcome)
                    + ". Do not ask another question or schedule anything."
                )
            elif self.tools.ready_to_complete():
                available_tools = TOOL_SCHEMAS[3:]
                action = "Call finish_collection now with completed, notes and refused_fields=[]."
            messages = deepcopy(session.history)
            messages[0]["content"] += (
                "\nAPPLICATION STATE: Identity confirmed. " + action + " " + correction
            )
            if (
                session.validated_values
                or self.tools.ready_to_complete()
                or session.requested_outcome
            ):
                # Application control is never added to candidate evidence.
                messages.append(
                    {
                        "role": "user",
                        "content": "Application control, not candidate "
                        "input: " + action + ". Use the native tool, not a text response.",
                    }
                )
            try:
                response = self.llm.chat(
                    model=self.model,
                    messages=messages,
                    tools=available_tools,
                    stream=False,
                    options={"temperature": 0, "num_predict": 700, "num_ctx": 8192},
                )
                assistant = response.message.model_dump(exclude_none=True)
                if assistant.get("role") != "assistant":
                    raise ValueError("Expected an assistant message.")
            except Exception:
                # Provider exception strings may contain request content; expose only a safe error.
                session.last_error = "inference_error"
                self.stop(
                    "needs_human", "Local model inference failed; review conversation progress."
                )
                return self._say("I'm having a technical problem and need human help to continue.")
            session.history.append(assistant)
            calls = assistant.get("tool_calls") or []
            if not calls:
                text = assistant.get("content", "").strip()
                if session.requested_outcome and not session.validated_values:
                    session.history.pop()
                    self.stop()
                    return self._say(self._outcome_message())
                if not inspected or any(
                    schema["function"]["name"] in text for schema in TOOL_SCHEMAS
                ):
                    session.history.pop()
                    correction = "Emit a native tool call, never simulated tool results in prose."
                    continue
                if session.validated_values or self.tools.ready_to_complete():
                    session.history.pop()
                    correction = "The operation is still pending. Emit a native tool call now."
                    continue
                if not text:
                    self.stop("needs_human", "Model returned neither text nor a tool request.")
                    return self._say("I couldn't continue reliably; this needs human review.")
                if re.search(r"%|\bpercent(?:age)?\b", text, re.I) and re.search(
                    r"\b(?:convert|divide|enter)\b", text, re.I
                ):
                    text = (
                        "Please share your official CGPA on a 10-point scale. A percentage "
                        "cannot be converted reliably without your institution's grading rules."
                    )
                    session.history[-1]["content"] = text
                if re.search(
                    r"\b(saved|submitted)\b|\b(?:application|information|record)\s+"
                    r"(?:is\s+|has been\s+)?(?:complete|completed|updated)\b",
                    text,
                    re.I,
                ):
                    return self._safe_save_reply(text)
                return text
            limited = round_index == self.max_tool_rounds or len(calls) > 8
            for call in calls:
                function = call.get("function", {})
                name, arguments = function.get("name", ""), function.get("arguments", {})
                result = (
                    {"ok": False, "error": "tool_limit"}
                    if limited
                    else self.tools.execute(name, arguments)
                )
                session.tool_calls.append(
                    {"name": name, "arguments": deepcopy(arguments), "result": deepcopy(result)}
                )
                session.history.append(
                    {"role": "tool", "tool_name": name, "content": json.dumps(result, default=str)}
                )
            if session.ended:
                return self._say(self._outcome_message())
            if session.disputed_fields:
                session.requested_outcome = None
                self.stop(
                    "needs_human", "Disputed fields: " + ", ".join(sorted(session.disputed_fields))
                )
                return self._say(self._outcome_message())
            if limited:
                break
        self.stop("needs_human", "Tool-call safety limit reached; review before continuing.")
        return self._say("I couldn't resolve this reliably; this needs human review.")

    def _safe_save_reply(self, text: str) -> str:
        if self.session.outcome == "completed" and self.session.outcome_persisted:
            return text
        # Do not let generated prose substitute for the actual database outcome.
        replacement = "Let's continue checking the remaining information."
        if self.session.confirmed_values:
            replacement = "Your confirmed information was saved. " + replacement
        else:
            replacement = "I haven't confirmed a save of your information yet. " + replacement
        self.session.history[-1]["content"] = replacement
        return replacement

    def _outcome_message(self) -> str:
        if not self.session.outcome_persisted:
            return "I'll stop here. Saving the outcome was not confirmed; human review is needed."
        return {
            "completed": "Thank you! Your application information is complete and saved.",
            "in_progress": "We'll stop here. Your progress and follow-up notes have been saved.",
            "no_answer": "I'll stop here and record that collection could not proceed.",
            "refused": "Understood. I've recorded your preference and will stop here.",
            "needs_human": "This needs human review. I've recorded the issue and will stop here.",
        }[self.session.outcome]

    def stop(
        self, status: str = "no_answer", notes: str = "Candidate left without providing data."
    ) -> dict[str, Any]:
        """Host hook for a future CLI's timeout, quit or failure handling."""
        if self.session.ended:
            return {"ok": self.session.outcome_persisted, "ended": True}
        if status == "no_answer" and (
            self.session.confirmed_values or self.session.validated_values
        ):
            status = "in_progress"
            notes = "Candidate left after providing information; retain saved progress."
        arguments = self.session.requested_outcome or {
            "collection_status": status,
            "notes": notes,
            "refused_fields": [],
        }
        result = self.tools.execute("finish_collection", arguments)
        if not result["ok"]:
            self.session.last_error = result.get("error")
            self.session.outcome = "needs_human"
            self.session.ended = True
            self.session.outcome_persisted = False
        return result
