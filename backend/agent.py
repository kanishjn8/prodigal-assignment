import json
import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ollama import Client

import config
from decision_log import RecordDecision, ignore_decision
from outcomes import FIELD_LABELS, requested_exit
from tools import CONFIRMATION_FIELDS, TOOL_SCHEMAS, AgentSession, CandidateAPI, CandidateTools
from validators import is_candidate_complete, validate_field

DIRECT_ANSWER_FIELDS = {"email", "cgpa", "internship_duration_months"} | CONFIRMATION_FIELDS
RETRY_QUESTIONS = {
    "email": "What is your email address?",
    "graduation_year": "What is your expected graduation year?",
    "available_from": "What is your start date (YYYY-MM-DD)?",
    "cgpa": "What is your official CGPA on a 10-point scale?",
    "internship_duration_months": "How many months can you intern?",
}
CONFIRMATION_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["accept", "edit", "unclear"]}},
    "required": ["decision"],
    "additionalProperties": False,
}


class CandidateAgent:
    def __init__(
        self,
        candidate_id: str,
        api: CandidateAPI,
        *,
        llm: Any = None,
        model: str | None = None,
        max_tool_rounds: int = 6,
        on_progress: Callable[[str], None] | None = None,
        on_decision: RecordDecision = ignore_decision,
    ) -> None:
        if not 1 <= max_tool_rounds <= 12:
            raise ValueError("max_tool_rounds must be between 1 and 12.")
        self.on_progress = on_progress or (lambda _: None)
        self.on_decision = on_decision
        self.on_decision("candidate_fetch_started")
        self.on_progress("Fetching your application details")
        record = api.get_candidate(candidate_id)
        self.session = AgentSession(
            candidate_id, deepcopy(record), deepcopy(record), record["version"]
        )
        self.session.refresh(record)
        self.on_decision(
            "candidate_plan",
            version=self.session.version,
            missing=self.session.missing_fields,
            invalid=sorted(self.session.invalid_fields),
            needs_clarification=sorted(self.session.pending_clarification),
        )
        self.tools = CandidateTools(
            self.session, api, on_progress=self.on_progress, on_decision=self.on_decision
        )
        self.llm = llm if llm is not None else Client(host=config.OLLAMA_BASE_URL, timeout=120.0)
        self.model = model or config.OLLAMA_MODEL
        self.max_tool_rounds = max_tool_rounds
        self.started = False
        self.retry_field: str | None = None
        self.awaiting_confirmation = False
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
        prompt = (config.BACKEND_ROOT / "prompts/system_prompt.txt").read_text()
        self.session.history.append({"role": "system", "content": prompt})

    def set_progress(self, callback: Callable[[str], None] | None) -> None:
        self.on_progress = callback or (lambda _: None)
        self.tools.on_progress = self.on_progress

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
            self.on_decision("exit_detected", status=intent["collection_status"])
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
            self.on_decision("identity_confirmed")
        else:
            session.candidate_messages.append(message)
            if self.awaiting_confirmation:
                decision = "edit" if intent else self._confirmation_decision(message)
                if decision == "unclear":
                    return self._say(
                        "Please use /accept to save the values shown, or /edit to change them."
                    )
                self.awaiting_confirmation = False
                if decision == "accept":
                    updates = dict(session.validated_values)
                    session.confirmed_by_candidate = {
                        key: value for key, value in updates.items() if key in CONFIRMATION_FIELDS
                    }
                    try:
                        result = self.tools.execute(
                            "save_candidate_information", {"updates": updates}
                        )
                    finally:
                        session.confirmed_by_candidate.clear()
                    if not result["ok"]:
                        self.stop("needs_human", "Could not save the confirmed values safely.")
                        return self._say(self._outcome_message())
                    self.retry_field = None
                    return self._run_turn()
                declined = session.validated_values.keys() & CONFIRMATION_FIELDS
                if len(declined) == 1:
                    self.retry_field = next(iter(declined))
                if intent is None and message.strip().casefold() in {"no", "/edit"}:
                    session.validated_values.clear()
                    return self._say(
                        RETRY_QUESTIONS[self.retry_field]
                        if self.retry_field
                        else "Please provide your corrected graduation year and start date."
                    )
            session.validated_values.clear()
            session.requested_outcome = intent
            if intent:
                self.on_decision("exit_detected", status=intent["collection_status"])
            if intent is None:
                correction = self._check_direct_answer(message)
                if correction:
                    return self._say(correction)
        return self._run_turn()

    def _check_direct_answer(self, message: str) -> str | None:
        answer = message.strip()
        field = self.retry_field
        if field is None:
            previous = next(
                (
                    entry.get("content", "")
                    for entry in reversed(self.session.history[:-1])
                    if entry["role"] == "assistant" and not entry.get("tool_calls")
                ),
                "",
            )
            if previous.count("?") != 1:
                return None
            question = re.split(r"[.!]\s+", previous.split("?")[0])[-1]
            fields = [
                key for key, pattern in FIELD_LABELS.items() if re.search(pattern, question, re.I)
            ]
            if len(fields) != 1 or fields[0] not in DIRECT_ANSWER_FIELDS:
                return None
            field = fields[0]
        if field == "internship_duration_months":
            duration = re.fullmatch(r"([0-9]{1,15})\s*months?\.?", answer, re.I)
            if duration:
                answer = duration[1]
        if not re.fullmatch(r"[^\s?]+", answer):
            return None
        value: str | int | float = answer
        if field not in {"email", "available_from"}:
            if re.fullmatch(r"[+-]?[0-9]{1,15}", answer):
                value = int(answer)
            elif field == "cgpa" and re.fullmatch(r"[+-]?[0-9]{1,15}\.[0-9]{1,15}", answer):
                value = float(answer)
            elif not self.retry_field:
                return None
        self.on_progress("Validating your answer")
        result = validate_field(field, value)
        if result["valid"]:
            if field in CONFIRMATION_FIELDS:
                self.tools.validate_candidate_information({field: value}, {field: answer})
                if field in self.session.validated_values:
                    return self._confirmation_question()
            if field == "internship_duration_months":
                validated = self.tools.validate_candidate_information(
                    {field: value}, {field: answer}
                )
                if not validated["ok"]:
                    self.retry_field = field
                    return RETRY_QUESTIONS[field]
                saved = self.tools.execute(
                    "save_candidate_information", {"updates": {field: value}}
                )
                if not saved["ok"]:
                    self.stop("needs_human", "Could not save the internship duration safely.")
                    return self._outcome_message()
                self.retry_field = None
            return None
        self.retry_field = field
        self.session.validation_errors[field] = result["error"]
        self.on_decision("field_retry", field=field, reason="invalid_value")
        return f"{result['error']} {RETRY_QUESTIONS[field]}"

    def _confirmation_question(self) -> str:
        self.awaiting_confirmation = True
        self.on_decision(
            "confirmation_requested",
            fields=sorted(self.session.validated_values.keys() & CONFIRMATION_FIELDS),
        )
        labels = {"available_from": "start date", "internship_duration_months": "duration (months)"}
        values = ", ".join(
            f"{labels.get(key, key.replace('_', ' '))}: {value}"
            for key, value in self.session.validated_values.items()
        )
        return f"Please confirm {values}. Use /accept to save or /edit to make a correction."

    def _confirmation_decision(self, message: str) -> str:
        answer = message.strip().casefold()
        if answer in {"yes", "/accept"}:
            return "accept"
        if answer in {"no", "/edit"} or re.fullmatch(r"[0-9]+(?:-[0-9]+){0,2}", answer):
            return "edit"
        try:
            self.on_progress("Checking your confirmation")
            response = self.llm.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the reply to the displayed values. Accept only explicit, "
                            "unconditional agreement to all values. Use edit for any correction or "
                            "rejection, even if the reply begins with agreement. "
                            "Use unclear otherwise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "displayed_values": self.session.validated_values,
                                "reply": message,
                            }
                        ),
                    },
                ],
                format=CONFIRMATION_SCHEMA,
                stream=False,
                options={"temperature": 0, "num_predict": 48},
            )
            decision = json.loads(response.message.content or "").get("decision")
        except Exception:
            decision = "unclear"
        if decision not in ("accept", "edit"):
            decision = "unclear"
        self.on_decision("confirmation_classified", decision=decision)
        return decision

    def _say(self, message: str) -> str:
        self.session.history.append({"role": "assistant", "content": message})
        return message

    def _run_turn(self) -> str:
        session = self.session
        correction = ""
        for round_index in range(self.max_tool_rounds + 1):
            if (
                session.identity == "confirmed"
                and session.requested_outcome is None
                and not session.validated_values
                and self.tools.ready_to_complete()
            ):
                self.on_decision("completion_ready")
                result = self.stop("completed", "")
                if not result["ok"]:
                    return self._say(
                        "Your application details are saved, "
                        "but I couldn't verify the completion status."
                    )
                return self._say(self._outcome_message())
            inspected = any(
                entry["name"] == "inspect_candidate_state" and entry["result"].get("ok")
                for entry in session.tool_calls
            )
            available_tools = TOOL_SCHEMAS
            action = (
                "Choose the next useful question or tool from actual state and the latest reply. "
                f"Current missing fields: {', '.join(session.missing_fields) or 'none'}."
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
            elif self.retry_field:
                action = (
                    f"Resolve {self.retry_field} before asking about any other field. "
                    "Use validate_candidate_information if the latest reply supplies the value."
                )
            messages = deepcopy(session.history)
            messages[0]["content"] += (
                "\nAPPLICATION STATE: Identity confirmed. " + action + " " + correction
            )
            if session.validated_values or session.requested_outcome or self.retry_field:
                # Application control is never added to candidate evidence.
                messages.append(
                    {
                        "role": "user",
                        "content": "Application control, not candidate "
                        "input: " + action + ". Use the native tool, not a text response.",
                    }
                )
            try:
                self.on_decision(
                    "model_request",
                    round=round_index + 1,
                    available_tools=[schema["function"]["name"] for schema in available_tools],
                )
                self.on_progress("Preparing a reply")
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
                self.on_decision("model_replied_without_tools")
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
                if session.validated_values:
                    session.history.pop()
                    correction = "The operation is still pending. Emit a native tool call now."
                    continue
                if self.retry_field:
                    field = self.retry_field
                    error = session.validation_errors.get(field)
                    session.history.pop()
                    question = RETRY_QUESTIONS[field]
                    return self._say(f"{error} {question}" if error else question)
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
            if (
                session.requested_outcome is None
                and session.validated_values.keys() & CONFIRMATION_FIELDS
            ):
                return self._say(self._confirmation_question())
            if (
                self.retry_field in session.confirmed_values
                and self.retry_field not in session.validation_errors
            ):
                self.retry_field = None
            if session.disputed_fields:
                session.requested_outcome = None
                self.stop(
                    "needs_human", "Disputed fields: " + ", ".join(sorted(session.disputed_fields))
                )
                return self._say(self._outcome_message())
            if limited:
                self.on_decision("tool_limit_reached")
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
            self.session.confirmed_values
            or self.session.validated_values.keys() - CONFIRMATION_FIELDS
        ):
            status = "in_progress"
            notes = "Candidate left after providing information; retain saved progress."
        arguments = self.session.requested_outcome or {
            "collection_status": status,
            "notes": notes,
            "refused_fields": [],
        }
        result = self.tools.execute("finish_collection", arguments)
        self.on_decision("stop_result", status=arguments["collection_status"], ok=result["ok"])
        if not result["ok"]:
            self.session.last_error = result.get("error")
            self.session.outcome = "needs_human"
            self.session.ended = True
            self.session.outcome_persisted = False
        return result
