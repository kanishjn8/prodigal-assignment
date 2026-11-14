import json
import re
from time import sleep
from typing import Any
from uuid import uuid4

import httpx

import config

WRITABLE_FIELDS = frozenset(
    {
        "full_name",
        "email",
        "college",
        "degree",
        "branch",
        "graduation_year",
        "cgpa",
        "preferred_location",
        "available_from",
        "internship_duration_months",
        "linkedin_url",
        "github_url",
        "skills",
        "collection_status",
        "attempt_count",
        "last_contacted_at",
        "notes",
    }
)
COLLECTION_STATUSES = frozenset(
    {
        "pending",
        "in_progress",
        "completed",
        "no_answer",
        "refused",
        "needs_human",
    }
)


class APIError(Exception):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.idempotency_key = idempotency_key


class VersionConflict(APIError):
    def __init__(
        self,
        latest_candidate: dict[str, Any] | None,
        refresh_error: APIError | None = None,
    ) -> None:
        super().__init__(
            "conflict", "Candidate changed; reconcile before writing again.", status_code=409
        )
        self.latest_candidate = latest_candidate
        self.refresh_error = refresh_error


def _candidate_path(candidate_id: str) -> str:
    if not isinstance(candidate_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id):
        raise ValueError("A nonempty candidate ID containing letters, digits, _ or - is required")
    return f"/v1/candidates/{candidate_id}"


def _check_record(data: Any, candidate_id: str | None = None) -> dict[str, Any]:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("id"), str)
        or not data["id"]
        or type(data.get("version")) is not int
        or data["version"] < 1
        or not isinstance(data.get("updated_at"), str)
        or not data["updated_at"]
        or (candidate_id is not None and data["id"] != candidate_id)
    ):
        raise APIError("malformed_response", "Invalid candidate response or record metadata.")
    return data


class CandidateAPIClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = config.CANDIDATE_API_KEY if api_key is None else api_key
        self._http = httpx.Client(
            base_url=base_url or config.CANDIDATE_API_BASE_URL,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> "CandidateAPIClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        protected: bool = True,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> Any:
        request_headers = {"Accept": "application/json", **(headers or {})}
        if protected:
            if not self._api_key.strip():
                raise APIError("missing_api_key", "Set CANDIDATE_API_KEY for protected routes.")
            request_headers["Authorization"] = f"Bearer {self._api_key}"
        attempts = 2 if method == "PATCH" else 1
        for attempt in range(attempts):
            try:
                response = self._http.request(
                    method,
                    path,
                    headers=request_headers,
                    params=params,
                    content=body,
                )
                break
            except httpx.TransportError as exc:
                if attempt + 1 < attempts:
                    # An ambiguous PATCH retry must retain the exact bytes, version and key.
                    sleep(0.25)
                    continue
                kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection"
                message = (
                    "API request timed out." if kind == "timeout" else "API connection failed."
                )
                if method != "GET":
                    message += " Write outcome is unknown; do not start a new write blindly."
                raise APIError(
                    kind,
                    message,
                    idempotency_key=request_headers.get("Idempotency-Key"),
                ) from None

        if not response.is_success:
            descriptions = {
                401: ("authentication", "API authentication failed."),
                403: ("authentication", "API access denied."),
                404: ("not_found", "Candidate or API route not found."),
                409: ("conflict", "Candidate version conflict."),
                422: ("invalid_request", "API rejected the request structure."),
                429: ("rate_limit", "API rate limit reached; defer further requests."),
            }
            kind, message = descriptions.get(
                response.status_code,
                (
                    "server" if response.status_code >= 500 else "http",
                    f"API returned HTTP {response.status_code}.",
                ),
            )
            # Do not echo response bodies or transport errors: they can contain sensitive data.
            raise APIError(
                kind,
                message,
                status_code=response.status_code,
                retry_after=response.headers.get("Retry-After"),
                idempotency_key=request_headers.get("Idempotency-Key"),
            )
        try:
            return response.json()
        except ValueError:
            raise APIError(
                "malformed_response",
                "API returned a non-JSON response.",
                idempotency_key=request_headers.get("Idempotency-Key"),
            ) from None

    def get_schema(self) -> dict[str, Any]:
        data = self._request("GET", "/v1/schema", protected=False)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("fields"), list)
            or not data["fields"]
            or any(
                not isinstance(field, dict)
                or not all(isinstance(field.get(key), str) for key in ("name", "type", "rules"))
                or type(field.get("writable")) is not bool
                for field in data["fields"]
            )
        ):
            raise APIError("malformed_response", "API returned invalid field definitions.")
        return data

    def list_candidates(self, collection_status: str | None = None) -> dict[str, Any]:
        if collection_status is not None and collection_status not in COLLECTION_STATUSES:
            raise ValueError("Unknown collection_status filter")
        params = {"collection_status": collection_status} if collection_status is not None else None
        data = self._request("GET", "/v1/candidates", params=params)
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            raise APIError("malformed_response", "API returned an invalid candidate list.")
        for record in data["candidates"]:
            _check_record(record)
        return data

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        data = self._request("GET", _candidate_path(candidate_id))
        return _check_record(data, candidate_id)

    def update_candidate(
        self,
        candidate_id: str,
        changes: dict[str, Any],
        version: int,
    ) -> dict[str, Any]:
        """Send a caller-computed delta; field values must be validated before calling."""
        path = _candidate_path(candidate_id)
        if not isinstance(changes, dict) or not changes:
            raise ValueError("PATCH changes must be a nonempty dictionary")
        if not changes.keys() <= WRITABLE_FIELDS:
            raise ValueError("PATCH contains unknown or read-only fields")
        if type(version) is not int or version < 1:
            raise ValueError("version must be a positive integer from the candidate record")
        try:
            body = json.dumps(changes, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("PATCH changes must contain JSON-serializable values") from None
        key = str(uuid4())
        headers = {
            "Content-Type": "application/json",
            "If-Match": f'"{version}"',
            "Idempotency-Key": key,
        }
        try:
            data = self._request("PATCH", path, headers=headers, body=body)
            return _check_record(data, candidate_id)
        except APIError as exc:
            if exc.status_code == 409:
                try:
                    latest = self.get_candidate(candidate_id)
                except APIError as refresh_error:
                    conflict = VersionConflict(None, refresh_error)
                else:
                    conflict = VersionConflict(latest)
                conflict.idempotency_key = key
                raise conflict from None
            exc.idempotency_key = key
            raise

    def reset_workspace(self) -> dict[str, Any]:
        """Destructively reset the workspace only when explicitly requested for testing."""
        data = self._request("POST", "/v1/reset")
        if not isinstance(data, dict) or not data:
            raise APIError("malformed_response", "API returned an invalid reset response.")
        return data
