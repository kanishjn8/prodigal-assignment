import json
from uuid import UUID

import httpx
import pytest

from api_client import APIError, CandidateAPIClient, VersionConflict

RECORD = {
    "id": "c001",
    "version": 1,
    "updated_at": "2026-11-14T12:00:00+05:30",
    "email": "old@example.com",
}
SCHEMA = {
    "fields": [
        {"name": "email", "type": "string", "rules": "Valid email address", "writable": True}
    ],
    "mandatory_fields": ["email"],
}


@pytest.fixture
def make_client():
    clients = []

    def factory(handler, api_key="test-key"):
        client = CandidateAPIClient(
            api_key=api_key,
            base_url="https://candidate.test",
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("api_client.sleep", lambda _: None)


def test_schema_without_auth(make_client):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/schema"
        assert "Authorization" not in request.headers
        return httpx.Response(200, json=SCHEMA)

    assert make_client(handler, api_key="").get_schema() == SCHEMA


def test_list_and_get_preserve_metadata(make_client):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.url.path == "/v1/candidates":
            assert dict(request.url.params) == {"collection_status": "pending"}
            return httpx.Response(200, json={"candidates": [RECORD]})
        assert request.url.path == "/v1/candidates/c001"
        return httpx.Response(200, json=RECORD)

    client = make_client(handler)
    assert client.list_candidates("pending") == {"candidates": [RECORD]}
    assert client.get_candidate("c001") == RECORD


def test_list_without_filter(make_client):
    def handler(request):
        assert not request.url.params
        return httpx.Response(200, json={"candidates": []})

    assert make_client(handler).list_candidates() == {"candidates": []}


def test_patch_headers_delta_and_distinct_keys(make_client):
    keys = []
    updated = {**RECORD, "version": 2, "email": "new@example.com"}

    def handler(request):
        assert request.method == "PATCH"
        assert request.url.path == "/v1/candidates/c001"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["If-Match"] == '"1"'
        key = request.headers["Idempotency-Key"]
        assert UUID(key).version == 4
        keys.append(key)
        assert json.loads(request.content) == {"email": "new@example.com"}
        return httpx.Response(200, json=updated)

    client = make_client(handler)
    for _ in range(2):
        assert client.update_candidate("c001", {"email": "new@example.com"}, 1) == updated
    assert keys[0] != keys[1]


@pytest.mark.parametrize(
    "changes",
    [
        {},
        [],
        {"id": "c002"},
        {"phone": "+1234567890"},
        {"version": 2},
        {"updated_at": "now"},
        {"unknown": 1},
        {"email": "x", "phone": "x"},
        {"cgpa": float("nan")},
        {"skills": {"python"}},
    ],
)
def test_invalid_patch_rejected_locally(make_client, changes):
    client = make_client(lambda _: pytest.fail("Invalid input must not reach network"))
    with pytest.raises(ValueError):
        client.update_candidate("c001", changes, 1)


@pytest.mark.parametrize("version", [None, "1", True, 0, -1, 1.5])
def test_invalid_versions(make_client, version):
    client = make_client(lambda _: pytest.fail("Invalid version must not reach network"))
    with pytest.raises(ValueError):
        client.update_candidate("c001", {"notes": "test"}, version)


@pytest.mark.parametrize("candidate_id", [None, "", " ", "../schema", "c001?x=1", "a/b"])
def test_invalid_candidate_ids(make_client, candidate_id):
    client = make_client(lambda _: pytest.fail("Invalid ID must not reach network"))
    with pytest.raises(ValueError):
        client.get_candidate(candidate_id)


def test_invalid_filter(make_client):
    client = make_client(lambda _: pytest.fail("Invalid filter must not reach network"))
    with pytest.raises(ValueError):
        client.list_candidates("invalid")


def test_missing_key(make_client):
    client = make_client(lambda _: pytest.fail("Missing key must not reach network"), api_key=" ")
    with pytest.raises(APIError) as caught:
        client.get_candidate("c001")
    assert caught.value.kind == "missing_api_key"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (404, "not_found"),
        (422, "invalid_request"),
        (429, "rate_limit"),
        (500, "server"),
        (503, "server"),
    ],
)
def test_http_errors_are_safe_and_not_retried(make_client, status, kind):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text="secret test-key", headers={"Retry-After": "60"})

    with pytest.raises(APIError) as caught:
        make_client(handler).update_candidate("c001", {"notes": "test"}, 1)
    assert caught.value.kind == kind
    assert caught.value.status_code == status
    assert caught.value.idempotency_key
    assert "test-key" not in str(caught.value)
    assert len(requests) == (2 if status == 503 else 1)
    if status == 429:
        assert caught.value.retry_after == "60"


def test_conflict_refreshes_without_resending_write(make_client):
    methods = []
    latest = {**RECORD, "version": 2, "email": "newer@example.com"}

    def handler(request):
        methods.append(request.method)
        if request.method == "PATCH":
            return httpx.Response(409, json={"detail": "conflict"})
        return httpx.Response(200, json=latest)

    with pytest.raises(VersionConflict) as caught:
        make_client(handler).update_candidate("c001", {"email": "stale@example.com"}, 1)
    assert caught.value.latest_candidate == latest
    assert caught.value.refresh_error is None
    assert methods == ["PATCH", "GET"]


def test_conflict_preserved_when_refresh_fails(make_client):
    def handler(request):
        return httpx.Response(409 if request.method == "PATCH" else 503)

    with pytest.raises(VersionConflict) as caught:
        make_client(handler).update_candidate("c001", {"notes": "test"}, 1)
    assert caught.value.latest_candidate is None
    assert caught.value.refresh_error.kind == "server"


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
def test_patch_network_retry_reuses_exact_request(make_client, failure):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise failure("secret test-key")
        return httpx.Response(200, json={**RECORD, "version": 2})

    client = make_client(handler)
    assert client.update_candidate("c001", {"notes": "test"}, 1)["version"] == 2
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert requests[0].headers == requests[1].headers


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectError, "connection"),
    ],
)
@pytest.mark.parametrize("operation", ["get", "patch", "reset"])
def test_transport_failures_bounded_and_sanitized(make_client, failure, kind, operation):
    requests = []

    def handler(request):
        requests.append(request)
        raise failure("secret test-key")

    client = make_client(handler)
    with pytest.raises(APIError) as caught:
        if operation == "get":
            client.get_candidate("c001")
        elif operation == "patch":
            client.update_candidate("c001", {"notes": "test"}, 1)
        else:
            client.reset_workspace()
    assert caught.value.kind == kind
    assert "test-key" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert len(requests) == (2 if operation == "patch" else 1)
    if operation == "patch":
        assert caught.value.idempotency_key == requests[0].headers["Idempotency-Key"]
        assert requests[0].content == requests[1].content


@pytest.mark.parametrize("operation", ["schema", "list", "get", "patch", "reset"])
@pytest.mark.parametrize("payload", [None, [], {}, {"unexpected": "data"}, "not-json"])
def test_malformed_responses(make_client, operation, payload):
    def handler(_):
        if payload == "not-json":
            return httpx.Response(200, text="<html>not json</html>")
        return httpx.Response(200, content=json.dumps(payload))

    client = make_client(handler)
    # Reset's response contract is an unspecified object; preserve its actual keys.
    if operation == "reset" and payload == {"unexpected": "data"}:
        assert client.reset_workspace() == payload
        return
    operations = {
        "schema": client.get_schema,
        "list": client.list_candidates,
        "get": lambda: client.get_candidate("c001"),
        "patch": lambda: client.update_candidate("c001", {"notes": "test"}, 1),
        "reset": client.reset_workspace,
    }
    with pytest.raises(APIError) as caught:
        operations[operation]()
    assert caught.value.kind == "malformed_response"


def test_wrong_candidate_response(make_client):
    client = make_client(lambda _: httpx.Response(200, json={**RECORD, "id": "c002"}))
    with pytest.raises(APIError, match="metadata"):
        client.get_candidate("c001")


def test_reset_only_on_explicit_call(make_client):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/reset"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"reset": True})

    client = make_client(handler)
    assert not requests
    assert client.reset_workspace() == {"reset": True}
    assert len(requests) == 1


def test_redirect_does_not_forward_credentials(make_client):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://other.test/"})

    with pytest.raises(APIError):
        make_client(handler).get_candidate("c001")
    assert len(requests) == 1
