"""Client for SpiralChat's existing host-owned work journal.

The host remains the only executor, journal and inference owner. This module
does not lease work, replay effects, acknowledge another client's delivery, or
maintain a second conversation database.
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import ssl
from urllib.parse import quote, urlsplit

import httpx

PROTOCOL_VERSION = 2
ASSISTANT_PATH = "/v2/assistant/turns"
TERMINAL = frozenset({"completed", "failed", "interrupted"})
READ_TOOLS = (
    "environment.discover", "web.search", "network.fetch",
    "workspace.list", "workspace.stat", "workspace.read", "workspace.search",
)
WORK_TOOLS = READ_TOOLS + (
    "spiral.run", "workspace.write", "workspace.command", "network.download",
)
_ID = re.compile(r"[A-Za-z0-9._:@+\-=]{1,128}\Z")
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class HostClientError(RuntimeError):
    """A connection, protocol or host rejection; never proof of task failure."""


def identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise HostClientError("IDs must contain 1–128 letters, digits or ._:@+-= characters")
    return value


def _segment(value: str) -> str:
    value = identifier(value)
    if value in {".", ".."}:
        raise HostClientError("work ID cannot be a URL dot segment")
    # The production handler matches these literal protocol characters before
    # percent decoding. None can introduce a path separator, query or fragment.
    return quote(value, safe=":@+-=")


def _text(value: str, maximum: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise HostClientError(f"{field} must contain 1–{maximum} characters")
    return value


def start_body(objective: str, *, request_id: str, workspace: str,
               conversation_id: str = "", model: str = "", read_only: bool = False) -> dict:
    """Deterministic identity and policy, including across an ambiguous POST retry."""
    identifier(request_id)
    _text(objective, 64 * 1024, "objective")
    # This is a path on the HOST. Do not resolve or inspect it on a remote client.
    if not isinstance(workspace, str) or not workspace.startswith("/") or "\x00" in workspace:
        raise HostClientError("workspace must be an absolute path on the SpiralChat host")
    digest = hashlib.sha256(request_id.encode()).hexdigest()[:32]
    conversation_id = identifier(conversation_id or "spiral-cli-" + digest)
    execution = {
        "mode": "host", "model_adapter": "ollama",
        "allowed_tools": list(READ_TOOLS if read_only else WORK_TOOLS),
        "completion_policy": "response_or_tool",
        "run_policy": {
            "allowed_kinds": ["build", "research", "research_build"],
            "access": "workspace", "project_path": workspace, "create_project": False,
        },
    }
    if model:
        execution["model"] = _text(model, 256, "model")
    return {
        "conversation_id": conversation_id, "turn_id": "cli-turn-" + digest,
        "request_id": request_id, "objective": objective,
        "input": {"message_id": "cli-input-" + digest,
                  "output_message_id": "cli-output-" + digest, "text": objective},
        "context": {"client_mode": "host_runtime", "client_intent": "task",
                    "client_protocol": PROTOCOL_VERSION, "messages": []},
        "execution": execution,
    }


class HostClient:
    def __init__(self, host: str, *, token: str = "", ca_cert: str = "",
                 transport: httpx.BaseTransport | None = None):
        parsed = urlsplit(host)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise HostClientError("host must be an HTTP(S) base URL without credentials, query or fragment")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if parsed.scheme == "http" and not loopback:
            raise HostClientError("remote hosts require HTTPS; HTTP is supported only on loopback")
        if "\r" in token or "\n" in token:
            raise HostClientError("host token contains an invalid newline")
        verify = ssl.create_default_context(cafile=ca_cert) if ca_cert else True
        self._token = token
        self._client = httpx.Client(
            base_url=host.rstrip("/") + "/", verify=verify, transport=transport,
            headers={"Authorization": "Bearer " + token} if token else {},
            timeout=15, follow_redirects=False, trust_env=False,
        )
        self.manifest: dict | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, *, body=None, params=None) -> dict:
        # Relative paths preserve gateway prefixes such as /spiral. Never follow
        # redirects with credentials or automatically repeat a mutating request.
        try:
            with self._client.stream(method, path.lstrip("/"), json=body, params=params) as response:
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise HostClientError("host response exceeds the 8 MiB limit")
                try:
                    import json
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise HostClientError(f"host returned invalid JSON (HTTP {response.status_code})") from exc
                if not isinstance(value, dict):
                    raise HostClientError("host returned a non-object response")
                if not 200 <= response.status_code < 300:
                    detail = str(value.get("error") or "request rejected")[:1000]
                    if self._token:
                        detail = detail.replace(self._token, "[redacted]")
                    raise HostClientError(f"host returned HTTP {response.status_code}: {detail}")
                return value
        except httpx.HTTPError as exc:
            # Do not print request headers or an arbitrary transport exception.
            suffix = (" The POST outcome is unknown; inspect the host and reuse the same request ID."
                      if method == "POST" else "")
            raise HostClientError(f"host connection failed ({type(exc).__name__}).{suffix}") from exc

    def connect(self, *, execute: bool = False) -> dict:
        if self.manifest is None:
            value = self._request("GET", "/v2/manifest")
            if (value.get("protocol") != "spiral-v2"
                    or type(value.get("protocol_version")) is not int
                    or value["protocol_version"] != PROTOCOL_VERSION
                    or type(value.get("minimum_client_protocol")) is not int
                    or not 1 <= value["minimum_client_protocol"] <= PROTOCOL_VERSION):
                raise HostClientError("host does not support the SpiralChat v2 client protocol")
            capabilities = value.get("capabilities")
            if not isinstance(capabilities, list) or not all(isinstance(x, str) for x in capabilities):
                raise HostClientError("host manifest has invalid capabilities")
            if "durable_assistant_turns" not in capabilities:
                raise HostClientError("host does not advertise durable assistant work")
            self.manifest = value
        if execute and "host_assistant_runtime" not in self.manifest["capabilities"]:
            raise HostClientError("host assistant runtime is unavailable; no work was submitted")
        return self.manifest

    def list(self, *, conversation_id: str = "", cursor: str = "", limit: int = 50,
             delivery_state: str = "") -> dict:
        self.connect()
        if type(limit) is not int or not 1 <= limit <= 100:
            raise HostClientError("limit must be between 1 and 100")
        if delivery_state not in {"", "pending", "delivered"}:
            raise HostClientError("delivery state must be pending or delivered")
        if cursor and not delivery_state:
            raise HostClientError("list cursors require --delivery-state pending or delivered")
        params = {"limit": limit}
        if delivery_state:
            params["delivery_state"] = delivery_state
        if conversation_id:
            params["conversation_id"] = identifier(conversation_id)
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", ASSISTANT_PATH, params=params)

    def get(self, work_id: str) -> dict:
        self.connect()
        return self._request("GET", ASSISTANT_PATH + "/" + _segment(work_id))

    def events(self, work_id: str, *, after: int = 0) -> dict:
        self.connect()
        if type(after) is not int or not 0 <= after <= 2**63 - 1:
            raise HostClientError("event cursor must be a non-negative signed 64-bit integer")
        value = self._request("GET", ASSISTANT_PATH + "/" + _segment(work_id) + "/events",
                              params={"after": after})
        events, cursor = value.get("events"), value.get("next_seq")
        if (not isinstance(events, list) or type(cursor) is not int
                or not after <= cursor <= 2**63 - 1 or type(value.get("has_more")) is not bool):
            raise HostClientError("host returned an invalid event page")
        previous = after
        for event in events:
            seq = event.get("seq") if isinstance(event, dict) else None
            if type(seq) is not int or not previous < seq <= cursor:
                raise HostClientError("host returned unordered or replayed events")
            previous = seq
        if cursor != previous or value["has_more"] and cursor == after:
            raise HostClientError("host returned an event cursor without corresponding events")
        return value

    def tools(self, work_id: str) -> dict:
        self.connect()
        if "assistant_tool_inspection_v1" not in self.manifest["capabilities"]:
            raise HostClientError("host does not support tool inspection")
        return self._request("GET", ASSISTANT_PATH + "/" + _segment(work_id) + "/tools")

    def start(self, body: dict) -> dict:
        self.connect(execute=True)
        return self._request("POST", ASSISTANT_PATH, body=body)

    def message(self, work_id: str, message: str, *, request_id: str) -> dict:
        self.connect(execute=True)
        if "assistant_dialogue" not in self.manifest["capabilities"]:
            raise HostClientError("host does not support in-task dialogue")
        return self._request("POST", ASSISTANT_PATH + "/" + _segment(work_id) + "/messages",
                             body={"request_id": identifier(request_id),
                                   "message": _text(message, 8192, "message")})

    def direct(self, work_id: str, kind: str, message: str, *, request_id: str) -> dict:
        if kind not in {"steer", "cancel"}:
            raise HostClientError("directive must be steer or cancel")
        self.connect(execute=True)
        body = {"request_id": identifier(request_id)}
        body["message" if kind == "steer" else "reason"] = _text(
            message, 65536 if kind == "steer" else 4096, kind)
        return self._request("POST", ASSISTANT_PATH + "/" + _segment(work_id) + "/" + kind,
                             body=body)
