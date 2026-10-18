"""Terminal access to the same durable work used by SpiralChat."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

from spiral.host_client import HostClient, HostClientError, TERMINAL, start_body


def _connection_arguments(parser):
    # Suppressed defaults let connection flags appear before OR after the action.
    parser.add_argument("--host", default=argparse.SUPPRESS,
                        help="SpiralChat host or HTTPS gateway /spiral URL; defaults to SPIRALCHAT_HOST or loopback :8124")
    parser.add_argument("--token-file", default=argparse.SUPPRESS,
                        help="gateway token file; otherwise read SPIRALCHAT_TOKEN")
    parser.add_argument("--ca-cert", default=argparse.SUPPRESS,
                        help="trusted gateway certificate/CA PEM; otherwise SPIRALCHAT_CA_CERT")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit JSON (watch emits JSON lines)")


def add_arguments(subparsers):
    parser = subparsers.add_parser("session", help="share host-owned work with SpiralChat")
    _connection_arguments(parser)
    actions = parser.add_subparsers(dest="session_action", required=True)
    for name, helptext in (
        ("list", "list the host journal, including work started in SpiralChat"),
        ("start", "start work on the host, visible in the shared conversation"),
        ("show", "inspect current work and waits"),
        ("watch", "follow public events; Ctrl-C detaches without stopping the host"),
        ("message", "ask a question during work without issuing a direct revision"),
        ("steer", "explicitly revise active work"),
        ("cancel", "request cancellation and draining by the host"),
        ("tools", "inspect recorded tool results"),
    ):
        action = actions.add_parser(name, help=helptext)
        _connection_arguments(action)
        if name not in {"list", "start"}:
            action.add_argument("work_id")
        if name in {"start", "message", "steer", "cancel"}:
            action.add_argument("--request-id", default="",
                                help="reuse the same ID and arguments after an ambiguous POST outcome")
        if name == "list":
            action.add_argument("--conversation", default="")
            action.add_argument("--delivery-state", choices=("pending", "delivered"), default="",
                                help="filter terminal deliveries; enables oldest-first pagination")
            action.add_argument("--cursor", default="", help="next cursor from a delivery-filtered list")
            action.add_argument("--limit", type=int, default=50)
        elif name == "start":
            action.add_argument("objective")
            action.add_argument("--dir", default=os.getcwd(),
                                help="absolute workspace path on the host (default: current directory)")
            action.add_argument("--conversation", default="",
                                help="existing SpiralChat conversation ID; omitted creates a new conversation")
            action.add_argument("--model", default="", help="local model; omitted keeps the host default")
            action.add_argument("--read-only", action="store_true",
                                help="allow observation tools only; no shell, writes, downloads or child runs")
        elif name in {"message", "steer"}:
            action.add_argument("message")
        elif name == "cancel":
            action.add_argument("--reason", default="Stopped from the Spiral CLI")
        elif name == "watch":
            action.add_argument("--after", type=int, default=0, help="resume after this printed event cursor")
            action.add_argument("--once", action="store_true", help="read one event page and detach")
            action.add_argument("--interval", type=float, default=1.0, help="poll interval, 0.1–30 seconds")


def _safe(text) -> str:
    # Host text is data; never interpret Rich markup or terminal control sequences.
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", str(text))


def _print(console, text) -> None:
    console.print(_safe(text), markup=False, highlight=False)


def _output(console, value: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=True), flush=True)
        return
    if "turns" in value:
        for turn in value["turns"]:
            _print(console, f"{turn['id']}  {turn['status']}  {turn.get('objective', '')[:100]}")
            _print(console, f"  conversation: {turn.get('conversation_id', '')}")
        if value.get("next_cursor"):
            _print(console, "Next page: --cursor " + value["next_cursor"])
        return
    turn = value.get("turn")
    if isinstance(turn, dict):
        _print(console, f"{turn['id']} · {turn['status']}")
        _print(console, "conversation: " + str(turn.get("conversation_id", "")))
        _print(console, turn.get("objective", ""))
        outcome = turn.get("outcome")
        if isinstance(outcome, dict) and isinstance(outcome.get("message"), str):
            _print(console, outcome["message"])
        if turn.get("error"):
            _print(console, turn["error"])
        runtime = value.get("runtime") or {}
        if any(runtime.get(key) for key in (
                "attention", "permission_request", "budget_request", "clarification_question")):
            _print(console, json.dumps(runtime, ensure_ascii=True, indent=2))
        if value.get("dialogue"):
            _print(console, "Message: " + str(value["dialogue"].get("status", "accepted")))
    else:
        _print(console, json.dumps(value, ensure_ascii=True, indent=2))


def watch(client, args, console, *, sleep=time.sleep) -> int:
    if not 0.1 <= args.interval <= 30:
        raise HostClientError("poll interval must be between 0.1 and 30 seconds")
    cursor = args.after
    as_json = getattr(args, "json", False)
    while True:
        page = client.events(args.work_id, after=cursor)
        for event in page["events"]:
            if as_json:
                print(json.dumps({"type": "event", "event": event}, ensure_ascii=True), flush=True)
            else:
                # Token deltas remain available in JSON; human output uses public
                # activity and the authoritative final message rather than a
                # misleading transcript assembled across model retry/reset events.
                if event.get("type") == "assistant_turn.response_delta":
                    continue
                _print(console, f"{event['seq']}  {event.get('type', 'event')}")
                # The host's public projection excludes raw receipts/private reasoning.
                for item in (event.get("payload") or {}).get("progress") or []:
                    _print(console, "  " + str(item.get("message") or item.get("title") or ""))
        cursor = page["next_seq"]
        turn, runtime = page.get("turn") or {}, page.get("runtime") or {}
        waiting = bool(runtime.get("attention") or runtime.get("permission_request")
                       or runtime.get("budget_request") or runtime.get("clarification_question"))
        if args.once or not page["has_more"] and (turn.get("status") in TERMINAL or waiting):
            if as_json:
                print(json.dumps({"type": "state", "cursor": cursor, "turn": turn,
                                  "runtime": runtime}, ensure_ascii=True), flush=True)
            else:
                _output(console, page, as_json=False)
                _print(console, f"Event cursor: {cursor}; use watch --after {cursor} to reconnect.")
            return 4 if turn.get("status") in {"failed", "interrupted"} else 0
        # Drain paginated events before waiting or treating a terminal snapshot as done.
        if not page["has_more"]:
            sleep(args.interval)


def run(args, console, *, client_factory=HostClient) -> int:
    try:
        token_file = getattr(args, "token_file", "")
        token = (Path(token_file).expanduser().read_text().strip() if token_file
                 else os.environ.get("SPIRALCHAT_TOKEN", ""))
        host = getattr(args, "host", os.environ.get("SPIRALCHAT_HOST", "http://127.0.0.1:8124"))
        ca_cert = getattr(args, "ca_cert", os.environ.get("SPIRALCHAT_CA_CERT", ""))
        with client_factory(host, token=token, ca_cert=ca_cert) as client:
            action = args.session_action
            if action == "watch":
                return watch(client, args, console)
            if action == "list":
                result = client.list(conversation_id=args.conversation, cursor=args.cursor, limit=args.limit,
                                     delivery_state=args.delivery_state)
            elif action == "show":
                result = client.get(args.work_id)
            elif action == "tools":
                result = client.tools(args.work_id)
            else:
                request_id = args.request_id or "cli-" + uuid.uuid4().hex
                # This reaches the user before any ambiguous network mutation. The
                # host validates exact-payload replay, including after task completion.
                print("Request ID: " + _safe(request_id), file=sys.stderr, flush=True)
                if action == "start":
                    result = client.start(start_body(
                        args.objective, request_id=request_id, workspace=args.dir,
                        conversation_id=args.conversation, model=args.model, read_only=args.read_only))
                elif action == "message":
                    result = client.message(args.work_id, args.message, request_id=request_id)
                else:
                    result = client.direct(args.work_id, action,
                                           args.message if action == "steer" else args.reason,
                                           request_id=request_id)
                result = {**result, "client_request_id": request_id}
            _output(console, result, as_json=getattr(args, "json", False))
            return 0
    except KeyboardInterrupt:
        print("Detached. Host work continues; use session cancel to stop it.", file=sys.stderr)
        return 130
    except (HostClientError, OSError, ValueError) as exc:
        print("SpiralChat connection: " + _safe(exc), file=sys.stderr)
        return 2
