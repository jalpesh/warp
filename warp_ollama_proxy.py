"""
Warp → Ollama proxy.

Intercepts the agent-mode SSE endpoint (/ai/multi-agent) and routes it to
a local Ollama instance, synthesising the exact protobuf SSE wire format
that the Warp client expects.

Field tags verified against the generated Rust types in:
  target/release/build/warp_multi_agent_api-.../out/warp.multi_agent.v1.rs

Key facts:
  ResponseEvent (oneof type):
    field 1 = init        (StreamInit)
    field 2 = client_actions (ClientActions)
    field 3 = finished    (StreamFinished)

  StreamInit:
    field 1 = conversation_id (string)
    field 2 = request_id      (string)
    field 3 = run_id          (string)

  ClientActions:
    field 1 = actions (repeated ClientAction)

  ClientAction (oneof action):
    field 1  = create_task              (CreateTask)
    field 3  = add_messages_to_task     (AddMessagesToTask)
    field 5  = append_to_message_content (AppendToMessageContent)
    field 9  = begin_transaction        (BeginTransaction – empty)
    field 10 = commit_transaction       (CommitTransaction – empty)

  CreateTask:
    field 1 = task (Task)

  Task:
    field 1 = id          (string)
    field 2 = description (string)
    field 5 = messages    (repeated Message)

  AddMessagesToTask:
    field 1 = task_id   (string)
    field 2 = messages  (repeated Message)

  AppendToMessageContent:
    field 1 = message (Message)
    field 2 = mask    (FieldMask)   ← google.protobuf.FieldMask { paths: ["agent_output.text"] }
    field 3 = task_id (string)

  Message:
    field 1  = id           (string)
    field 11 = task_id      (string)
    oneof message:
      field 3 = agent_output (AgentOutput)

  AgentOutput:
    field 1 = text (string)

  StreamFinished (oneof reason):
    field 2 = done (Done – empty)

  FieldMask (google.protobuf.FieldMask):
    field 1 = paths (repeated string)
"""

import base64
import json
import logging
import os
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warp_proxy")

app = FastAPI()

WARP_BACKEND = os.environ.get("WARP_BACKEND", "https://app.warp.dev")
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ---------------------------------------------------------------------------
# Minimal, correct protobuf encoder
# ---------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    out = []
    while True:
        bits = n & 0x7F
        n >>= 7
        if n:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            break
    return bytes(out)

def _tag(field: int, wire: int) -> bytes:
    """Encode a protobuf tag (field_number << 3 | wire_type)."""
    return _varint((field << 3) | wire)

WIRE_VARINT = 0
WIRE_BYTES  = 2  # length-delimited (strings, embedded messages, bytes)

def _str_field(field: int, s: str) -> bytes:
    """Encode a string field."""
    encoded = s.encode("utf-8")
    return _tag(field, WIRE_BYTES) + _varint(len(encoded)) + encoded

def _msg_field(field: int, payload: bytes) -> bytes:
    """Encode an embedded message field."""
    return _tag(field, WIRE_BYTES) + _varint(len(payload)) + payload

# ---------------------------------------------------------------------------
# Build ResponseEvent messages
# ---------------------------------------------------------------------------

def _make_stream_init(conv_id: str, req_id: str) -> bytes:
    """ResponseEvent { init: StreamInit { … } }"""
    init_payload = (
        _str_field(1, conv_id) +
        _str_field(2, req_id) +
        _str_field(3, req_id)   # run_id same as request_id for simplicity
    )
    return _msg_field(1, init_payload)   # ResponseEvent.init = field 1


def _client_actions_event(action_payloads: list) -> bytes:
    """ResponseEvent { client_actions: ClientActions { actions: [...] } }"""
    # Each action_payload is the raw bytes of a ClientAction message
    actions_bytes = b"".join(
        _msg_field(1, ap) for ap in action_payloads   # ClientActions.actions = field 1 (repeated)
    )
    return _msg_field(2, actions_bytes)   # ResponseEvent.client_actions = field 2


def _begin_transaction_action() -> bytes:
    """ClientAction { begin_transaction: BeginTransaction{} }"""
    return _msg_field(9, b"")   # ClientAction.begin_transaction = field 9, empty msg


def _commit_transaction_action() -> bytes:
    """ClientAction { commit_transaction: CommitTransaction{} }"""
    return _msg_field(10, b"")  # ClientAction.commit_transaction = field 10, empty msg


def _make_message(message_id: str, task_id: str, text: str) -> bytes:
    """
    Message {
      id = message_id           (field 1)
      task_id = task_id         (field 11)
      message = AgentOutput {   (field 3)
        text = text             (field 1)
      }
    }
    """
    agent_output = _str_field(1, text)           # AgentOutput.text = field 1
    return (
        _str_field(1, message_id) +              # Message.id = field 1
        _str_field(11, task_id) +                # Message.task_id = field 11
        _msg_field(3, agent_output)              # Message.agent_output = field 3
    )


def _make_field_mask(*paths: str) -> bytes:
    """google.protobuf.FieldMask { paths: [...] }"""
    return b"".join(_str_field(1, p) for p in paths)   # FieldMask.paths = field 1 repeated


def _create_task_action(task_id: str, message_id: str) -> bytes:
    """
    ClientAction { create_task: CreateTask { task: Task { id, messages: [Message] } } }
    """
    message_bytes = _make_message(message_id, task_id, "")
    task_bytes = (
        _str_field(1, task_id) +                # Task.id = field 1
        _msg_field(5, message_bytes)             # Task.messages = field 5 (repeated)
    )
    create_task = _msg_field(1, task_bytes)      # CreateTask.task = field 1
    return _msg_field(1, create_task)            # ClientAction.create_task = field 1


def _append_text_action(task_id: str, message_id: str, text: str) -> bytes:
    """
    ClientAction {
      append_to_message_content: AppendToMessageContent {
        message: Message { id, task_id, agent_output: AgentOutput { text } }
        mask:    FieldMask { paths: ["agent_output.text"] }
        task_id: task_id
      }
    }
    """
    msg_bytes = _make_message(message_id, task_id, text)
    mask_bytes = _make_field_mask("agent_output.text")
    append_bytes = (
        _msg_field(1, msg_bytes) +              # AppendToMessageContent.message = field 1
        _msg_field(2, mask_bytes) +             # AppendToMessageContent.mask = field 2
        _str_field(3, task_id)                  # AppendToMessageContent.task_id = field 3
    )
    return _msg_field(5, append_bytes)           # ClientAction.append_to_message_content = field 5


def _make_stream_finished_done() -> bytes:
    """ResponseEvent { finished: StreamFinished { done: Done{} } }"""
    done_bytes = _msg_field(2, b"")             # StreamFinished.done = field 2, empty
    return _msg_field(3, done_bytes)             # ResponseEvent.finished = field 3


def _to_sse(proto_bytes: bytes) -> str:
    """Encode proto bytes as a base64-quoted SSE data line."""
    b64 = base64.urlsafe_b64encode(proto_bytes).decode("ascii")
    return f'data: "{b64}"\n\n'


# ---------------------------------------------------------------------------
# Ollama streaming
# ---------------------------------------------------------------------------

async def _stream_ollama(messages: list):
    """
    Call Ollama /api/chat (streaming) and yield SSE data lines containing
    valid ResponseEvent protobuf messages.
    """
    conversation_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    # 1. StreamInit
    yield _to_sse(_make_stream_init(conversation_id, request_id))

    # 2. BeginTransaction
    yield _to_sse(_client_actions_event([_begin_transaction_action()]))

    # 3. CreateTask with empty initial message
    yield _to_sse(_client_actions_event([_create_task_action(task_id, message_id)]))

    # 4. Stream text from Ollama
    total_text = ""
    error_occurred = False
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        total_text += delta
                        yield _to_sse(_client_actions_event(
                            [_append_text_action(task_id, message_id, delta)]
                        ))
                    if chunk.get("done"):
                        break
    except Exception as e:
        logger.error(f"Ollama streaming error: {e}")
        error_text = f"\n\n[Local Ollama error: {e}]"
        yield _to_sse(_client_actions_event(
            [_append_text_action(task_id, message_id, error_text)]
        ))
        error_occurred = True

    logger.info(f"Ollama stream complete – {len(total_text)} chars, error={error_occurred}")

    # 5. CommitTransaction
    yield _to_sse(_client_actions_event([_commit_transaction_action()]))

    # 6. StreamFinished / Done
    yield _to_sse(_make_stream_finished_done())


# ---------------------------------------------------------------------------
# Proto body parser – extract user query text from the Request protobuf
# ---------------------------------------------------------------------------

def _extract_query(body: bytes) -> str:
    """
    Best-effort extraction of the user query string from the binary protobuf
    Request body.  We do a depth-first scan for length-delimited fields
    (wire type 2) and collect UTF-8 decodable strings longer than 5 chars.
    The longest one is almost always the user's query.
    """
    def _scan(data: bytes, depth=0):
        strings = []
        i = 0
        while i < len(data):
            # Read tag varint
            tag = 0
            shift = 0
            while i < len(data):
                b = data[i]; i += 1
                tag |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
            wire = tag & 0x07
            # field = tag >> 3
            if wire == 2:  # length-delimited
                length = 0
                shift2 = 0
                while i < len(data):
                    b = data[i]; i += 1
                    length |= (b & 0x7F) << shift2
                    shift2 += 7
                    if not (b & 0x80):
                        break
                if length < 0 or i + length > len(data):
                    break
                payload = data[i: i + length]
                i += length
                # Try as UTF-8 string
                try:
                    s = payload.decode("utf-8")
                    if len(s) > 5 and s.isprintable():
                        strings.append(s)
                except Exception:
                    pass
                # Recurse into sub-messages (if depth < 6)
                if depth < 6:
                    strings.extend(_scan(payload, depth + 1))
            elif wire == 0:  # varint
                while i < len(data) and (data[i] & 0x80):
                    i += 1
                i += 1
            elif wire == 1:  # 64-bit
                i += 8
            elif wire == 5:  # 32-bit
                i += 4
            else:
                break
        return strings

    try:
        candidates = _scan(body)
        if candidates:
            # Return the longest printable string (most likely the query)
            return max(candidates, key=len)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Generic transparent proxy
# ---------------------------------------------------------------------------

async def _forward(request: Request, path: str):
    headers = dict(request.headers)
    headers.pop("host", None)
    body = await request.body()
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{WARP_BACKEND}/{path}"
        if request.url.query:
            url += "?" + request.url.query
        warp_req = client.build_request(request.method, url, headers=headers, content=body)
        logger.info(f"Forwarding {request.method} /{path} → Warp")
        warp_resp = await client.send(warp_req, stream=True)
    return StreamingResponse(
        warp_resp.aiter_raw(),
        status_code=warp_resp.status_code,
        headers=dict(warp_resp.headers),
        background=BackgroundTask(warp_resp.aclose),
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_all(request: Request, path: str):

    # ── Agent-mode SSE endpoint ─────────────────────────────────────────────
    if path in ("ai/multi-agent", "ai/passive-suggestions") and request.method == "POST":
        logger.info(f"Intercepting /{path} → local Ollama ({OLLAMA_MODEL})")
        body = await request.body()
        query = _extract_query(body)
        logger.info(f"Query extracted ({len(query)} chars): {query[:120]!r}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding and terminal assistant built into the Warp terminal. "
                    "You are running locally via Ollama. Help the user with their shell, code, and "
                    "file-system tasks. Be concise and practical."
                ),
            },
            {"role": "user", "content": query or "(User sent an agent request)"},
        ]
        return StreamingResponse(
            _stream_ollama(messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── GraphQL: intercept AI dialogue / commands ───────────────────────────
    if path == "graphql/v2" and request.method == "POST":
        body = await request.body()
        try:
            body_json = json.loads(body)
            op = body_json.get("operationName", "")
            if not op:
                q = body_json.get("query", "").lower()
                if "generatedialogue" in q:
                    op = "GenerateDialogue"
                elif "generatecommands" in q:
                    op = "GenerateCommands"
                elif "getfeaturemodelchoices" in q or "freeavailablemodels" in q:
                    op = "GetModels"

            if op == "GenerateDialogue":
                return await _handle_dialogue(body_json)
            if op == "GenerateCommands":
                return await _handle_commands(body_json)
            if op == "GetModels":
                return await _handle_get_models(request, body)
        except json.JSONDecodeError:
            pass

    return await _forward(request, path)


# ---------------------------------------------------------------------------
# GraphQL handlers (legacy panel mode)
# ---------------------------------------------------------------------------

async def _handle_dialogue(body_json: dict):
    variables = body_json.get("variables", {})
    inp = variables.get("input", {})
    prompt = inp.get("prompt", "")
    transcript = inp.get("transcript", [])
    messages = [{"role": "system", "content": "You are a helpful coding and terminal assistant built into Warp. Running locally via Ollama."}]
    for part in transcript:
        if part.get("user"):
            messages.append({"role": "user", "content": part["user"]})
        if part.get("assistant"):
            messages.append({"role": "assistant", "content": part["assistant"]})
    messages.append({"role": "user", "content": prompt})
    try:
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_MODEL, "messages": messages, "stream": False})
            r.raise_for_status()
            answer = r.json().get("message", {}).get("content", "")
    except Exception as e:
        answer = f"Ollama error: {e}"
    return JSONResponse({"data": {"generateDialogue": {
        "__typename": "GenerateDialogueOutput",
        "status": {
            "__typename": "GenerateDialogueSuccess",
            "answer": answer,
            "requestLimitInfo": {"isUnlimited": True, "nextRefreshTime": "2099-01-01T00:00:00Z", "requestLimit": 9999, "requestsUsedSinceLastRefresh": 0},
            "transcriptSummarized": False,
            "truncated": False,
        },
        "responseContext": {"serverVersion": "ollama-proxy"},
    }}})


async def _handle_commands(body_json: dict):
    prompt = body_json.get("variables", {}).get("input", {}).get("prompt", "")
    messages = [
        {"role": "system", "content": "You are a shell command generator. Reply with ONLY the shell command, no markdown, no explanation."},
        {"role": "user", "content": prompt},
    ]
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_MODEL, "messages": messages, "stream": False})
            r.raise_for_status()
            answer = r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        answer = f"echo 'Ollama error: {e}'"
    return JSONResponse({"data": {"generateCommands": {
        "__typename": "GenerateCommandsOutput",
        "status": {
            "__typename": "GenerateCommandsSuccess",
            "commands": [{"command": answer, "description": "Generated locally via Ollama", "parameters": []}],
        },
        "responseContext": {"serverVersion": "ollama-proxy"},
    }}})


async def _handle_get_models(request: Request, body_bytes: bytes):
    import copy
    headers = dict(request.headers)
    headers.pop("host", None)
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{WARP_BACKEND}/graphql/v2", headers=headers, content=body_bytes)
            data = r.json()
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return JSONResponse({"data": {}})

    def inject(node):
        if isinstance(node, dict):
            if "choices" in node and isinstance(node["choices"], list) and node["choices"]:
                local = copy.deepcopy(node["choices"][0])
                local.update({"id": f"local-{OLLAMA_MODEL}", "displayName": f"Local: {OLLAMA_MODEL}",
                               "baseModelName": OLLAMA_MODEL, "disableReason": None,
                               "description": "Local model via Ollama", "provider": "UNKNOWN"})
                if not any(m.get("id") == local["id"] for m in node["choices"]):
                    node["choices"].insert(0, local)
                node["defaultId"] = local["id"]
            else:
                for v in node.values():
                    inject(v)
        elif isinstance(node, list):
            for item in node:
                inject(item)

    inject(data)
    return JSONResponse(data)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Warp→Ollama proxy on :8080 (model={OLLAMA_MODEL})")
    uvicorn.run(app, host="127.0.0.1", port=8080)
