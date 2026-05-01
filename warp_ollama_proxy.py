import os
import json
import base64
import uuid
import struct
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warp_proxy")

app = FastAPI()

WARP_BACKEND = os.environ.get("WARP_BACKEND", "https://app.warp.dev")
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ---------------------------------------------------------------------------
# Minimal protobuf / SSE helpers
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    bits = value & 0x7F
    value >>= 7
    result = b""
    while value:
        result += bytes([0x80 | bits])
        bits = value & 0x7F
        value >>= 7
    result += bytes([bits])
    return result

def encode_proto_string(field_number: int, value: str) -> bytes:
    """Encode a protobuf string field."""
    encoded = value.encode("utf-8")
    tag = encode_varint((field_number << 3) | 2)   # wire type 2 = length-delimited
    return tag + encode_varint(len(encoded)) + encoded

def encode_proto_message(field_number: int, payload: bytes) -> bytes:
    """Encode an embedded message field."""
    tag = encode_varint((field_number << 3) | 2)
    return tag + encode_varint(len(payload)) + payload

# ---------------------------------------------------------------------------
# Build ResponseEvent protobuf bytes
#
# ResponseEvent fields (from response.proto):
#   1 = init  (StreamInit)
#   2 = client_actions (ClientActions)
#   3 = finished (StreamFinished)
#
# StreamInit fields:
#   1 = conversation_id (string)
#   2 = request_id (string)
#   3 = run_id (string)
#
# ClientActions field:
#   1 = repeated ClientAction actions
#
# ClientAction – AppendToMessageContent (field 5):
#   message AppendToMessageContent { string task_id=3; Message message=1; FieldMask mask=2 }
#
# Message fields (task.proto):
#   1 = id (string)
#   2 = task_id (string)
#   (content field varies by type; assistant text lives in field 4 = AssistantMessage)
#
# We emit:
#  1. Init event
#  2. BeginTransaction
#  3. CreateTask (with initial assistant message)
#  4. AppendToMessageContent chunks (streaming text)
#  5. CommitTransaction
#  6. StreamFinished / Done
# ---------------------------------------------------------------------------

def make_init_event(conversation_id: str, request_id: str) -> bytes:
    """Build StreamInit protobuf."""
    init = (
        encode_proto_string(1, conversation_id) +
        encode_proto_string(2, request_id) +
        encode_proto_string(3, request_id)  # run_id
    )
    # ResponseEvent.init = field 1
    return encode_proto_message(1, init)

def make_begin_transaction_event() -> bytes:
    """Build ClientActions containing BeginTransaction."""
    # ClientAction.begin_transaction = field 9, empty message
    client_action = encode_proto_message(9, b"")
    # ClientActions.actions = field 1, repeated
    client_actions_payload = encode_proto_message(1, client_action)
    # ResponseEvent.client_actions = field 2
    return encode_proto_message(2, client_actions_payload)

def make_commit_transaction_event() -> bytes:
    """Build ClientActions containing CommitTransaction."""
    # ClientAction.commit_transaction = field 10, empty message
    client_action = encode_proto_message(10, b"")
    client_actions_payload = encode_proto_message(1, client_action)
    return encode_proto_message(2, client_actions_payload)

def make_create_task_event(task_id: str, message_id: str) -> bytes:
    """Build ClientActions containing CreateTask with an empty assistant message."""
    # Message:
    #   1 = id, 2 = task_id, 4 = assistant message content (AssistantMessage)
    # We'll use a simple AssistantMessage with empty content.
    # task.proto Message field 4 = AssistantMessage { string content = 1 }
    assistant_message = encode_proto_string(1, "")  # content = ""
    # Message fields: 1=id, 2=task_id, 4=AssistantMessage
    message = (
        encode_proto_string(1, message_id) +
        encode_proto_string(2, task_id) +
        encode_proto_message(4, assistant_message)
    )
    # Task fields: 1=id (string), 3=messages (repeated Message)
    task = (
        encode_proto_string(1, task_id) +
        encode_proto_message(3, message)
    )
    # ClientAction.create_task = field 1
    client_action = encode_proto_message(1, encode_proto_message(1, task))
    client_actions_payload = encode_proto_message(1, client_action)
    return encode_proto_message(2, client_actions_payload)

def make_append_text_event(task_id: str, message_id: str, text_chunk: str) -> bytes:
    """Build ClientActions containing AppendToMessageContent for streaming text."""
    # AppendToMessageContent:
    #   1 = message (Message)
    #   2 = mask (FieldMask) – we set paths to "assistant_message.content"
    #   3 = task_id (string)
    #
    # Message.4 = AssistantMessage { 1 = content }
    assistant_message = encode_proto_string(1, text_chunk)
    message = (
        encode_proto_string(1, message_id) +
        encode_proto_string(2, task_id) +
        encode_proto_message(4, assistant_message)
    )
    # FieldMask: field 1 = paths (repeated string)
    field_mask = encode_proto_string(1, "assistant_message.content")
    append = (
        encode_proto_message(1, message) +
        encode_proto_message(2, field_mask) +
        encode_proto_string(3, task_id)
    )
    # ClientAction.append_to_message_content = field 5
    client_action = encode_proto_message(5, append)
    client_actions_payload = encode_proto_message(1, client_action)
    return encode_proto_message(2, client_actions_payload)

def make_finished_done_event() -> bytes:
    """Build StreamFinished with reason=Done."""
    # StreamFinished.done = field 2, empty message
    finished = encode_proto_message(2, b"")
    return encode_proto_message(3, finished)

def proto_to_sse_data(proto_bytes: bytes) -> str:
    """Encode proto bytes as base64 for SSE data field."""
    b64 = base64.urlsafe_b64encode(proto_bytes).decode("utf-8")
    return f"data: \"{b64}\"\n\n"

# ---------------------------------------------------------------------------
# Ollama agent-mode SSE stream handler
# ---------------------------------------------------------------------------

async def stream_ollama_response(messages: list):
    """
    Call Ollama's /api/chat streaming endpoint and yield protobuf SSE events
    that mimic what the Warp backend would send.
    """
    conversation_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    # 1. Init
    yield proto_to_sse_data(make_init_event(conversation_id, request_id))

    # 2. BeginTransaction
    yield proto_to_sse_data(make_begin_transaction_event())

    # 3. CreateTask with empty assistant message
    yield proto_to_sse_data(make_create_task_event(task_id, message_id))

    # 4. Stream text chunks from Ollama
    full_response = ""
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
                        full_response += delta
                        yield proto_to_sse_data(make_append_text_event(task_id, message_id, delta))
                    if chunk.get("done"):
                        break
    except Exception as e:
        error_msg = f"\n\n[Local Ollama error: {e}]"
        yield proto_to_sse_data(make_append_text_event(task_id, message_id, error_msg))
        logger.error(f"Ollama streaming error: {e}")

    # 5. CommitTransaction
    yield proto_to_sse_data(make_commit_transaction_event())

    # 6. StreamFinished / Done
    yield proto_to_sse_data(make_finished_done_event())

    logger.info(f"Ollama response complete ({len(full_response)} chars)")


def extract_user_query_from_proto(body: bytes) -> str:
    """
    Best-effort extraction of the user query string from the protobuf Request body.
    The Request has field 2=input, which has a oneof type, the user query is
    type 1=UserQuery with field 1=query (string).
    We scan for readable UTF-8 strings as a fallback.
    """
    # Try to find length-delimited strings in the proto
    # A simple heuristic: scan for sequences that look like UTF-8 text
    try:
        # Walk through the raw bytes looking for length-prefixed strings
        strings_found = []
        i = 0
        while i < len(body):
            # Read tag byte
            tag = body[i]
            wire_type = tag & 0x07
            i += 1
            if wire_type == 2:  # length-delimited
                # Read length varint
                length = 0
                shift = 0
                while i < len(body):
                    b = body[i]
                    i += 1
                    length |= (b & 0x7F) << shift
                    shift += 7
                    if not (b & 0x80):
                        break
                if i + length <= len(body):
                    data = body[i:i + length]
                    try:
                        s = data.decode("utf-8")
                        if len(s) > 5 and not s.startswith("\x00"):
                            strings_found.append(s)
                    except Exception:
                        pass
                    i += length
            elif wire_type == 0:  # varint
                while i < len(body) and (body[i] & 0x80):
                    i += 1
                i += 1
            elif wire_type == 5:  # 32-bit
                i += 4
            elif wire_type == 1:  # 64-bit
                i += 8
            else:
                break
        # Return the longest string found (likely the user query)
        if strings_found:
            return max(strings_found, key=len)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Generic transparent proxy
# ---------------------------------------------------------------------------

async def forward_to_warp(request: Request, path: str):
    """Forward the request to the real Warp backend and stream the response."""
    client = httpx.AsyncClient(timeout=60.0)
    url = httpx.URL(path=f"/{path}", query=request.url.query.encode("utf-8"))
    headers = dict(request.headers)
    headers.pop("host", None)
    body = await request.body()
    warp_req = client.build_request(
        request.method,
        f"{WARP_BACKEND}{url}",
        headers=headers,
        content=body,
    )
    logger.info(f"Forwarding {request.method} /{path} to Warp backend")
    warp_resp = await client.send(warp_req, stream=True)
    return StreamingResponse(
        warp_resp.aiter_raw(),
        status_code=warp_resp.status_code,
        headers=dict(warp_resp.headers),
        background=BackgroundTask(warp_resp.aclose),
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_all(request: Request, path: str):

    # ---- Intercept agent-mode SSE endpoint --------------------------------
    if path in ("ai/multi-agent", "ai/passive-suggestions") and request.method == "POST":
        logger.info(f"Intercepting /{path} → routing to local Ollama ({OLLAMA_MODEL})")
        body = await request.body()
        query_text = extract_user_query_from_proto(body)
        logger.info(f"Extracted user query: {query_text[:120]!r}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding and terminal assistant built into the Warp terminal. "
                    "You are running locally via Ollama. Help the user with their shell, code, and "
                    "file-system tasks. Be concise and practical."
                ),
            },
            {"role": "user", "content": query_text or "(User sent a request)"},
        ]
        return StreamingResponse(
            stream_ollama_response(messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- Intercept GraphQL model list to inject local model ---------------
    if path == "graphql/v2" and request.method == "POST":
        body = await request.body()
        try:
            body_json = json.loads(body)
            op_name = body_json.get("operationName", "")
            if not op_name and "query" in body_json:
                q = body_json["query"].lower()
                if "getfeaturemodelchoices" in q:
                    op_name = "GetFeatureModelChoices"
                elif "freeavailablemodels" in q:
                    op_name = "FreeAvailableModels"
                elif "generatedialogue" in q:
                    op_name = "GenerateDialogue"
                elif "generatecommands" in q:
                    op_name = "GenerateCommands"

            if op_name == "GenerateDialogue":
                logger.info("Intercepting GenerateDialogue")
                return await handle_generate_dialogue(body_json)
            if op_name == "GenerateCommands":
                logger.info("Intercepting GenerateCommands")
                return await handle_generate_commands(body_json)
            if op_name in ("GetFeatureModelChoices", "FreeAvailableModels"):
                return await handle_get_models(request, body)
        except json.JSONDecodeError:
            pass

    # ---- Intercept request_limit_info to return unlimited -----------------
    if path == "graphql/v2" and request.method == "POST":
        body = await request.body()
        try:
            body_json = json.loads(body)
            if "requestLimitInfo" in body_json.get("query", ""):
                return JSONResponse(content={
                    "data": {
                        "user": {
                            "requestLimitInfo": {
                                "isUnlimited": True,
                                "nextRefreshTime": "2099-01-01T00:00:00Z",
                                "requestLimit": 99999,
                                "requestsUsedSinceLastRefresh": 0
                            }
                        }
                    }
                })
        except Exception:
            pass

    return await forward_to_warp(request, path)


# ---------------------------------------------------------------------------
# GraphQL handlers (legacy, kept for completeness)
# ---------------------------------------------------------------------------

async def handle_generate_dialogue(body_json):
    variables = body_json.get("variables", {})
    input_data = variables.get("input", {})
    prompt = input_data.get("prompt", "")
    transcript = input_data.get("transcript", [])
    messages = [
        {"role": "system", "content": "You are a helpful coding and terminal assistant built into the Warp terminal. You are running locally via Ollama."}
    ]
    for part in transcript:
        if part.get("user"):
            messages.append({"role": "user", "content": part["user"]})
        if part.get("assistant"):
            messages.append({"role": "assistant", "content": part["assistant"]})
    messages.append({"role": "user", "content": prompt})
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": OLLAMA_MODEL, "messages": messages, "stream": False
            }, timeout=120.0)
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        answer = f"Error calling Ollama: {e}"
    return JSONResponse(content={"data": {"generateDialogue": {
        "__typename": "GenerateDialogueOutput",
        "status": {
            "__typename": "GenerateDialogueSuccess",
            "answer": answer,
            "requestLimitInfo": {
                "isUnlimited": True,
                "nextRefreshTime": "2099-01-01T00:00:00Z",
                "requestLimit": 9999,
                "requestsUsedSinceLastRefresh": 0
            },
            "transcriptSummarized": False, "truncated": False
        },
        "responseContext": {"serverVersion": "v1.0-ollama-proxy"}
    }}})


async def handle_generate_commands(body_json):
    variables = body_json.get("variables", {})
    prompt = variables.get("input", {}).get("prompt", "")
    messages = [
        {"role": "system", "content": "You are a shell command generator. Respond with EXACTLY the shell command. No markdown, no explanation."},
        {"role": "user", "content": prompt}
    ]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": OLLAMA_MODEL, "messages": messages, "stream": False
            }, timeout=30.0)
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        answer = f"echo 'Ollama error: {e}'"
    return JSONResponse(content={"data": {"generateCommands": {
        "__typename": "GenerateCommandsOutput",
        "status": {
            "__typename": "GenerateCommandsSuccess",
            "commands": [{"command": answer, "description": "Generated locally via Ollama", "parameters": []}]
        },
        "responseContext": {"serverVersion": "v1.0-ollama-proxy"}
    }}})


async def handle_get_models(request: Request, body_bytes: bytes):
    import copy
    client = httpx.AsyncClient()
    headers = dict(request.headers)
    headers.pop("host", None)
    warp_req = client.build_request("POST", f"{WARP_BACKEND}/graphql/v2", headers=headers, content=body_bytes)
    logger.info("Injecting local model into models response")
    try:
        warp_resp = await client.send(warp_req)
        data = warp_resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch models from backend: {e}")
        return JSONResponse(content={"data": {}})

    def inject_local_model(node):
        if isinstance(node, dict):
            if "choices" in node and isinstance(node["choices"], list) and node["choices"]:
                local_model = copy.deepcopy(node["choices"][0])
                local_model["id"] = f"local-{OLLAMA_MODEL}"
                local_model["displayName"] = f"Local: {OLLAMA_MODEL}"
                local_model["baseModelName"] = OLLAMA_MODEL
                local_model["disableReason"] = None
                local_model["description"] = "Local model running via Ollama"
                local_model["provider"] = "UNKNOWN"
                if not any(m.get("id") == local_model["id"] for m in node["choices"]):
                    node["choices"].insert(0, local_model)
                node["defaultId"] = local_model["id"]
                if "preferredCodexModelId" in node:
                    node["preferredCodexModelId"] = local_model["id"]
            else:
                for v in node.values():
                    inject_local_model(v)
        elif isinstance(node, list):
            for item in node:
                inject_local_model(item)

    inject_local_model(data)
    return JSONResponse(content=data)


if __name__ == "__main__":
    logger.info(f"Starting Warp→Ollama Proxy on port 8080 (model: {OLLAMA_MODEL})")
    uvicorn.run(app, host="127.0.0.1", port=8080)
