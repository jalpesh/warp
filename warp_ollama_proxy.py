import os
import json
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warp_proxy")

app = FastAPI()

WARP_BACKEND = "https://api.warp.dev"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3") # Default model

async def forward_to_warp(request: Request, path: str):
    client = httpx.AsyncClient()
    url = httpx.URL(path=f"/{path}", query=request.url.query.encode("utf-8"))
    
    headers = dict(request.headers)
    # Remove host so httpx uses api.warp.dev
    headers.pop("host", None)
    
    body = await request.body()
    
    warp_req = client.build_request(
        request.method,
        f"{WARP_BACKEND}{url}",
        headers=headers,
        content=body,
    )
    
    logger.info(f"Forwarding {request.method} {url} to Warp backend")
    warp_resp = await client.send(warp_req, stream=True)
    
    return StreamingResponse(
        warp_resp.aiter_raw(),
        status_code=warp_resp.status_code,
        headers=warp_resp.headers,
        background=BackgroundTask(warp_resp.aclose)
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_all(request: Request, path: str):
    if path == "graphql/v2" and request.method == "POST":
        body = await request.body()
        try:
            body_json = json.loads(body)
            op_name = body_json.get("operationName")
            
            if op_name == "GenerateDialogue":
                logger.info("Intercepting GenerateDialogue GraphQL request")
                return await handle_generate_dialogue(body_json)
            elif op_name == "GenerateCommands":
                logger.info("Intercepting GenerateCommands GraphQL request")
                return await handle_generate_commands(body_json)
        except json.JSONDecodeError:
            pass
            
    return await forward_to_warp(request, path)

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
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }, timeout=120.0)
            resp.raise_for_status()
            ollama_data = resp.json()
            answer = ollama_data.get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        answer = f"Error calling Ollama: {str(e)}\nMake sure Ollama is running and you have the '{OLLAMA_MODEL}' model pulled (`ollama pull {OLLAMA_MODEL}`)."
        
    graphql_response = {
        "data": {
            "generateDialogue": {
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
                    "transcriptSummarized": False,
                    "truncated": False
                },
                "responseContext": {
                    "serverVersion": "v1.0-ollama-proxy"
                }
            }
        }
    }
    
    from fastapi.responses import JSONResponse
    return JSONResponse(content=graphql_response)

async def handle_generate_commands(body_json):
    variables = body_json.get("variables", {})
    input_data = variables.get("input", {})
    prompt = input_data.get("prompt", "")
    
    messages = [
        {"role": "system", "content": "You are a shell command generator. The user will ask for a command. You must respond with EXACTLY and ONLY the shell command to run. Do not include markdown formatting like ```bash or ```. Do not provide any explanation."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }, timeout=30.0)
            resp.raise_for_status()
            ollama_data = resp.json()
            answer = ollama_data.get("message", {}).get("content", "").strip()
            
            # Basic cleanup in case the LLM still outputs markdown
            if answer.startswith("```"):
                lines = answer.split("\\n")
                if len(lines) > 1:
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                answer = "\\n".join(lines).strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        answer = f"echo 'Error calling Ollama: {str(e)}'"
        
    graphql_response = {
        "data": {
            "generateCommands": {
                "__typename": "GenerateCommandsOutput",
                "status": {
                    "__typename": "GenerateCommandsSuccess",
                    "commands": [
                        {
                            "command": answer,
                            "description": "Generated locally via Ollama",
                            "parameters": []
                        }
                    ]
                },
                "responseContext": {
                    "serverVersion": "v1.0-ollama-proxy"
                }
            }
        }
    }
    
    from fastapi.responses import JSONResponse
    return JSONResponse(content=graphql_response)

if __name__ == "__main__":
    logger.info("Starting Warp to Ollama Proxy Server on port 8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080)
