import json

OLLAMA_MODEL = "llama3.2"

def inject_local_model(node):
    local_model = {
        "id": f"local-{OLLAMA_MODEL}",
        "displayName": f"Local: {OLLAMA_MODEL}",
        "baseModelName": OLLAMA_MODEL,
        "reasoningLevel": None,
        "description": "Local model running via Ollama",
        "disableReason": None,
        "visionSupported": False,
        "provider": "Unknown",
        "spec": {
            "cost": 0.0,
            "quality": 10.0,
            "speed": 10.0
        },
        "pricing": {
            "discountPercentage": 100.0
        },
        "contextWindow": {
            "isConfigurable": False,
            "min": 0,
            "max": 128000,
            "default": 8192
        },
        "usageMetadata": {
            "creditMultiplier": 0.0,
            "requestMultiplier": 0
        },
        "hostConfigs": []
    }

    if isinstance(node, dict):
        if "choices" in node and isinstance(node["choices"], list):
            if not any(m.get("id") == local_model["id"] for m in node["choices"]):
                node["choices"].insert(0, local_model)
            node["defaultId"] = local_model["id"]
            if "preferredCodexModelId" in node:
                node["preferredCodexModelId"] = local_model["id"]
        else:
            for k, v in node.items():
                inject_local_model(v)
    elif isinstance(node, list):
        for item in node:
            inject_local_model(item)

data = {
    "data": {
        "user": {
            "user": {
                "workspaces": [
                    {
                        "featureModelChoice": {
                            "agentMode": {
                                "defaultId": "gpt-4",
                                "choices": [{"id": "gpt-4"}]
                            }
                        }
                    }
                ]
            }
        }
    }
}
inject_local_model(data)
print(json.dumps(data, indent=2))
