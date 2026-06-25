import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, ApiModel
from tools import elaborate_prompt, search_audio, generate_audio

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

model = ApiModel(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    token=HF_TOKEN,
)

agent = ToolCallingAgent(
    tools=[elaborate_prompt, search_audio, generate_audio],
    model=model,
    max_steps=6,
)

SYSTEM_PROMPT = """You are Audify, an AI sound designer. Your job is to take a user's text prompt and return the best matching audio.

Follow these steps in order:
1. Call elaborate_prompt with the user's input to get a detailed sound description.
2. Call search_audio with the elaborated description to check if a real audio clip exists.
3. If search_audio returns 'NO_MATCH', call generate_audio with the elaborated description to synthesise audio.
4. Verify the result makes sense for the original prompt. If something went wrong, try generate_audio again with a more specific description.
5. Return a JSON object with exactly these fields:
   - "description": the elaborated sound description
   - "source": either "retrieved" (from database) or "generated" (AudioLDM2)
   - "audio": the URL or file path to the audio

Only return the JSON. No extra text."""

def run_agent(prompt: str) -> dict:
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser prompt: {prompt}"
    result = agent.run(full_prompt)

    if isinstance(result, dict):
        return result

    import json, re
    match = re.search(r'\{.*\}', str(result), re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "description": f"Sound of {prompt}",
        "source": "generated",
        "audio": str(result),
    }
