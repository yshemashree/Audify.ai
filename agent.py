from tools import elaborate_prompt, search_audio, generate_audio

def run_agent(prompt: str) -> dict:
    description = elaborate_prompt(prompt)
    audio = search_audio(description)
    if audio == "NO_MATCH":
        source = "generated"
        audio = generate_audio(description)
    else:
        source = "retrieved"
    return {
        "description": description,
        "source": source,
        "audio": audio,
    }
