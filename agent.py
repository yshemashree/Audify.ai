from tools import elaborate_prompt, search_audio, generate_audio

def run_agent(prompt: str) -> dict:
    description = elaborate_prompt(prompt)

    # ChromaDB vector search: returns matched expert description or "NO_MATCH"
    matched = search_audio(description)

    if matched == "NO_MATCH":
        source = "generated"
        final_description = description
    else:
        # Use the matched expert description for higher-quality generation
        source = "retrieved"
        final_description = matched

    audio = generate_audio(final_description)

    return {
        "description": final_description,
        "source": source,
        "audio": audio,
    }
