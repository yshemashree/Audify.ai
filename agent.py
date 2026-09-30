from tools import elaborate_prompt, search_audio, generate_audio


def run_agent(prompt: str, out_base: str) -> dict:
    """Run the pipeline for one prompt, writing audio to `out_base` + extension."""
    description, key = elaborate_prompt(prompt)

    # Library hits are already expert descriptions; only consult the vector DB
    # for free-form prompts, to upgrade them to a curated description.
    source = "generated"
    if key is None:
        matched = search_audio(description)
        if matched != "NO_MATCH":
            source = "retrieved"
            description = matched
    else:
        source = "retrieved"

    path, engine = generate_audio(prompt, description, key, out_base)

    return {
        "description": description,
        "source": source,
        "engine": engine,
        "path": path,
    }
