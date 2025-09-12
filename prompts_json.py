import json

from prompt_wrapper import PromptWrapper, Response
from version import VERSION


def generate_prompt_json(prompts: list[PromptWrapper], path: str):
    prompt_dicts = [prompt.to_dict() for prompt in prompts]
    with open(path, 'w') as f:
        json.dump(prompt_dicts, f, indent=4)

    print(f"{len(prompts)} rompts successfully written to {path}")


def load_prompts_from_json(path: str):
    """
    Load a list of PromptWrapper objects from a JSON file.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    res = [PromptWrapper.from_dict(item) for item in data]
    if not res[0].version == VERSION:
        print("Warning: The version of the loaded prompts does not match the current library version.")

    # Convert each dictionary back into a PromptWrapper object
    return res


def generate_response_json(responses: list[Response], path: str, logging: bool = True):
    response_dicts = [response.to_dict() for response in responses]
    with open(path, 'w') as f:
        json.dump(response_dicts, f, indent=4)

    if logging:
        print(f"Responses successfully written to {path}")


def load_responses_from_json(path: str):
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert each dictionary back into a Response object
    res = [Response.from_dict(item) for item in data]

    if not res[0].wrapped_prompt.version == VERSION:
        print("Warning: The version of the loaded responses does not match the current library version.")
    return res
