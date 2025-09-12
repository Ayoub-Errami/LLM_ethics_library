import json
import openai
from prompt_wrapper import *


def query_mistral_api(api_key: str, wrapped_prompt: PromptWrapper, model: LlmName) -> Response:
    client = openai.OpenAI(
        base_url="", #insert Ollama server URL
        api_key="not-needed"
    )

    messages = []
    responses = []

    try:
        safeguard = 5
        count = 0
        prompt_tokens = 0
        completion_tokens = 0

        for prompt in wrapped_prompt.prompts:
            if count >= safeguard:
                raise Exception("Too many prompts")
            count += 1

            messages.append({"role": "user", "content": prompt})
            kwargs = {}

            if not wrapped_prompt.output_structure.first_unstructured_output or count == 2:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": wrapped_prompt.output_structure.get_json_schema()
                    }
                }
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(
                model="mixtral:8x22b",  # Hardcoded Modellname mixtral:8x22b
                messages=messages,
                n=1,
               # temperature=0,  # ← NEU
                #top_p=0.9,  # ← NEU
               # extra_body={  # ← NEU
               #     "num_ctx": 2048,
               #     "num_predict": 1024
               # },
                **kwargs
            )

            if len(response.choices) != 1 or response.choices[0].message.role != "assistant":
                raise Exception("Invalid response from Mistral")

            response_str = response.choices[0].message.content
            messages.append({"role": "assistant", "content": response_str})
            responses.append(response_str)

        parsed_response = json.loads(responses[-1])
        if not parsed_response.get("decision"):
            raise Exception("No decision in response")

        decision = DecisionOption(parsed_response["decision"])

        return Response(
            wrapped_prompt=wrapped_prompt,
            decision=decision,
            llm_identifier=LlmName.MISTRAL,
            unparsed_messages=[LlmMessage.from_dict(item) for item in messages],
            parsed_response=parsed_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as e:
        print(f" Mistral error: {e}")
        raise e
