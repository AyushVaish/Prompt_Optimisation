import time

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

# bring in cost‐calculators
from costs import (
    calculate_gpt4o_mini_cost,
    calculate_claude3_haiku_cost,
    calculate_mistral_cost,
)

def model_selection(prompt_from_user: str, model: str):
    """
    Choose and invoke an LLM based on the `model` argument, measure latency,
    extract token usage, calculate cost, and return a summary dictionary.

    Args:
        prompt_from_user (str): The user's prompt to send to the chosen LLM.
        model (str): The name of the model to use. Supported values are:
                     - "gpt-4o-mini"
                     - "claude-3-haiku-20240307"
                     - "open-mistral-7b"

    Returns:
        dict or str: If the model is supported, returns a dictionary with:
            - "Content": str, the LLM's response text.
            - "Prompt Tokens": int, number of tokens in the prompt.
            - "Completion tokens" or "Completion Tokens": int, number of tokens in the completion.
            - "Cost" or "Total Cost (USD)": float, cost computed by the relevant cost function.
            - "Latecy" or "Latency (s)": float, round-trip time (seconds) for the API call.
        If the model is not supported, returns the string "Model not supported".
    """
    model_lower = model.lower()
    start_time = time.perf_counter()

    if model_lower == "gpt-4o-mini":
        # Instantiate the ChatOpenAI client
        llm = ChatOpenAI(model='gpt-4o-mini')
        result = llm.invoke(prompt_from_user)

        # Measure elapsed time
        latency = time.perf_counter() - start_time

        # Extract token usage
        token_data = result.response_metadata['token_usage']
        prompt_tokens = token_data['prompt_tokens']
        completion_tokens = token_data['completion_tokens']

        # Compute cost
        cost = calculate_gpt4o_mini_cost(prompt_tokens, completion_tokens)

        # Original code printed result; we keep that behavior
        print(result)

        return {
            "Content": result.content,
            "Prompt Tokens": prompt_tokens,
            "Completion Tokens": completion_tokens,
            "Cost": f"{cost:.5f}",
            "Latency": f"{latency:.3f}"
        }

    elif model_lower == "claude-3-haiku-20240307":
        # Instantiate the ChatAnthropic client
        llm = ChatAnthropic(model=model)
        result = llm.invoke(prompt_from_user)

        latency = time.perf_counter() - start_time

        # Extract token usage from `usage`
        usage = result.response_metadata.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        cost = calculate_claude3_haiku_cost(prompt_tokens, completion_tokens)

        return {
            "Content": result.content,
            "Prompt Tokens": prompt_tokens,
            "Completion Tokens": completion_tokens,
            "Cost": f"{cost:.5f}",
            "Latency": f"{latency:.3f}"
        }

    elif model_lower == "open-mistral-7b":
        # Instantiate the ChatMistralAI client
        llm = ChatMistralAI(model=model)
        result = llm.invoke(prompt_from_user)

        latency = time.perf_counter() - start_time

        # Extract token usage from `token_usage`
        usage = result.response_metadata.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        cost = calculate_mistral_cost(prompt_tokens, completion_tokens)

        return {
            "Content": result.content,
            "Prompt Tokens": prompt_tokens,
            "Completion Tokens": completion_tokens,
            "Cost": f"{cost:.5f}",
            "Latency": f"{latency:.3f}"
        }

    else:
        return "Model not supported"