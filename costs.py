def calculate_gpt4o_mini_cost(prompt_tokens: int, completion_tokens: int) -> float:
    # Pricing per token as of 31 May 2025
    prompt_price_per_token = 0.00000015  # $0.15 per 1M tokens
    completion_price_per_token = 0.0000006  # $0.60 per 1M tokens

    # Calculate individual costs
    cost_prompt = prompt_tokens * prompt_price_per_token
    cost_completion = completion_tokens * completion_price_per_token

    # Total cost
    total_cost = cost_prompt + cost_completion
    return total_cost


def calculate_claude3_haiku_cost(prompt_tokens: int, completion_tokens: int) -> float:

    input_cost = (prompt_tokens / 1_000_000) * 0.25

    output_cost = (completion_tokens / 1_000_000) * 1.25

    total_cost = input_cost + output_cost

    return total_cost


def calculate_mistral_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens + completion_tokens) * 0.00025