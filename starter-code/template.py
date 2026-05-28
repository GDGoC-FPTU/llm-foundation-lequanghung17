# """
"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost from token counts using the pricing table above."""
    pricing = PRICING_1M_TOKENS[model]
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
    }
    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    response_text = "".join(
        getattr(content_part, "text", "")
        for content_part in getattr(response, "content", [])
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(
        prompt,
        model=OPENAI_MODEL,
    )
    mini_response, mini_latency, mini_usage = call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
    )
    gemini_response, gemini_latency, gemini_usage = call_gemini(
        prompt,
        model=GEMINI_MODEL,
    )

    return {
        "gpt4o": {
            "response": gpt4o_response,
            "latency": gpt4o_latency,
            "cost": _calculate_cost(
                OPENAI_MODEL,
                gpt4o_usage["input_tokens"],
                gpt4o_usage["output_tokens"],
            ),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_response,
            "latency": mini_latency,
            "cost": _calculate_cost(
                OPENAI_MINI_MODEL,
                mini_usage["input_tokens"],
                mini_usage["output_tokens"],
            ),
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_response,
            "latency": gemini_latency,
            "cost": _calculate_cost(
                GEMINI_MODEL,
                gemini_usage["input_tokens"],
                gemini_usage["output_tokens"],
            ),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        recent_history = history[-6:]
        formatted_history = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in recent_history
        )

        print("Assistant: ", end="", flush=True)
        assistant_response_parts: list[str] = []
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=formatted_history,
        )
        for chunk in response_stream:
            chunk_text = getattr(chunk, "text", "") or ""
            print(chunk_text, end="", flush=True)
            assistant_response_parts.append(chunk_text)
        print()

        history.append(
            {"role": "assistant", "content": "".join(assistant_response_parts)}
        )
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exception = exc
            if attempt == max_retries:
                break
            time.sleep(base_delay * (2 ** attempt))

    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except TypeError as exc:
            if "positional" not in str(exc) and "argument" not in str(exc):
                raise
            comparison = compare_models()
        comparison["prompt"] = prompt
        results.append(comparison)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    model_labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    for result in results:
        prompt = str(result.get("prompt", ""))
        for model_key, model_label in model_labels.items():
            if model_key not in result:
                continue
            stats = result[model_key]
            response = str(stats.get("response", ""))
            truncated_response = (
                response[:47] + "..." if len(response) > 50 else response
            )
            rows.append(
                "| {prompt} | {model} | {response} | {latency:.2f}s | {tokens} | ${cost:.8f} |".format(
                    prompt=prompt.replace("|", "\\|"),
                    model=model_label,
                    response=truncated_response.replace("|", "\\|"),
                    latency=stats.get("latency", 0.0),
                    tokens=f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}",
                    cost=stats.get("cost", 0.0),
                )
            )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")


# Day 1 — LLM API Foundation
# AICB-P1: AI Practical Competency Program, Phase 1

# Instructions:
#     1. Fill in every section marked with TODO.
#     2. Do NOT change function signatures.
#     3. Copy this file to solution/solution.py when done.
#     4. Run: pytest tests/ -v
# """

# import os
# import time
# from typing import Any, Callable

# # ---------------------------------------------------------------------------
# # Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# # Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# # ---------------------------------------------------------------------------
# PRICING_1M_TOKENS = {
#     "gpt-4o": {"input": 5.00, "output": 20.00},
#     "gpt-4o-mini": {"input": 0.150, "output": 0.600},
#     "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
#     "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
#     "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
#     "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
# }

# # Standard Model Identifiers
# OPENAI_MODEL = "gpt-4o"
# OPENAI_MINI_MODEL = "gpt-4o-mini"
# GEMINI_MODEL = "gemini-2.5-flash"
# ANTHROPIC_MODEL = "claude-3-5-haiku"


# # ---------------------------------------------------------------------------
# # Task 1 — Call OpenAI (GPT-4o)
# # ---------------------------------------------------------------------------
# def call_openai(
#     prompt: str,
#     model: str = OPENAI_MODEL,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     max_tokens: int = 256,
# ) -> tuple[str, float]:
#     """
#     Call the OpenAI Chat Completions API and return the response text + latency.

#     Args:
#         prompt:      The user message to send.
#         model:       The OpenAI model to use (default: gpt-4o).
#         temperature: Sampling temperature (0.0 – 2.0).
#         top_p:       Nucleus sampling threshold.
#         max_tokens:  Maximum number of tokens to generate.

#     Returns:
#         A tuple of (response_text: str, latency_seconds: float).

#     Hint:
#         from openai import OpenAI
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     """
#     # TODO: import OpenAI, create client, call chat.completions.create,
#     #       measure start/end time, return (response_text, latency)
#     from openai import OpenAI
#     # khởi tạo client
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     # đo thời gian
#     start = time.perf_counter()
#     # gọi API
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temperature,
#         top_p=top_p,
#         max_tokens=max_tokens,
#     )
#     # ket thuc đo thời gian
#     end = time.perf_counter()
#     latency = end - start
#     # trích xuất response text
#     response_text = response.choices[0].message.content
#     return response_text, latency
# # ---------------------------------------------------------------------------
# # Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# # ---------------------------------------------------------------------------
# def call_gemini(
#     prompt: str,
#     model: str = GEMINI_MODEL,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     max_tokens: int = 256,
# ) -> tuple[str, float, dict]:
#     """
#     Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
#     the response text, latency, and token usage stats.
#     """
#     import os
#     import time
#     from google import genai
#     from google.genai import types

#     api_key = os.getenv("GEMINI_API_KEY")

#     if not api_key:
#         raise ValueError(
#             "GEMINI_API_KEY is missing. Please add it to your .env file."
#         )

#     client = genai.Client(api_key=api_key)

#     config = types.GenerateContentConfig(
#         temperature=temperature,
#         top_p=top_p,
#         max_output_tokens=max_tokens,
#     )

#     start_time = time.perf_counter()

#     response = client.models.generate_content(
#         model=model,
#         contents=prompt,
#         config=config,
#     )

#     end_time = time.perf_counter()
#     latency_seconds = end_time - start_time

#     response_text = response.text if response.text is not None else ""

#     usage_metadata = response.usage_metadata

#     usage = {
#         "input_tokens": getattr(usage_metadata, "prompt_token_count", 0),
#         "output_tokens": getattr(usage_metadata, "candidates_token_count", 0),
#     }

#     return response_text, latency_seconds, usage


# # ---------------------------------------------------------------------------
# # Task 3 — Call Anthropic Claude (Exploratory track)
# # ---------------------------------------------------------------------------
# def call_anthropic(
#     prompt: str,
#     model: str = ANTHROPIC_MODEL,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     max_tokens: int = 256,
# ) -> tuple[str, float, dict]:
#     """
#     Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
#     the response text, latency, and token usage stats.
#     """
#     import os
#     import time
#     import anthropic

#     api_key = os.getenv("ANTHROPIC_API_KEY")

#     if not api_key:
#         raise ValueError(
#             "ANTHROPIC_API_KEY is missing. Please add it to your .env file."
#         )

#     client = anthropic.Anthropic(api_key=api_key)

#     start_time = time.perf_counter()

#     response = client.messages.create(
#         model=model,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#     )

#     end_time = time.perf_counter()
#     latency_seconds = end_time - start_time

#     response_text_parts = []

#     for block in response.content:
#         if block.type == "text":
#             response_text_parts.append(block.text)

#     response_text = "".join(response_text_parts)

#     usage = {
#         "input_tokens": response.usage.input_tokens,
#         "output_tokens": response.usage.output_tokens,
#     }

#     return response_text, latency_seconds, usage

# # ---------------------------------------------------------------------------
# # Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# # ---------------------------------------------------------------------------
# def compare_models(prompt: str) -> dict:
#     """
#     Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash
#     with the same prompt and return a structured comparison dictionary.
#     """

#     def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
#         """
#         Calculate cost using:
#         Cost = (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000
#         """

#         input_rate = PRICING_1M_TOKENS[model_name]["input"]
#         output_rate = PRICING_1M_TOKENS[model_name]["output"]

#         cost = (
#             input_tokens * input_rate
#             + output_tokens * output_rate
#         ) / 1_000_000

#         return cost

#     # 1. Call GPT-4o
#     gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(
#         prompt=prompt,
#         model=OPENAI_MODEL,
#     )

#     # 2. Call GPT-4o Mini
#     gpt4o_mini_response, gpt4o_mini_latency, gpt4o_mini_usage = call_openai(
#         prompt=prompt,
#         model=OPENAI_MINI_MODEL,
#     )

#     # 3. Call Gemini 2.5 Flash
#     gemini_response, gemini_latency, gemini_usage = call_gemini(
#         prompt=prompt,
#         model=GEMINI_MODEL,
#     )

#     # 4. Extract token usage
#     gpt4o_input_tokens = gpt4o_usage["input_tokens"]
#     gpt4o_output_tokens = gpt4o_usage["output_tokens"]

#     gpt4o_mini_input_tokens = gpt4o_mini_usage["input_tokens"]
#     gpt4o_mini_output_tokens = gpt4o_mini_usage["output_tokens"]

#     gemini_input_tokens = gemini_usage["input_tokens"]
#     gemini_output_tokens = gemini_usage["output_tokens"]

#     # 5. Calculate costs
#     gpt4o_cost = calculate_cost(
#         OPENAI_MODEL,
#         gpt4o_input_tokens,
#         gpt4o_output_tokens,
#     )

#     gpt4o_mini_cost = calculate_cost(
#         OPENAI_MINI_MODEL,
#         gpt4o_mini_input_tokens,
#         gpt4o_mini_output_tokens,
#     )

#     gemini_cost = calculate_cost(
#         GEMINI_MODEL,
#         gemini_input_tokens,
#         gemini_output_tokens,
#     )

#     # 6. Assemble final dictionary
#     comparison = {
#         "gpt4o": {
#             "response": gpt4o_response,
#             "latency": gpt4o_latency,
#             "cost": gpt4o_cost,
#             "input_tokens": gpt4o_input_tokens,
#             "output_tokens": gpt4o_output_tokens,
#         },
#         "gpt4o_mini": {
#             "response": gpt4o_mini_response,
#             "latency": gpt4o_mini_latency,
#             "cost": gpt4o_mini_cost,
#             "input_tokens": gpt4o_mini_input_tokens,
#             "output_tokens": gpt4o_mini_output_tokens,
#         },
#         "gemini_flash": {
#             "response": gemini_response,
#             "latency": gemini_latency,
#             "cost": gemini_cost,
#             "input_tokens": gemini_input_tokens,
#             "output_tokens": gemini_output_tokens,
#         },
#     }

#     return comparison

# def streaming_chatbot() -> None:
#     """
#     Run an interactive streaming chatbot in the terminal using Gemini 2.5.
#     """
#     import os
#     from dotenv import load_dotenv
#     from google import genai
#     from google.genai import types

#     load_dotenv()

#     api_key = os.getenv("GEMINI_API_KEY")

#     if not api_key:
#         raise ValueError(
#             "GEMINI_API_KEY is missing. Please add it to your .env file."
#         )

#     client = genai.Client(api_key=api_key)

#     history = []

#     config = types.GenerateContentConfig(
#         temperature=0.7,
#         top_p=0.9,
#         max_output_tokens=512,
#     )

#     print("=" * 60)
#     print("Gemini 2.5 Flash Streaming Chatbot")
#     print("Type 'quit' or 'exit' to end the session.")
#     print("=" * 60)

#     while True:
#         user_input = input("\nYou: ").strip()

#         if user_input.lower() in ["quit", "exit"]:
#             print("Goodbye!")
#             break

#         if not user_input:
#             continue

#         contents = []

#         for message in history:
#             contents.append(
#                 types.Content(
#                     role=message["role"],
#                     parts=[
#                         types.Part.from_text(text=message["content"])
#                     ],
#                 )
#             )

#         contents.append(
#             types.Content(
#                 role="user",
#                 parts=[
#                     types.Part.from_text(text=user_input)
#                 ],
#             )
#         )

#         print("Gemini: ", end="", flush=True)

#         assistant_response = ""

#         stream = client.models.generate_content_stream(
#             model=GEMINI_MODEL,
#             contents=contents,
#             config=config,
#         )

#         for chunk in stream:
#             if chunk.text:
#                 print(chunk.text, end="", flush=True)
#                 assistant_response += chunk.text

#         print()

#         history.append(
#             {
#                 "role": "user",
#                 "content": user_input,
#             }
#         )

#         history.append(
#             {
#                 "role": "model",
#                 "content": assistant_response,
#             }
#         )

#         # Keep only the last 3 turns.
#         # 1 turn = 1 user message + 1 model response.
#         history = history[-6:]
# # ---------------------------------------------------------------------------
# # Bonus Task A — Retry with exponential backoff
# # ---------------------------------------------------------------------------
# def retry_with_backoff(
#     fn: Callable[[], Any],
#     max_retries: int = 3,
#     base_delay: float = 0.1,
# ) -> Any:
#     """
#     Call fn(). If it raises an exception, retry up to max_retries times
#     with exponential backoff (delay = base_delay * 2^attempt).

#     Args:
#         fn:          Zero-argument callable to execute.
#         max_retries: Maximum number of retry attempts.
#         base_delay:  Initial delay in seconds before the first retry.

#     Returns:
#         The return value of fn() on success.

#     Raises:
#         The last exception raised by fn() after all retries are exhausted.
#     """
#     # TODO: implement retry loop with exponential backoff
#     raise NotImplementedError("Implement retry_with_backoff")


# # ---------------------------------------------------------------------------
# # Bonus Task B — Batch compare
# # ---------------------------------------------------------------------------
# def batch_compare(prompts: list[str]) -> list[dict]:
#     """
#     Run compare_models on each prompt in the list.

#     Args:
#         prompts: List of prompt strings.

#     Returns:
#         List of dicts, each being the compare_models result with an extra
#         key "prompt" containing the original prompt string.
#     """
#     # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
#     raise NotImplementedError("Implement batch_compare")


# # ---------------------------------------------------------------------------
# # Bonus Task C — Format comparison table
# # ---------------------------------------------------------------------------
# def format_comparison_table(results: list[dict]) -> str:
#     """
#     Format a list of batch compare results as a readable Markdown table string.

#     Args:
#         results: List of dicts as returned by batch_compare.

#     Returns:
#         A beautiful Markdown table string with columns:
#         | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
#     """
#     # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
#     raise NotImplementedError("Implement format_comparison_table")


# # ---------------------------------------------------------------------------
# # Entry point for manual testing
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("=== Model Comparison Test ===")
#     test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
#     try:
#         # Note: Requires valid API keys set in environment variables
#         result = compare_models(test_prompt)
#         for model_name, stats in result.items():
#             print(f"\n[{model_name.upper()}]")
#             print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
#             print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
#             print(f"Response: {stats['response']}")
#     except Exception as e:
#         print(f"Skipping live API comparison test: {e}")
#         print("Set your API keys to run manual tests.")

#     print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
#     try:
#         streaming_chatbot()
#     except Exception as e:
#         print(f"Chatbot failed to start: {e}")
