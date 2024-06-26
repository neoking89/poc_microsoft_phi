# poc_summary.py

import asyncio
import string
from typing import AsyncGenerator
from llm_invoke import LLM

llm = LLM(
    tokenizer_path="microsoft/Phi-3-mini-128k-instruct",
    model_path="./model/phi-3-mini-128k-instruct.Q8_0.gguf",
    context_length=40000
)

text = """
Incorporating a non-linear way of processing weights, such as using np.log or np.exp, can be useful in various scenarios. Here are some concrete reasons and examples where non-linear processing of weights might be beneficial:

1. Enhancing Differences: Non-linear functions can help to accentuate differences between weights. For instance, using np.exp can amplify the effect of higher weights, making them stand out more distinctly.

2. Diminishing Returns: Non-linear functions like np.log can be used to model scenarios with diminishing returns. This is useful when increasing weights should have progressively smaller impacts.

3. Handling Skewness: If the distribution of weights is skewed, applying np.log can help in normalizing the distribution, making it more symmetric and easier to work with in further processing.

4. Improving Numerical Stability: Sometimes, weights close to 0 or 1 can lead to numerical instability in computations. Applying transformations like np.log or np.exp can mitigate this.

5. Modeling Real-World Phenomena: Certain real-world phenomena are naturally modeled using exponential or logarithmic relationships. For example, in finance, compound interest is modeled using exponentials.

Example Usage:

Here's a simple example demonstrating how to use np.log and np.exp to process weights:

import numpy as np

# Example weights between 0 and 1
weights = np.array([0.1, 0.5, 0.8, 0.95])

# Applying np.exp to enhance differences
enhanced_weights_exp = np.exp(weights)
print("Enhanced weights using np.exp:", enhanced_weights_exp)

# Applying np.log to model diminishing returns
# Adding a small value to avoid log(0)
diminished_weights_log = np.log(weights + 1e-10)
print("Diminished weights using np.log:", diminished_weights_log)

# Normalizing the enhanced weights using np.exp
normalized_exp_weights = enhanced_weights_exp / np.sum(enhanced_weights_exp)
print("Normalized enhanced weights using np.exp:", normalized_exp_weights)

# Normalizing the diminished weights using np.log
normalized_log_weights = diminished_weights_log / np.sum(diminished_weights_log)
print("Normalized diminished weights using np.log:", normalized_log_weights)

Explanation:

1. Enhancing Differences with np.exp:
   - Applying np.exp to the weights increases the gap between smaller and larger weights. This can be useful in scenarios where you want to prioritize higher weights more significantly.

2. Modeling Diminishing Returns with np.log:
   - Using np.log can model situations where increasing weights should have less incremental impact. This transformation is particularly useful when handling weights that follow a power-law distribution or when you want to compress the range of the weights.

3. Normalization:
   - After applying non-linear transformations, normalization ensures that the processed weights sum to 1, which is often a requirement for probabilistic interpretations.

These examples showcase how non-linear transformations can be strategically used to manipulate weights in a manner that aligns with specific problem requirements or characteristics.
"""

async def summarize_text(max_words: int = 150) -> AsyncGenerator[str, None]:
    """
    Summarizes the given text using an LLM (Large Language Model) with a specified maximum number of words.

    Parameters
    ----------
    max_words : int, optional
        The maximum number of words for the summary (default is 150).

    Yields
    ------
    str
        Parts of the summary text generated by the LLM.
    """
    prompt = (
        "You are a helpful chatbot that summarizes texts. "
        "You have the following tasks:"
        "###TASK 1: Summarize the text###"
        "Create a summary of the text that is concise and to the point."
        "Use no more words than necessary. "
        "Focus on the key points and convey the essence of the text. "
        "Avoid unnecessary details and repetitions. "
        "If there are action points or recommendations in the text, include them in the summary."
        "Ensure the summary is clear and well-structured. "
        "If the text contains examples, you can include them in the summary if they are illustrative of the main points."
        "If the examples are code snippets, you can describe the functionality without providing the exact code."
        "Do not invent information and only convey the core points."
        "###TASK 2: Avoid Hallucination###"
        "Do not include information that is not present in the text. "
        "Stick to the content of the text and do not add any additional information. "
        ""
    )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Summarize the following text in a maximum of {max_words} words: `{text}`",
        },
    ]

    buffer = ""
    for content in llm.stream(messages, max_tokens=512):
        buffer += content
        if any(c in string.whitespace + string.punctuation for c in content):
            yield buffer
            buffer = ""

    if buffer:
        yield buffer

async def main(max_words: int = 150) -> None:
    """
    Main function to run the text summarization and print the result.

    Parameters
    ----------
    max_words : int, optional
        The maximum number of words for the summary (default is 150).
    """
    print("Summary:")
    async for part in summarize_text(max_words):
        print(part, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
