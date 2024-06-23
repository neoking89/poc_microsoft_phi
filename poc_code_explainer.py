# poc_summary.py

import asyncio
import string
from typing import AsyncGenerator
from llm_invoke import LLM
from get_code_content import CodeContentOrganizer, extract_docs_type_hints_and_contents_of

async def answer_question_with_code(question: str, llm, code_content: str, max_words: int = 150) -> AsyncGenerator[str, None]:
    prompt = (
        "You are a helpful chatbot that answers questions about code. "
        "Tasks: 1) Answer concisely and to the point. Use no more words than necessary. "
        "Focus on key points and code essence. Describe important functions/classes. "
        "Ensure clarity and structure. Include illustrative examples if present. "
        "2) Avoid hallucination. Only use information from the given code."
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Answer in max {max_words} words: `{question}`\n\nCode:\n{code_content}"},
    ]

    buffer = ""
    for content in llm.stream(messages, max_tokens=1024):
        buffer += content
        if set(content) & set(string.whitespace + string.punctuation):
            yield buffer
            buffer = ""
    
    if buffer:
        yield buffer

async def main(question: str, llm, code_content: str, max_words: int = 150) -> None:
    print("Answer:", end="", flush=True)
    async for part in answer_question_with_code(question, llm, code_content, max_words):
        print(part, end="", flush=True)

if __name__ == "__main__":
    root_dir = r".venv\Lib\site-packages\orpheus"
    organizer = CodeContentOrganizer(root_dir, ["__pycache__"], ["test"])
    code_content = organizer.optimize_content_length(organizer.get_code_content(), max_lines=50)
    code_content = extract_docs_type_hints_and_contents_of(code_content, organizer.separator) # extract docs, type hints, and contents of code to fix windows context window

    llm = LLM(
        tokenizer_path="microsoft/Phi-3-mini-128k-instruct",
        model_path="microsoft/Phi-3-mini-128k-instruct",
        context_length=40000
    )
    
    question = "What is the main purpose of the orpheus package?"
    asyncio.run(main(question, llm, code_content, max_words=500))