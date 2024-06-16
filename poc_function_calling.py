import asyncio
import string
import aioconsole
from llm_invoke import LLM
from configs import MODEL_PATH
import difflib

llm = LLM(tokenizer_path="microsoft/Phi-3-mini-4k-instruct", model_path="./model/fietje-3-mini-4k-instruct-Q5_K_M.gguf")

async def call_hello_world():
    yield "call_hello_world()"

async def execute_sql():
    yield "execute_sql()"

function_map = {
    "print_hello": call_hello_world,
    "execute_sql": execute_sql,
}

def get_closest_function(query, functions):
    closest_match = difflib.get_close_matches(query, functions, n=1, cutoff=0.6)
    return closest_match[0] if closest_match else None

async def ask(question: str):
    prompt = (
        "Je bent een behulpzame chatbot die vragen van gebruikers beantwoordt. "
        "Gebruik niet het volle aantal tokens als je denkt dat het niet nodig is. "
        "Als je een antwoord niet weet, geef dan een kort antwoord dat aangeeft dat je het antwoord niet weet. "
        "Verzin geen informatie. "
        "Als de gebruiker vraagt om een functie uit te voeren, gebruik dan een van de volgende functies uit de function_map variable: "
        f"{list(function_map.keys())}. "
        "Geef alleen de naam van de functie terug zoals: function_name()."
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Vraag: {question}"},
    ]

    # Controleer of de vraag een functie-aanroep bevat of iets dat erop lijkt
    closest_function = get_closest_function(question, function_map.keys())
    if closest_function:
        async for message in function_map[closest_function]():
            yield message
        return

    buffer = ""
    for content in llm.stream(messages, max_tokens=512):
        buffer += content
        if any(c in string.whitespace + string.punctuation for c in content):
            yield buffer
            buffer = ""

    if buffer:
        yield buffer

async def interactive_chatbot():
    print("Interactieve Chatbot. Typ je vraag en druk op Enter. Typ 'exit' of 'quit' om af te sluiten.")
    while True:
        question = "Roep de functie aan om de SQL-query uit te voeren."
        # question = await aioconsole.ainput("Jij: ")
        if question.lower() in {"exit", "quit"}:
            print("Sessie wordt beëindigd. Tot ziens!")
            break

        print("Chatbot: ", end="", flush=True)
        async for message in ask(question):
            print(message, end="", flush=True)
        print()  # Voor een nieuwe regel na het antwoord van de chatbot

if __name__ == "__main__":
    asyncio.run(interactive_chatbot())
