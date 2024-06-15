import asyncio
import string
from llm_invoke import LLM
from configs import MODEL_PATH

print("Model wordt geladen")
llm = LLM(MODEL_PATH)
print("Model is geladen")


text = """
spreker 1:
Goedemorgen, iedereen. Laten we beginnen met de vergadering over de invoer van appels. Henk, kun je ons een update geven over de huidige stand van zaken?

spreker 2:
Goedemorgen Jan, zeker. Op dit moment hebben we een overeenkomst met een leverancier uit Frankrijk. Raggamuffin, wat is jouw mening over de kwaliteit van de appels die we tot nu toe hebben ontvangen?

spreker 3:
Goedemorgen Henk en Jan. De kwaliteit van de appels is over het algemeen goed, maar er zijn een paar partijen geweest die niet aan onze standaard voldeden. Jan, heb jij toevallig nog klachten ontvangen van klanten?

spreker 1:
Ja, ik heb een paar klachten gehad, maar niets ernstigs. Henk, hoe zit het met de leveringsschema's? Liggen we nog op schema?

spreker 2:
Ja, we liggen grotendeels op schema. Er zijn wel een paar kleine vertragingen geweest, maar die hebben we kunnen opvangen. Raggamuffin, heb jij nog iets toe te voegen over de logistieke kant?

spreker 3:
Eigenlijk wel, Henk. Er zijn wat problemen geweest met de douane in Rotterdam. Jan, denk je dat we meer lokale leveranciers moeten overwegen om deze problemen te omzeilen?

spreker 1:
Dat is zeker iets om te overwegen, Raggamuffin. We moeten alle opties openhouden. Henk, kun je een onderzoek starten naar lokale leveranciers en hun mogelijkheden?

spreker 2:
Zeker Jan, ik zal er meteen mee beginnen. Raggamuffin, heb jij nog contacten die we kunnen benaderen voor dit onderzoek?

spreker 3:
Ja, ik ken een paar lokale boeren die misschien geïnteresseerd zijn. Ik zal ze deze week nog benaderen. Jan, wil jij dan de eerste evaluatie doen zodra we de gegevens hebben?

spreker 1:
Dat lijkt me een goed plan. Bedankt Henk en Raggamuffin voor jullie inzet. Laten we ervoor zorgen dat we dit proces zo soepel mogelijk laten verlopen.

spreker 2:
Absoluut, Jan. We houden je op de hoogte van de voortgang.

spreker 3:
Ja, we zullen ons best doen. Bedankt voor de vergadering, Jan.

spreker 1:
Dank jullie wel. Vergadering gesloten.
"""


async def summarize_text(max_words: int = 150):
    prompt = (
        "Je bent een behulpzame chatbot die teksten samenvat. "
        "Je hebt de volgende taken:"
        "###TAAK 1: Samenvatten van tekst###"
        "Maak een samenvatting van de tekst die kort en bondig is."
        "Gebruik niet meer woorden dan nodig is. "
        "Focus op de belangrijkste punten en geef de essentie van de tekst weer. "
        "Vermijd onnodige details en herhalingen. "
        "Als er actiepunten of aanbevelingen in de tekst staan, neem deze dan op in de samenvatting."
        "Zorg ervoor dat de samenvatting helder en goed gestructureerd is. "
        "Verzin geen informatie en geef alleen de kernpunten weer."
        "###TAAK 2: Infereren van de echte voornamen van de sprekers###"
        "Op dit moment zijn de namen van de sprekers in het transcript genoteerd als 'spreker 1', 'spreker 2', enzovoort. "
        "Infereer de echte namen van de sprekers in de tekst op basis van de context."
        "Gebruik de informatie in de tekst om de juiste namen aan de sprekers toe te wijzen. "
        ""
    )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Vat de volgende tekst samen in maximaal {max_words} woorden: `{text}`",
        },
    ]

    buffer = ""
    for content in llm.__stream__(messages, max_tokens=512):
        buffer += content
        if any(c in string.whitespace + string.punctuation for c in content):
            yield buffer
            buffer = ""

    if buffer:
        yield buffer


async def main(max_words=150):
    print("Samenvatting:")
    async for part in summarize_text(max_words):
        print(part, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
