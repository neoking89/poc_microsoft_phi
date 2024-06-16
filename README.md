# POC nieuw taalmodel Phi voor samenvatten Nederlandse teksten en OCR

**Door: Vincent Ouwendijk**

## Intro

Deze repository bevat een proof of concept (POC), bedoeld als experiment met een nieuw klein taalmodel ontwikkeld door Microsoft, genaamd Phi. Dit model (in GGUF-formaat) kan lokaal op een CPU worden gerund en biedt als groot voordeel dat er bv. geen GPU's nodig zijn. Daarnaast is dit model door het kleine formaat (ca 3GB) uitermate geschikt om op consumer-grade pc's te draaien.

In deze POC wordt dit model voor 2 toepassingen gebruikt:

1. Het samenvatten van Nederlandse tekst. In de file poc_summary.py
2. Het uitvoeren van OCR op een foto met Nederlandse tekst. In de file poc_ocr.py

In de context van deze POC, wordt dit model gebruikt om Nederlandstalige transcripten samen te vatten. Hierbij wordt de gegenereerde samenvatting live gestreamd naar de console, waardoor het voor de gebruiker direct inzichtelijk is wat voor tekst er gegenereerd wordt.

Installeren

- *commandline -> setup_py_project1.ps1*
  Deze stap maakt en activeert een virtual environment in python
- *pip3 install -r requirements.txt*
  Installeer alle benodigde dependencies
- Ga naar https://huggingface.co/BramVanroy/fietje-3-mini-4k-instruct-GGUF/tree/main en download een van de modellen.
  Plaats deze vervolgens lokaal in een nieuwe directory genaamd *"model"* en verwijs naar dit model in de file *configs.py* in de MODEL_PATH variable. Er is hier ruimte voor experiment, waarbij de verwachting is dat de kwaliteit van de samenvatting zal verbeteren met het kiezen van een groter model.
- *python poc_summary.py*
  Vat de tekst in het script samen
- python poc_ocr.py
  Voer OCR uit op de foto in de "url" variable

## Resultaten

De gebruikte inputdata (voor samenvatten en ocr) is direct terug te vinden in de relevante scripts.

***Resultaten van poc_summary.py:***

*De spreker begint een vergadering over het invoeren van appels en vraagt Henk om een update. Er wordt besproken dat er sprake is van een contract met Frankrijkse leveranciers, maar sommige appels niet aan kwaliteitseisen voldoen. Jan heeft klachten ontvangen over de kwaliteit, maar deze zijn minimaal. Er worden levertijden op schema gehouden, al waren er kleine vertragingen die ingewikkeld werden. Logistieke problemen bij douane in Rotterdam worden genoemd. Het wordt voorgesteld om lokale leveranciers te overwegen als oplossing en Henk zal onderzoek doen naar lokale opties, terwijl Raggamuffin contacten heeft die hij kan benaderen. De spreker wil de evaluatie van deze opties ontvangen en is dankbaar voor het inzetten van alle betrokken partijen. De vergadering sluit af met een bedankje aan Henk en Raggamuffin, waarna Jan de updates zal volgen over de voortgang.*

De samenvatting is vrij accuraat, maar zou nog meer tot de kern kunnen komen. Opvallend is ook dat er enkele spelfouten worden gemaakt. Ik heb hiervoor het *fietje-3-mini-4k-instruct-Q5_K_M.gguf* model gebruikt. Een groter model zou de samenvatting wellicht ten goede komen. Wat opvalt, is dat de samenvatting ontzettend snel gegenereerd wordt op een CPU.

logs:

llama_print_timings:        load time =   10456.05 ms
llama_print_timings:      sample time =      29.20 ms /   255 runs   (    0.11 ms per token,  8734.07 tokens per second)
llama_print_timings: prompt eval time =   15002.58 ms /   730 tokens (   20.55 ms per token,    48.66 tokens per second)
llama_print_timings:        eval time =   16445.57 ms /   254 runs   (   64.75 ms per token,    15.44 tokens per second)
llama_print_timings:       total time =   31598.13 ms /   984 tokens

***Resultaten van poc_ocr.py:***

*The image contains a bar chart with three categories: Bovenste (10%), Middelste (40%), and Onderste (50%). Each category has a corresponding bar divided into two colors, red and blue, with percentages and monetary values indicated. The red portion of the bar represents 36%, 40%, and 55% respectively, while the blue portion represents the remaining percentage. There is also a note pointing to the blue portion of the bars with the text "Betaalde belasting" and "Jaarinkomen (bruto, gemiddeld)". The monetary values are €51.845, €54.865, and €17.524 for Bovenste, Middelste, and Onderste respectively. The text "Jaarinkomen (bruto, gemiddeld)" suggests that the values are net income after taxes.*

Alhoewel dit model alleen in het Engels beschikbaar is, weet het de meegegeven foto 100% accuraat te beschrijven.

***Let op dat het gebruikte model in de code voor OCR wel op de GPU gerund moet worden.***

## Gebruikte Bronnen

**Deel van de code:**

https://medium.com/towards-artificial-intelligence/the-microsoft-phi-3-mini-is-mighty-impressive-a0f1e7bb6a8c

**Modellen:**

https://huggingface.co/BramVanroy/fietje-3-mini-4k-instruct-GGUF (klein taalmodel getraind op nederlandse tekst)

https://huggingface.co/microsoft/Phi-3-vision-128k-instruct (model getraind voor OCR)
