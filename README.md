# POC klein taalmodel voor samenvatten Nederlandstalige teksten

## Overzicht

Deze repository bevat een proof of concept (POC), bedoeld als experiment met een nieuw klein taalmodel ontwikkeld door Microsoft, genaamd Phi. Dit model (in GGUF-formaat) is bedoeld om lokaal op een CPU te runnen en biedt als voordeel dat er dus geen GPU's nodig zijn. Daarnaast is dit model door het kleine formaat (ca 3GB) uitermate geschikt om op consumer-grade pc's te draaien.

In de context van deze POC, wordt dit model gebruikt om Nederlandstalige transcripten samen te vatten. Hierbij wordt de gegenereerde samenvatting live gestreamd naar de console, waardoor het voor de gebruiker direct inzichtelijk is wat voor tekst er gegenereerd wordt.

## Installeren

- *commandline -> setup_py_project1.ps1*
  Deze stap activeert een virtual environment in python
- *pip3 install -r requirements.txt*
  Installeer alle benodigde dependencies
- *python poc_summary.py*
  Vat de tekst in het script samen
- Ga naar https://huggingface.co/BramVanroy/fietje-3-mini-4k-instruct-GGUF/tree/main en download een van de modellen
  Plaats deze vervolgens lokaal in een nieuwe directory genaamd *"model"* en verwijs naar dit model in de file *configs.py* in de MODEL_PATH variable.

## Gebruikte Bronnen

**Deel van de code:**

https://medium.com/towards-artificial-intelligence/the-microsoft-phi-3-mini-is-mighty-impressive-a0f1e7bb6a8c

**Modellen:**

https://huggingface.co/BramVanroy/fietje-3-mini-4k-instruct-GGUF (klein taalmodel getraind op nederlandse tekst)

https://huggingface.co/microsoft/Phi-3-vision-128k-instruct (model getraind voor OCR)
