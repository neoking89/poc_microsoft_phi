# poc_ocr.py`

from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

model_id = "microsoft/Phi-3-vision-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager",
)  # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# following prompt is meant to generate a response from which the JSON can be parsed afterwards
prompt = (
    "<|image_1|>\Infer the complete text from the image. "
    "In order to do so, first understand the image and its language."
    "Also, add a description of the image, which best describes the content of the image."
    "Then provide it the following JSON format: {'text': <complete text>, 'language': <language> 'description': <description>}."
    "Be aware that you should provide the COMPLETE text in the language that is displayed in the image."
    "Do it as accurately as possible, by first understanding the image and its language and then providing the text in the JSON format."
)

messages = [
    {"role": "user", "content": prompt},
    # {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."},
    # {"role": "user", "content": "Provide insightful questions to spark discussion."}
]

url = "https://media.nu.nl/m/eylxrz9agmrt_wd854.jpg"
image = Image.open(requests.get(url, stream=True).raw)


prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 3000,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
