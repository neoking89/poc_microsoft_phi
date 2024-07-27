import time
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread
import os
from typing import List, Optional


def is_flash_attention_available() -> bool:
    try:
        from flash_attn import flash_attn_fn

        return True
    except ImportError:
        return False


class PhiProcessor:
    def __init__(
        self,
        model_id: str,
        prompt: str,
        device: str = None,
        max_new_tokens: int = 5000,
        temperature: float = 0.0,
        quantization_bits: Optional[int] = None,
    ):
        """
        A class specially designed as a wrapper for Microsoft Phi-3 based models.

        Parameters:
        - model_id: The model identifier to be used. Check huggingface.co/models for available phi-3 models.
        - prompt: The prompt to be used for generating responses.
        - device: The device to be used for processing. If None, it will automatically select "cuda" if available, otherwise "cpu".
        - max_new_tokens: The maximum number of tokens to generate in the response.
        - temperature: The temperature to be used for sampling. If 0.0, greedy decoding will be used.
        - quantization_bits: The number of bits to use for quantization. If None, no quantization will be used. Not applicable for CPU.
        """

        if quantization_bits is not None and quantization_bits not in [4, 8]:
            raise ValueError("Quantization bits must be either None, 4 or 8.")

        self.model_id = model_id
        self.prompt = prompt
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._attention_implementation = (
            "flash" if is_flash_attention_available() else "eager"
        )

        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count())
            torch.set_num_interop_threads(os.cpu_count())
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": "cpu"},
                trust_remote_code=True,
                torch_dtype=torch.float32,
                _attn_implementation=self._attention_implementation,
            )
        else:
            quantization_config = None
            if quantization_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf8",
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
                quantization_config=quantization_config,
                _attn_implementation=self._attention_implementation,
            )

    def prepare_inputs(self, image: Image.Image, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt_text, [image], return_tensors="pt").to(
            self.device
        )
        return inputs

    @torch.inference_mode()
    def generate_response(self, inputs: dict) -> str:
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "streamer": streamer,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        if self.temperature > 0.0:
            generation_args["temperature"] = self.temperature
            generation_args["do_sample"] = True

        thread = Thread(
            target=self.model.generate, kwargs={**inputs, **generation_args}
        )
        thread.start()

        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text

        thread.join()
        print()
        return generated_text

    def process_input_data(self, input_data) -> str:
        inputs = self.prepare_inputs(input_data, self.prompt)
        return self.generate_response(inputs)


def process_images_from_directory(
    directory_path: str, ocr_processor: PhiProcessor
) -> List[str]:
    results = []
    for filename in os.listdir(directory_path):
        try:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(directory_path, filename)
                with Image.open(image_path) as image:
                    start = time.time()
                    print(f"\nProcessing image: {filename}")
                    print("-" * 50)
                    result = ocr_processor.process_input_data(image)
                    end = time.time()
                print("-" * 50)
                print(f"Processed in {end - start:.2f} seconds")
                results.append(result)
        except Exception as e:
            print(f"Error processing image: {filename}")
            print(e)

    return results


# Example usage:
if __name__ == "__main__":
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    prompt = (
        "<|image_1|>\Extract the complete literal text from the image using Optical Character Recognition. "
        "In order to do so, first understand the image and its language. "
        "Then, provide the exact text in the language displayed in the image. "
        "Just give the complete text as your answer. "
        "Do NOT provide a description or any additional information!"
    )
    ocr_processor = PhiProcessor(
        model_id, prompt, device="cuda", max_new_tokens=5000, quantization_bits=8
    )

    directory_path = r"C:\Users\user\Documents\images"
    results = process_images_from_directory(directory_path, ocr_processor)
    for result in results:
        print(result)
