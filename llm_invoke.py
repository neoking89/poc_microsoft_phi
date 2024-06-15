import torch
from llama_cpp import Llama
from ctx import ContextManagement
from typing import List, Dict

class LLM:

    def __init__(self, model_path: str, max_available_tokens: int = 2560, **kwargs):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            seed=kwargs.get("seed", 1337),
            n_ctx=kwargs.get("n_ctx", 4096),
            n_threads=kwargs.get("n_threads", 8),
        )
        self.ctx = ContextManagement(max_available_tokens)
        self.check_gpu_availability()

    def check_gpu_availability(self):
        if torch.cuda.is_available():
            print("GPU is available. Using GPU.")
        else:
            print("GPU is not available. Using CPU.")

    def _strip_bos_token(self, text: str) -> str:
        bos_token = self.ctx.tokenizer.bos_token
        if text.startswith(bos_token):
            return text[len(bos_token) :]
        return text

    def __stream__(self, messages: List[Dict], **kwargs):
        input_message = self.ctx(messages)
        input_message = self._strip_bos_token(input_message)
        output = self.llm(input_message, stream=True, echo=False, **kwargs)
        for op in output:
            yield op.get("choices")[0].get("text") or ""

    def __complete__(self, messages: List[Dict], **kwargs):
        input_message = self.ctx(messages)
        input_message = self._strip_bos_token(input_message)
        output = self.llm(input_message, echo=False, **kwargs)
        return output.get("choices")[0].get("text")
