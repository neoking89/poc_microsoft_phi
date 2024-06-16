# llm_invoke.py

import torch
from llama_cpp import Llama
from ctx import ContextManagement
from typing import List, Dict, Generator
from transformers import AutoTokenizer


class LLM:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_available_tokens: int = 2560,
        **kwargs
    ) -> None:
        """
        Initializes the LLM (Large Language Model) with specified parameters.

        Parameters
        ----------
        tokenizer_path : str
            The path to the tokenizer.
        model_path : str
            The path to the LLM model.
        max_available_tokens : int, optional
            Maximum tokens available for context management (default is 2560).
        **kwargs
            Additional keyword arguments for model configuration.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            seed=kwargs.get("seed", 1337),
            n_ctx=kwargs.get("n_ctx", 4096),
            n_threads=kwargs.get("n_threads", 8),
        )
        self.ctx = ContextManagement(tokenizer, max_available_tokens)
        self.check_gpu_availability()

    def check_gpu_availability(self) -> None:
        """
        Checks the availability of GPU and prints the result.
        """
        if torch.cuda.is_available():
            print("GPU is available. Using GPU.")
        else:
            print("GPU is not available. Using CPU.")

    def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Generator[str, None, None]:
        """
        Streams the output from the LLM based on the input messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            A list of messages to be processed by the LLM.
        **kwargs
            Additional keyword arguments for the LLM.

        Yields
        ------
        str
            Parts of the generated text by the LLM.
        """
        input_message = self.ctx(messages)
        input_message = self._strip_bos_token(input_message)
        output = self.llm(input_message, stream=True, echo=False, **kwargs)
        for op in output:
            yield op.get("choices")[0].get("text") or ""

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Completes the input messages using the LLM and returns the result.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            A list of messages to be processed by the LLM.
        **kwargs
            Additional keyword arguments for the LLM.

        Returns
        -------
        str
            The completed text generated by the LLM.
        """
        input_message = self.ctx(messages)
        input_message = self._strip_bos_token(input_message)
        output = self.llm(input_message, echo=False, **kwargs)
        return output.get("choices")[0].get("text")

    def _strip_bos_token(self, text: str) -> str:
        """
        Strips the beginning-of-sequence (BOS) token from the input text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without the BOS token.
        """
        bos_token = self.ctx.tokenizer.bos_token
        if text.startswith(bos_token):
            return text[len(bos_token) :]
        return text
