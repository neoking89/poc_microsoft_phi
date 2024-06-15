from typing import List, Dict
from transformers import AutoTokenizer

class ContextManagement:

    def __init__(self, max_available_tokens: int = 3000):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.max_available_tokens = max_available_tokens

    def __count_tokens__(self, content: str) -> int:
        return len(self.tokenizer.tokenize(content))

    def __pad_tokens__(self, content: str, num_tokens: int) -> str:
        tokens = self.tokenizer.encode(content, max_length=num_tokens, truncation=True)
        return self.tokenizer.decode(tokens)

    def __manage_context__(self, messages: List[Dict]) -> List[Dict]:
        managed_messages = []
        system_message = None
        if messages[0]["role"] == "system":
            system_message = messages[0]
        
        current_length = 0
        if system_message:
            current_length += self.__count_tokens__(system_message.get("content"))
        
        current_message_role = None
        for message in messages[1:]:
            content = message.get("content")
            message_tokens = self.__count_tokens__(content)
            
            if current_length + message_tokens >= self.max_available_tokens:
                tokens_to_keep = self.max_available_tokens - current_length
                if tokens_to_keep > 0:
                    content = self.__pad_tokens__(content, tokens_to_keep)
                    current_length += tokens_to_keep
                    managed_messages.append({"role": message.get("role"), "content": content})
                break
            else:
                if message.get("role") == current_message_role:
                    managed_messages[-1]["content"] += f"\n\n{content}"
                else:
                    managed_messages.append({"role": message.get("role"), "content": content})
                current_message_role = message.get("role")
                current_length += message_tokens

        managed_messages = managed_messages[::-1]
        if system_message:
            managed_messages.insert(0, system_message)
        return managed_messages

    def __create_message_input__(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def __call__(self, messages: List[Dict]) -> str:
        managed_messages = self.__manage_context__(messages)
        return self.__create_message_input__(managed_messages)
