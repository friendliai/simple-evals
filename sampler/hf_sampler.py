import torch
import transformers
from typing import Any

from ..types import MessageList, SamplerBase


class HfChatCompletionSampler(SamplerBase):
    """
    Sample from Huggingface's chat completion API
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "meta-llama/Meta-Llama-3.1-8B-instruct",
        device: str = "cuda",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
        )
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def __repr__(self):
        return f"model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens}"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        raise NotImplementedError()

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        outputs = self.pipeline(
            message_list,
            temperature=self.temperature,
            max_length=self.max_tokens,
            do_sample=False,
            top_p=1
        )
        print(outputs[0]["generated_text"][-1])
        return outputs[0]["generated_text"][-1]
