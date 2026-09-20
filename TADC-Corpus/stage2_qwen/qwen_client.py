# -*- coding: utf-8 -*-
"""DashScope/Qwen-VL client wrapper."""

from __future__ import annotations

import os
import time
from http import HTTPStatus
from typing import Optional

from dashscope import MultiModalConversation


class QwenVLClient:
    def __init__(self, model: str, max_retries: int = 2, retry_interval: float = 3.0):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. "
                "Please export it as an environment variable."
            )
        self.model = model
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def call(self, image_path: str, prompt: str) -> Optional[str]:
        messages = [{
            "role": "user",
            "content": [{"image": image_path}, {"text": prompt}],
        }]

        for attempt in range(self.max_retries):
            try:
                response = MultiModalConversation.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                )
                if response.status_code == HTTPStatus.OK:
                    content = response.output.choices[0]["message"]["content"]
                    return content[0]["text"] if isinstance(content, list) else content

                print(
                    f"[WARN] Qwen response error: "
                    f"{response.code} - {response.message}"
                )
                if getattr(response, "code", None) == "DataInspectionFailed":
                    return None
            except Exception as exc:
                print(
                    f"[WARN] Qwen exception "
                    f"(attempt {attempt + 1}/{self.max_retries}): {exc}"
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_interval)

        print("[ERROR] Maximum Qwen retries reached.")
        return None
