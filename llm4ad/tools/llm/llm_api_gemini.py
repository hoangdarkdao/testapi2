from __future__ import annotations

import http.client
import json
import time
from typing import Any
import traceback
from urllib.parse import urlencode
from ...base import LLM
import itertools

import google.generativeai as genai

class GeminiApi(LLM):
    def __init__(self, keys, model, timeout=60, **kwargs):
        """
        keys: list các API keys (hoặc 1 key string)
        """
        super().__init__(**kwargs)

        if isinstance(keys, str):
            keys = [keys]

        self._keys_cycle = itertools.cycle(keys)  # vòng xoay vô hạn
        self._current_key = next(self._keys_cycle)

        genai.configure(api_key=self._current_key)
        self._model = genai.GenerativeModel(model)
        self.model_name = model
        self._timeout = timeout
        self._kwargs = kwargs

    def _switch_key(self):
        """Chuyển sang API key kế tiếp"""
        self._current_key = next(self._keys_cycle)
        print(f"[INFO] Switched to new API key: {self._current_key[:8]}...")
        genai.configure(api_key=self._current_key)
        self._model = genai.GenerativeModel(self.model_name)

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        try:
            print(f"[INFO] Calling Gemini model: {self._model.model_name} with key {self._current_key[:8]}...")
            response = self._model.generate_content(prompt)
            result = response.text
            # 🔄 Sau khi gọi thành công thì đổi sang key kế tiếp
            self._switch_key()
            return result
        except Exception:
            print(f"[ERROR] Gemini API call failed:\n{traceback.format_exc()}")
            # Thử chuyển sang key khác và gọi lại
            self._switch_key()
            try:
                response = self._model.generate_content(prompt)
                result = response.text
                # Sau khi gọi lại thì tiếp tục xoay key
                self._switch_key()
                return result
            except Exception:
                print(f"[ERROR] Gemini API call failed again:\n{traceback.format_exc()}")
                return "API_FAILED"