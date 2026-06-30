#!/usr/bin/env python3
"""
Fine-tuned Qwen2.5-Coder-7B LoRA 기반 코드 생성기
Zephyr TC JSON → Python Appium 테스트 코드 (단일 모델 + 플랫폼 조건부)

학습=추론 일치: 프롬프트(SYSTEM_PROMPT, elements/app_settings 컨텍스트, user 템플릿)를
collect_data.py에서 import해서 학습 데이터와 동일하게 구성. 어댑터는 adapters/ 의
최종 가중치(adapters.safetensors)를 사용.
"""

import re
from pathlib import Path

from collect_data import (
    SYSTEM_PROMPT, elements_context, appsettings_context,
    build_user_prompt, tc_to_training_format,
)

MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"  # HF id(캐시 자동탐색)
ADAPTER_DIR = Path(__file__).parent / "adapters"           # adapters.safetensors=최종


class MLXCodeGenerator:
    """Fine-tuned Qwen2.5-Coder-7B LoRA 기반 Zephyr TC → Python 코드 생성기."""

    def __init__(self, adapter_dir: str = None):
        self._model = None
        self._tokenizer = None
        self._adapter_dir = str(adapter_dir or ADAPTER_DIR)

    def _load(self):
        if self._model is not None:
            return
        from mlx_lm import load
        print("🔄 Qwen2.5-Coder-7B + LoRA 로드 중...")
        self._model, self._tokenizer = load(MODEL_ID, adapter_path=self._adapter_dir)
        print("✅ 모델 로드 완료")

    def generate(self, tc: dict, platform: str = "ios", max_tokens: int = 2000,
                 temp: float = 0.1, repetition_penalty: float = 1.1) -> str:
        """Zephyr TC(dict: tc_to_training_format 형식) → Python Appium 테스트 함수.

        학습과 동일하게 플랫폼별 elements/app_settings 컨텍스트를 주입한다.
        temp 0.1 + rep_penalty 1.1 이 실측상 가장 안정적이었다.
        장문 TC 일부는 반복루프가 남는데, 근본 원인이 학습 시 truncation(>4096)이라
        추론 튜닝으로는 해결이 안 되고 max_seq를 늘려 재학습해야 한다.
        """
        self._load()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        elem_ctx = elements_context(platform)
        app_ctx = appsettings_context(platform)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(tc, platform, elem_ctx, app_ctx)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        lp = (make_logits_processors(repetition_penalty=repetition_penalty)
              if repetition_penalty and repetition_penalty != 1.0 else None)
        response = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=False, sampler=make_sampler(temp=temp),
            logits_processors=lp,
        )
        return self._extract_code(response)

    def generate_from_jira_tc(self, jira_tc, platform: str = "ios", **kw) -> str:
        """JiraTestCase 객체를 받아 학습 포맷으로 변환 후 생성."""
        return self.generate(tc_to_training_format(jira_tc), platform=platform, **kw)

    def _extract_code(self, response: str) -> str:
        response = response.split("<|im_end|>")[0].strip()
        m = re.search(r"```python\s*(.*?)(?:```|$)", response, re.DOTALL)
        return (m.group(1) if m else response).strip()
