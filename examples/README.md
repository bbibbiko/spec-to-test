# 예시 데이터

실제 학습 데이터(사내 앱 테스트케이스·로케이터)는 포함하지 않습니다. 포맷과 흐름을 보여주기 위한 합성 예시입니다.

TC → 테스트 코드 생성은 다음과 같이 동작합니다.

- `sample_tc.json` — 입력. Zephyr/Jira에서 조회한 테스트케이스(TC)
- `sample_elements.py` — 컨텍스트. 화면별 로케이터 시그니처 (모델 프롬프트에 주입 → grounding)
- `sample_generated_test.py` — 출력. 모델이 생성한 Appium 코드 (학습 타깃 포맷)

학습 페어 구성 방식:

```
TC(JSON) + elements/app_settings 시그니처   →  user 프롬프트
손으로 검수·트림한 테스트 코드              →  assistant 타깃
                                            →  {"messages": [...]}  (JSONL 한 줄)
```

- `code_gen/collect_data.py` 가 `test_qa_*.py` ↔ TC 를 매핑해 페어 생성
- `code_gen/prepare_data.py` 가 train/valid 로 분할
- 추론 시에도 동일한 프롬프트(`build_user_prompt`)를 사용해 train=infer 를 일치시킴
