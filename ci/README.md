# Slack → GitHub Actions 테스트 자동화

팀원 누구나 **Slack 슬래시 커맨드 한 줄로** 모바일 자동화 테스트를 실행하고, 결과를 Slack에서 바로 받아보는 구조입니다. CI 콘솔에 들어가거나 명령을 외울 필요 없이 테스트 실행 진입장벽을 낮췄습니다.

## 흐름

```
Slack:  /test start android all 36042
          │
          ▼
slack_webhook_server.py  (Flask)
  - Slack 서명(HMAC) 검증 + 허용 사용자 체크
  - 명령 파싱 (플랫폼 · 디바이스 · 앱환경 · TC 필터 · Zephyr Cycle)
  - GitHub API 로 workflow_dispatch 트리거
          │
          ▼
GitHub Actions  (self-hosted runner)
  workflows/run-{ios,android}-tests.yml
  - Appium 서버/디바이스 준비 → pytest 실행
  - 결과를 Zephyr 에 동기화
          │
          ▼
slack_notifier.py
  - pytest JSON 결과 → Slack 블록으로 포맷 → 채널 회신
    (통과/실패 · 실패 케이스 목록 · 소요시간 · 워크플로 링크)
```

## 파일

- `slack_webhook_server.py` — Slack 슬래시 커맨드 수신 → 파싱 → GitHub Actions dispatch (`start`/`stop`/`status`/`generate`)
- `slack_notifier.py` — pytest 결과 → Slack 알림 포맷·전송
- `workflows/run-ios-tests.yml`, `workflows/run-android-tests.yml` — 디바이스에서 Appium 테스트 실행 + Zephyr 동기화 + Slack 알림

## 명령 예시

```
/test start                       # iOS 기본 디바이스, 베타 앱
/test start android all           # Android 전 디바이스
/test start all 36042,35554       # 특정 TC 복수 실행
/test start android except 43692  # 특정 TC 제외
/test stop                        # 실행 중인 테스트 중지
/test status                      # 최근 실행 상태
```

## 설계 포인트 (QA 관점)

- **진입장벽↓** — 콘솔/CLI 없이 Slack에서 누구나 트리거 → 팀 전체 접근성·가시성↑
- **유연한 범위 지정** — 플랫폼·디바이스·앱환경(staging/prod)·TC 필터/제외를 한 명령으로
- **결과 자동 회신 + Zephyr 동기화** — 실행 → 결과 → 리포팅까지 자동화
- **보안** — Slack 서명(HMAC) 검증 + 허용 사용자 ID 제한

> 익명화 참고: 사내 repo/Jira 호스트·프로젝트 키는 placeholder 로 치환했습니다. secret 은 모두 `${{ secrets.* }}` / 환경변수 참조(값 없음)입니다. 워크플로 내 경로는 원본 repo 레이아웃 기준의 참고용 아티팩트입니다.
