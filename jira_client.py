#!/usr/bin/env python3
"""
Jira/Zephyr API 클라이언트 (Python)
"""

import requests
from typing import List, Dict, Any, Optional
import json


class JiraTestStep:
    """Jira 테스트 스텝"""
    def __init__(self, index: int, description: str, test_data: str = "", expected_result: str = ""):
        self.index = index
        self.description = description
        self.test_data = test_data
        self.expected_result = expected_result

    def to_text(self) -> str:
        """텍스트 형식으로 변환"""
        text = f"### Step {self.index}\n"
        text += f"**실행**: {self.description}\n"
        if self.test_data:
            text += f"**테스트 데이터**: {self.test_data}\n"
        text += f"**예상 결과**: {self.expected_result}\n\n"
        return text


class JiraTestCase:
    """Jira 테스트 케이스"""
    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name
        self.objective: str = ""
        self.precondition: str = ""
        self.steps: List[JiraTestStep] = []
        self.priority: str = ""
        self.status: str = ""
        self.labels: List[str] = []
        self.custom_fields: Dict[str, Any] = {}

    def to_text(self) -> str:
        """텍스트 형식으로 변환"""
        text = f"# 테스트 케이스: {self.key}\n\n"
        text += f"**테스트 케이스 제목(요약)**: {self.name}\n\n"

        if self.objective:
            text += f"**목적**: {self.objective}\n\n"

        if self.precondition:
            text += f"**사전조건**: {self.precondition}\n\n"

        if self.priority:
            text += f"**우선순위**: {self.priority}\n\n"

        if self.status:
            text += f"**상태**: {self.status}\n\n"

        if self.labels:
            text += f"**레이블**: {', '.join(self.labels)}\n\n"

        if self.steps:
            text += f"## 테스트 단계 ({len(self.steps)}개)\n\n"
            for step in self.steps:
                text += step.to_text()

        return text


class JiraClient:
    """Jira API 클라이언트"""

    def __init__(self, base_url: str, api_token: str):
        """
        Jira API 클라이언트 초기화 (Bearer 토큰 방식)

        Args:
            base_url: Jira 베이스 URL (예: https://jira.yourcompany.com)
            api_token: Personal Access Token (Bearer 토큰)
        """
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()

        # Bearer 토큰 인증
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def test_connection(self) -> tuple[bool, str]:
        """연결 테스트"""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/2/myself",
                timeout=10
            )
            if response.status_code == 200:
                user_data = response.json()
                return True, f"✓ 연결 성공: {user_data.get('displayName', 'Unknown')}"
            else:
                return False, f"✗ 연결 실패: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"✗ 연결 실패: {str(e)}"

    def get_test_case(self, issue_key: str, debug: bool = False) -> JiraTestCase:
        """테스트 케이스 가져오기"""
        try:
            # Jira 이슈 정보 가져오기
            response = self.session.get(
                f"{self.base_url}/rest/api/2/issue/{issue_key}",
                timeout=30
            )
            response.raise_for_status()
            issue_data = response.json()

            if debug:
                print(f"[DEBUG] 이슈 정보: key={issue_data.get('key')}, id={issue_data.get('id')}")

            # 테스트 케이스 객체 생성
            test_case = JiraTestCase(
                key=issue_data['key'],
                name=issue_data['fields'].get('summary', '')
            )

            # 기본 필드
            test_case.objective = issue_data['fields'].get('description', '')
            test_case.priority = issue_data['fields'].get('priority', {}).get('name', '')
            test_case.status = issue_data['fields'].get('status', {}).get('name', '')
            test_case.labels = issue_data['fields'].get('labels', [])

            # Zephyr Scale 테스트 스텝 시도
            try:
                if debug:
                    print(f"[DEBUG] Zephyr Scale API 시도 중...")
                test_case.steps = self._get_zephyr_scale_steps(issue_key)
                if debug:
                    print(f"[DEBUG] Zephyr Scale: {len(test_case.steps)}개 스텝 발견")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Zephyr Scale 실패: {e}")
                # Zephyr Squad 테스트 스텝 시도
                try:
                    if debug:
                        print(f"[DEBUG] Zephyr Squad API 시도 중... (issue_id={issue_data['id']})")
                    test_case.steps = self._get_zephyr_squad_steps(issue_data['id'])
                    if debug:
                        print(f"[DEBUG] Zephyr Squad: {len(test_case.steps)}개 스텝 발견")
                except Exception as e2:
                    if debug:
                        print(f"[DEBUG] Zephyr Squad 실패: {e2}")
                        print(f"[DEBUG] Description 파싱 시도 중...")
                    # 일반 Jira 이슈 - description에서 스텝 파싱
                    test_case.steps = self._parse_steps_from_description(
                        issue_data['fields'].get('description', '')
                    )
                    if debug:
                        print(f"[DEBUG] Description 파싱: {len(test_case.steps)}개 스텝 발견")

            return test_case

        except Exception as e:
            raise Exception(f"테스트 케이스를 가져올 수 없습니다: {str(e)}")

    def _get_zephyr_scale_steps(self, test_case_key: str) -> List[JiraTestStep]:
        """Zephyr Scale 테스트 스텝 가져오기"""
        response = self.session.get(
            f"{self.base_url}/rest/atm/1.0/testcase/{test_case_key}",
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        steps = []
        if 'testScript' in data and 'steps' in data['testScript']:
            for idx, step_data in enumerate(data['testScript']['steps'], 1):
                step = JiraTestStep(
                    index=idx,
                    description=step_data.get('description', '') or step_data.get('action', ''),
                    test_data=step_data.get('testData', '') or step_data.get('data', ''),
                    expected_result=step_data.get('expectedResult', '') or step_data.get('expected', '')
                )
                steps.append(step)

        return steps

    def _get_zephyr_squad_steps(self, issue_id: str) -> List[JiraTestStep]:
        """Zephyr Squad 테스트 스텝 가져오기"""
        response = self.session.get(
            f"{self.base_url}/rest/zapi/latest/teststep/{issue_id}",
            timeout=30
        )
        response.raise_for_status()
        steps_data = response.json()

        steps = []

        # 응답이 리스트인 경우 그대로 사용
        if isinstance(steps_data, list):
            step_list = steps_data
        # 응답이 딕셔너리인 경우 stepBeanCollection 또는 steps 추출
        elif isinstance(steps_data, dict):
            step_list = steps_data.get("stepBeanCollection", steps_data.get("steps", []))
        else:
            return steps

        # 스텝 파싱
        for idx, step_data in enumerate(step_list, 1):
            if isinstance(step_data, dict):
                step = JiraTestStep(
                    index=idx,
                    description=step_data.get('step', ''),
                    test_data=step_data.get('data', ''),
                    expected_result=step_data.get('result', '')
                )
                steps.append(step)

        return steps

    def _parse_steps_from_description(self, description: str) -> List[JiraTestStep]:
        """Description에서 테스트 스텝 파싱 (일반 Jira 이슈용)"""
        steps = []
        if not description:
            return steps

        # 간단한 파싱 - 숫자로 시작하는 줄을 스텝으로 간주
        lines = description.split('\n')
        current_step = None
        step_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 스텝 시작 패턴: "1.", "1)", "Step 1:", etc.
            import re
            step_match = re.match(r'^(?:Step\s+)?(\d+)[\.\)\:]?\s+(.+)', line, re.IGNORECASE)
            if step_match:
                step_index = int(step_match.group(1))
                description = step_match.group(2)
                current_step = JiraTestStep(
                    index=step_index,
                    description=description,
                    test_data="",
                    expected_result=""
                )
                steps.append(current_step)
            elif current_step:
                # 현재 스텝에 내용 추가
                if line.lower().startswith(('expected:', 'result:')):
                    current_step.expected_result = line.split(':', 1)[1].strip()
                elif line.lower().startswith(('data:', 'input:')):
                    current_step.test_data = line.split(':', 1)[1].strip()
                else:
                    current_step.description += " " + line

        return steps

    # ───────────────────────── 업로드 (Zephyr Squad / ZAPI) ─────────────────────────
    #
    # 생성된 TC(generate_tc 출력: zephyr_tc.steps=[{step,data,result}])를 Zephyr Squad에 업로드.
    # 인증은 기존과 동일한 Bearer PAT. 두 가지 방식 지원:
    #   - 새 Test 이슈 생성: project_key 지정 → Jira 이슈(Test) 생성 후 그 이슈에 스텝 추가
    #   - 기존 키에 추가:    existing_key 지정 → 해당 이슈에 스텝 추가
    # 외부 시스템에 실제로 쓰는 작업이므로 dry_run(미리보기)과 명확한 에러를 제공.

    def resolve_issue_id(self, issue_key: str) -> str:
        """이슈 키 → 내부 수치 id (ZAPI teststep 엔드포인트가 id를 요구)."""
        resp = self.session.get(f"{self.base_url}/rest/api/2/issue/{issue_key}",
                                params={"fields": "id"}, timeout=30)
        resp.raise_for_status()
        return resp.json()["id"]

    def _fetch_project_issue_types(self, project_key: str) -> List[dict]:
        """프로젝트에서 사용 가능한 이슈타입 목록 조회 (Jira 버전 무관 폴백).

        1) GET /rest/api/2/project/{key} 의 issueTypes (Server/DC·Cloud 공통)
        2) 비면 GET /rest/api/2/issue/createmeta/{key}/issuetypes (Jira 9+ 신엔드포인트)
        """
        # 1) 프로젝트 엔드포인트 (구버전 createmeta 제거된 Jira 9.x 대응)
        resp = self.session.get(
            f"{self.base_url}/rest/api/2/project/{project_key}", timeout=30)
        if resp.status_code == 200:
            types = resp.json().get("issueTypes", []) or []
            if types:
                return types
        elif resp.status_code == 404:
            raise Exception(
                f"프로젝트 '{project_key}'를 찾을 수 없습니다. 프로젝트 키를 확인하세요.")

        # 2) Jira 9+ 신 createmeta 엔드포인트
        resp2 = self.session.get(
            f"{self.base_url}/rest/api/2/issue/createmeta/{project_key}/issuetypes",
            timeout=30)
        if resp2.status_code == 200:
            data = resp2.json()
            return data.get("values", data.get("issueTypes", [])) or []

        raise Exception(
            f"이슈타입 조회 실패 (project={project_key}): "
            f"{resp.status_code} {resp.text[:200]} / {resp2.status_code} {resp2.text[:200]}")

    def resolve_issue_type(self, project_key: str, issue_type: str) -> Dict[str, str]:
        """프로젝트에서 이슈타입 이름을 id로 해석 → {'id','name'}.

        이름 매칭은 대소문자·공백 무시. 못 찾으면 사용 가능한 타입 목록과 함께 에러.
        """
        types = self._fetch_project_issue_types(project_key)
        if not types:
            raise Exception(
                f"프로젝트 '{project_key}'의 이슈타입을 가져오지 못했습니다 "
                f"(권한 또는 프로젝트 설정 확인).")
        want = (issue_type or "").strip().lower()
        for t in types:
            if str(t.get("name", "")).strip().lower() == want:
                return {"id": str(t.get("id", "")), "name": t.get("name", "")}
        names = ", ".join(str(t.get("name", "")) for t in types)
        raise Exception(
            f"이슈타입 '{issue_type}'을(를) 프로젝트 '{project_key}'에서 찾을 수 없습니다. "
            f"사용 가능: [{names}]")

    _priority_cache: Optional[List[dict]] = None

    def resolve_priority(self, priority: str) -> Optional[Dict[str, str]]:
        """우선순위 이름을 Jira 실제 우선순위(id/name)로 해석 (대소문자 무시).

        예: 'Minor' ↔ 'minor' 자동 매칭. 못 찾으면 None (이름 그대로 전송).
        """
        if not priority:
            return None
        if self._priority_cache is None:
            try:
                resp = self.session.get(
                    f"{self.base_url}/rest/api/2/priority", timeout=30)
                self._priority_cache = resp.json() if resp.status_code == 200 else []
            except Exception:
                self._priority_cache = []
        want = priority.strip().lower()
        for p in self._priority_cache:
            if str(p.get("name", "")).strip().lower() == want:
                return {"id": str(p.get("id", "")), "name": p.get("name", "")}
        return None

    _current_user: Optional[dict] = None

    def current_user(self) -> Optional[dict]:
        """PAT 사용자(보고자) 정보 조회 → assignee 지정용. 1회 캐시.

        반환에 'name'(Server/DC) 또는 'accountId'(Cloud)가 들어감.
        """
        if self._current_user is None:
            try:
                resp = self.session.get(f"{self.base_url}/rest/api/2/myself", timeout=15)
                self._current_user = resp.json() if resp.status_code == 200 else {}
            except Exception:
                self._current_user = {}
        return self._current_user or None

    def _assignee_field(self) -> Optional[dict]:
        """PAT 사용자를 가리키는 assignee 필드 값 (담당자=보고자)."""
        u = self.current_user() or {}
        if u.get("name"):            # Jira Server/DC
            return {"name": u["name"]}
        if u.get("accountId"):       # Jira Cloud
            return {"accountId": u["accountId"]}
        return None

    def create_test_issue(self, project_key: str, summary: str,
                          description: str = "", issue_type: str = "Test case",
                          priority: str = "", labels: Optional[List[str]] = None,
                          fix_versions: Optional[List[str]] = None,
                          components: Optional[List[str]] = None) -> Dict[str, str]:
        """새 Jira Test 이슈 생성 → {'key','id'} 반환. 담당자=보고자(PAT 사용자)로 지정."""
        it = self.resolve_issue_type(project_key, issue_type)
        fields = {
            "project": {"key": project_key},
            "summary": summary[:255] if summary else "(제목 없음)",
            "issuetype": {"id": it["id"]},
        }
        assignee = self._assignee_field()
        if priority:
            pr = self.resolve_priority(priority)
            fields["priority"] = {"id": pr["id"]} if pr else {"name": priority}
        if labels:
            fields["labels"] = [l for l in labels if l]
        if fix_versions:
            fields["fixVersions"] = [{"name": v} for v in fix_versions if v]
        if components:
            fields["components"] = [{"name": c} for c in components if c]
        body = {"fields": fields}
        if description:
            body["fields"]["description"] = description
        resp = self.session.post(f"{self.base_url}/rest/api/2/issue",
                                 data=json.dumps(body), timeout=30)
        if resp.status_code not in (200, 201):
            raise Exception(f"이슈 생성 실패 {resp.status_code}: {resp.text}")
        d = resp.json()
        key = d.get("key", "")
        # 담당자=보고자(PAT 사용자)로 지정 — 전용 엔드포인트(생성 화면에 없어도 동작).
        # 실패해도 이슈 생성은 유지(경고만).
        if assignee and key:
            try:
                ar = self.session.put(
                    f"{self.base_url}/rest/api/2/issue/{key}/assignee",
                    data=json.dumps(assignee), timeout=15)
                if ar.status_code not in (200, 204):
                    print(f"  ⚠️ 담당자 지정 실패({ar.status_code}): {ar.text[:120]}")
            except Exception as e:
                print(f"  ⚠️ 담당자 지정 예외: {e}")
        return {"key": key, "id": str(d.get("id", ""))}

    def add_test_step(self, issue_id: str, step: str, data: str = "",
                      result: str = "") -> dict:
        """ZAPI: 이슈에 테스트 스텝 1개 추가 (step/data/result = Squad 필드)."""
        body = {"step": step or "", "data": data or "", "result": result or ""}
        resp = self.session.post(f"{self.base_url}/rest/zapi/latest/teststep/{issue_id}",
                                 data=json.dumps(body), timeout=30)
        if resp.status_code not in (200, 201):
            raise Exception(f"스텝 추가 실패 {resp.status_code}: {resp.text}")
        return resp.json() if resp.text else {}

    # 우선순위(모델 high/medium/low) → Jira 우선순위명 + 레이블
    PRIORITY_MAP = {"high": "Major", "medium": "Minor", "low": "Trivial"}
    LABEL_MAP = {"high": "중요도_상", "medium": "중요도_중", "low": "중요도_하"}

    def _severity_fields(self, severity: str) -> tuple:
        """severity(high/medium/low) → (Jira 우선순위명, 레이블). high이상=상, low이하=하."""
        sev = (severity or "medium").strip().lower()
        # Jira 우선순위명으로 들어와도 역매핑
        rev = {"major": "high", "minor": "medium", "trivial": "low"}
        sev = rev.get(sev, sev)
        if sev not in self.PRIORITY_MAP:
            sev = "medium"
        return self.PRIORITY_MAP[sev], self.LABEL_MAP[sev]

    def upload_test_case(self, zephyr_tc: dict, project_key: str = "",
                         existing_key: str = "", dry_run: bool = False,
                         issue_type: str = "Test case", severity: str = "medium",
                         fix_versions: Optional[List[str]] = None,
                         components: Optional[List[str]] = None) -> dict:
        """단일 TC(zephyr_tc={summary,steps:[{step,data,result}]}) 업로드.

        existing_key 있으면 그 이슈에 스텝 추가, 없으면 project_key로 새 Test 이슈 생성.
        severity → Jira 우선순위 + 중요도 레이블. fix_versions → fixVersion 등록.
        반환: {'key','id','steps_added','dry_run','summary','priority','label'}.
        """
        summary = zephyr_tc.get("summary", "") or "(제목 없음)"
        steps = zephyr_tc.get("steps", []) or []
        priority, label = self._severity_fields(severity)

        if dry_run:
            target = existing_key or f"(새 이슈 @ {project_key})"
            return {"key": existing_key, "id": "", "steps_added": len(steps),
                    "dry_run": True, "summary": summary, "target": target,
                    "priority": priority, "label": label,
                    "fix_versions": fix_versions or [], "components": components or []}

        if existing_key:
            key = existing_key
            issue_id = self.resolve_issue_id(existing_key)
        else:
            if not project_key:
                raise Exception("project_key 또는 existing_key 중 하나는 필수입니다.")
            created = self.create_test_issue(
                project_key, summary, issue_type=issue_type,
                priority=priority, labels=[label], fix_versions=fix_versions,
                components=components)
            key, issue_id = created["key"], created["id"]

        added = 0
        for s in steps:
            self.add_test_step(issue_id,
                               step=s.get("step", ""),
                               data=s.get("data", ""),
                               result=s.get("result", ""))
            added += 1
        return {"key": key, "id": issue_id, "steps_added": added,
                "dry_run": False, "summary": summary,
                "priority": priority, "label": label,
                "fix_versions": fix_versions or [], "components": components or []}

    def upload_generated_tc(self, generated: dict, project_key: str = "",
                            existing_key: str = "", dry_run: bool = False,
                            issue_type: str = "Test case", severity: str = "",
                            fix_versions: Optional[List[str]] = None,
                            components: Optional[List[str]] = None) -> List[dict]:
        """generate_tc 결과(requirements=[{zephyr_tc}...]) 전체 업로드.

        requirement가 여러 개면 각각 별도 TC. existing_key는 단일 TC일 때만 의미.
        severity가 비면 각 req의 priority 필드 사용(없으면 medium). fix_versions는 공통.
        반환: 업로드 결과 리스트.
        """
        reqs = (generated.get("requirements")
                or generated.get("testable_requirements") or [])
        if not reqs and "zephyr_tc" in generated:
            reqs = [generated]
        if not reqs:
            raise Exception("업로드할 requirements가 없습니다 (생성 결과 확인).")

        if existing_key and len(reqs) > 1:
            raise Exception(
                f"기존 키 모드는 TC 1개만 가능한데 {len(reqs)}개입니다. "
                "새 이슈 생성 모드를 쓰거나 1개만 업로드하세요.")

        results = []
        for req in reqs:
            ztc = req.get("zephyr_tc") or req
            sev = severity or req.get("priority") or "medium"
            results.append(self.upload_test_case(
                ztc, project_key=project_key,
                existing_key=existing_key, dry_run=dry_run, issue_type=issue_type,
                severity=sev, fix_versions=fix_versions, components=components))
        return results

    def search_test_cases(self, jql: str, max_results: int = 50) -> List[JiraTestCase]:
        """JQL로 테스트 케이스 검색"""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/2/search",
                params={
                    'jql': jql,
                    'maxResults': max_results,
                    'fields': 'summary,description,status,priority,labels'
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            test_cases = []
            for issue in data.get('issues', []):
                try:
                    test_case = self.get_test_case(issue['key'])
                    test_cases.append(test_case)
                except Exception as e:
                    print(f"⚠ {issue['key']} 가져오기 실패: {e}")

            return test_cases

        except Exception as e:
            raise Exception(f"테스트 케이스 검색 실패: {str(e)}")
