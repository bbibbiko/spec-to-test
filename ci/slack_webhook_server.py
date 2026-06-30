import os
import re
import hmac
import hashlib
import time
import logging
import sys
import requests
from flask import Flask, request, jsonify
from urllib.parse import parse_qs
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_NOTIFICATION_WEBHOOK = os.environ.get("SLACK_NOTIFICATION_WEBHOOK")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "your-org/your-repo")
JIRA_URL = os.environ.get("JIRA_URL", "https://jira.example.com")
ZEPHYR_API_TOKEN = os.environ.get("ZEPHYR_API_TOKEN")
ALLOWED_USER_IDS = os.environ.get("ALLOWED_USER_IDS", "").split(",")
DEFAULT_CYCLE_ID = os.environ.get("DEFAULT_CYCLE_ID", "1234")

IOS_ENVIRONMENTS = {
    "se3": "iOS-SE3",
    "12mini": "iOS-12mini",
    "16e": "iOS-16e",
    "all": "iOS-All"
}
ANDROID_ENVIRONMENTS = {
    "fold6": "Android-ZFold6",
    "s23": "Android-S23Plus",
    "q52": "Android-Q52",
    "all": "Android-All"
}

DEFAULT_IOS_DEVICE = "se3"
DEFAULT_ANDROID_DEVICE = "fold6"


def send_to_notification_channel(message):
    if not SLACK_NOTIFICATION_WEBHOOK:
        logger.warning("SLACK_NOTIFICATION_WEBHOOK not set")
        return False
    try:
        response = requests.post(SLACK_NOTIFICATION_WEBHOOK, json={"text": message})
        logger.info(f"Notification sent: status={response.status_code}, message={message[:50]}...")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '\\"').replace("'", "\\'")
    return text.strip()


def clean_for_comment(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:80]


def verify_slack_signature(req):
    if not SLACK_SIGNING_SECRET:
        return True
    timestamp = req.headers.get("X-Slack-Request-Timestamp", "")
    signature = req.headers.get("X-Slack-Signature", "")
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False
    sig_basestring = f"v0:{timestamp}:{req.get_data(as_text=True)}"
    my_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(my_signature, signature)


def trigger_github_workflow(inputs, workflow_file="run-ios-tests.yml"):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN이 설정되지 않았습니다"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    payload = {"ref": "main", "inputs": inputs}
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code in [200, 204]:
            return True, None
        return False, f"GitHub API 오류: {response.status_code}"
    except Exception as e:
        return False, str(e)


def get_workflow_runs():
    if not GITHUB_TOKEN:
        return []
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/run-ios-tests.yml/runs"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.get(url, headers=headers, params={"per_page": 5})
        if response.status_code == 200:
            return response.json().get("workflow_runs", [])
    except:
        pass
    return []


def cancel_workflow_run(run_id):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN이 설정되지 않았습니다"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}/cancel"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 202:
            return True, None
        return False, f"취소 실패: {response.status_code}"
    except Exception as e:
        return False, str(e)


def search_cycle_by_name(cycle_name_or_id):
    if cycle_name_or_id.isdigit():
        return cycle_name_or_id, None

    if not ZEPHYR_API_TOKEN:
        return None, "ZEPHYR_API_TOKEN이 설정되지 않았습니다"

    try:
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from zephyr_client import ZephyrSquadClient

        client = ZephyrSquadClient(
            jira_url=JIRA_URL,
            api_token=ZEPHYR_API_TOKEN,
            project_key="QA"
        )

        cycles = client.get_cycles(version_id=-1)

        if not cycles:
            return None, "Cycle 목록을 가져올 수 없습니다"

        search_name_lower = cycle_name_or_id.lower()
        matches = []

        for cycle in cycles:
            cycle_name = cycle.get("name", "")
            if search_name_lower in cycle_name.lower():
                matches.append(cycle)

        exact_match = None
        for cycle in matches:
            if cycle.get("name", "").lower() == search_name_lower:
                exact_match = cycle
                break

        if exact_match:
            return exact_match["id"], exact_match["name"]

        if len(matches) == 1:
            return matches[0]["id"], matches[0]["name"]

        if len(matches) > 1:
            match_list = "\\n".join([f"• `{c['id']}`: {c['name']}" for c in matches[:5]])
            return None, f"여러 Cycle이 일치합니다:\\n{match_list}"

        similar = cycles[:5]
        similar_list = "\\n".join([f"• `{c['id']}`: {c['name']}" for c in similar])
        return None, f"일치하는 Cycle이 없습니다. 최근 Cycle:\\n{similar_list}"

    except Exception as e:
        return None, f"Cycle 검색 실패: {str(e)}"


def parse_devices_option(devices_str, platform):
    if platform == "ios":
        valid_devices = ["se3", "12mini", "16e"]
        env_map = IOS_ENVIRONMENTS
        default_device = DEFAULT_IOS_DEVICE
    else:
        valid_devices = ["fold6", "s23", "q52"]
        env_map = ANDROID_ENVIRONMENTS
        default_device = DEFAULT_ANDROID_DEVICE

    if not devices_str or devices_str == "default":
        return default_device, env_map.get(default_device, f"{platform.upper()}-Default"), default_device

    devices_str = devices_str.lower().strip()

    if devices_str == "all":
        return "all", env_map.get("all", f"{platform.upper()}-All"), "모든 디바이스"

    devices = [d.strip() for d in devices_str.split(",")]
    valid = [d for d in devices if d in valid_devices]

    if not valid:
        return default_device, env_map.get(default_device, f"{platform.upper()}-Default"), default_device

    if len(valid) == 1:
        return valid[0], env_map.get(valid[0], f"{platform.upper()}-{valid[0]}"), valid[0]

    devices_key = ",".join(valid)
    display_name = ", ".join(valid)
    environment = f"{platform.upper()}-Multi"
    return devices_key, environment, display_name


@app.route("/slack/test", methods=["POST"])
def handle_test_command():
    if not verify_slack_signature(request):
        return jsonify({"error": "Invalid signature"}), 401

    data = parse_qs(request.get_data(as_text=True))
    text = data.get("text", [""])[0].strip()
    user_id = data.get("user_id", [""])[0]
    user_name = data.get("user_name", [""])[0]

    if ALLOWED_USER_IDS and ALLOWED_USER_IDS[0]:
        if user_id not in ALLOWED_USER_IDS:
            return jsonify({"response_type": "ephemeral", "text": f":no_entry: 권한이 없습니다. ID: `{user_id}`"})

    if not text or text == "help":
        return jsonify({
            "response_type": "ephemeral",
            "text": "*테스트 명령어:*\n"
                    "*iOS 테스트:*\n"
                    "• `/test start` - iOS 기본 디바이스(SE3), 베타 앱\n"
                    "• `/test start production` - iOS 정식 앱 테스트\n"
                    "• `/test start all` - iOS 모든 디바이스 테스트\n"
                    "• `/test start production all` - 정식 앱 + 모든 디바이스\n"
                    "• `/test start all 36042` - 특정 TC 실행\n"
                    "• `/test start all 36042,35554` - 복수 TC 실행 (쉼표 구분)\n"
                    "• `/test start all 36042,35554 8256` - 복수 TC + Cycle\n"
                    "• `/test start all except 43692` - 특정 TC 제외\n"
                    "• `/test start production all 36042 8112` - 정식 + TC + Cycle\n"
                    "*Android 테스트:*\n"
                    "• `/test start android` - Android 베타 앱\n"
                    "• `/test start android production` - Android 정식 앱\n"
                    "• `/test start android all` - 모든 디바이스\n"
                    "• `/test start android all 36042,35554` - 복수 TC 실행\n"
                    "• `/test start android except 43692` - 특정 TC 제외\n"
                    "• `/test start android production all 36042` - 정식 + TC\n"
                    "*기타:*\n"
                    "• `/test stop` - 실행 중인 테스트 중지\n"
                    "• `/test status` - 테스트 상태 확인\n"
                    "• `/test generate <cycle_id/name>` - TC 생성\n"
                    "\n*참고:* `production` 또는 `prod`로 정식 앱 지정 가능"
        })

    parts = text.split()
    action = parts[0].lower()

    if action == "start":
        test_filter = ""
        cycle_id = DEFAULT_CYCLE_ID
        platform = "ios"
        workflow_file = "run-ios-tests.yml"
        devices = DEFAULT_IOS_DEVICE
        app_env = "staging"

        remaining_parts = parts[1:]
        if remaining_parts and remaining_parts[0].lower() == "android":
            platform = "android"
            workflow_file = "run-android-tests.yml"
            devices = DEFAULT_ANDROID_DEVICE
            remaining_parts = remaining_parts[1:]

        if remaining_parts and remaining_parts[0].lower() in ["production", "prod"]:
            app_env = "production"
            remaining_parts = remaining_parts[1:]

        cycle_name = None

        def is_tc_number(s):
            """단일 또는 쉼표 구분 복수 TC 번호 여부 (예: 35554 or 35554,36042)"""
            parts = [p.strip() for p in s.split(",")]
            return all(p.isdigit() and len(p) == 5 for p in parts)

        def build_tc_filter(s):
            """TC 번호(복수 가능)를 pytest -k 필터 문자열로 변환"""
            parts = [p.strip() for p in s.split(",")]
            return " or ".join(f"test_qa_{p}" for p in parts)

        def is_except_tc(s):
            """except <번호> 또는 except <번호,번호> 형식 여부"""
            if not s.lower().startswith("except"):
                return False
            rest = s[len("except"):].strip()
            return bool(rest) and is_tc_number(rest)

        def build_except_filter(s):
            """except <번호[,번호]>를 pytest -k not 필터 문자열로 변환"""
            rest = s[len("except"):].strip()
            parts = [p.strip() for p in rest.split(",")]
            return " and ".join(f"not test_qa_{p}" for p in parts)

        def is_cycle_id(s):
            return s.isdigit() and len(s) <= 4

        # except 사전 스캔: remaining_parts 어디에 있든 추출 후 제거
        # 예: ["fold6", "except", "43692"] → test_filter="not test_qa_43692", remaining=["fold6"]
        lower_parts = [p.lower() for p in remaining_parts]
        if "except" in lower_parts:
            except_idx = lower_parts.index("except")
            if except_idx + 1 < len(remaining_parts):
                except_tcs = remaining_parts[except_idx + 1]
                if is_tc_number(except_tcs):
                    test_filter = build_except_filter(f"except {except_tcs}")
                    remaining_parts = remaining_parts[:except_idx] + remaining_parts[except_idx + 2:]

        if len(remaining_parts) >= 1:
            first_arg = remaining_parts[0]

            is_device_option = False
            if platform == "ios":
                if first_arg.lower() in ["all", "se3", "12mini", "16e"]:
                    is_device_option = True
            else:
                if first_arg.lower() in ["all", "fold6", "s23", "q52"]:
                    is_device_option = True

            if is_device_option:
                devices = first_arg.lower()
                if len(remaining_parts) >= 2:
                    second_arg = remaining_parts[1]

                    if is_tc_number(second_arg):
                        test_filter = build_tc_filter(second_arg)
                        if len(remaining_parts) >= 3:
                            third_arg = remaining_parts[2]
                            if is_cycle_id(third_arg):
                                cycle_id = third_arg
                            else:
                                search_result_id, search_result_name_or_error = search_cycle_by_name(third_arg)
                                if search_result_id:
                                    cycle_id = search_result_id
                                    cycle_name = search_result_name_or_error
                    elif is_cycle_id(second_arg):
                        cycle_id = second_arg
                    else:
                        search_result_id, search_result_name_or_error = search_cycle_by_name(second_arg)
                        if search_result_id:
                            cycle_id = search_result_id
                            cycle_name = search_result_name_or_error
                        else:
                            test_filter = second_arg
                        if len(remaining_parts) >= 3:
                            third_arg = remaining_parts[2]
                            if is_cycle_id(third_arg):
                                cycle_id = third_arg
                            else:
                                search_result_id, search_result_name_or_error = search_cycle_by_name(third_arg)
                                if search_result_id:
                                    cycle_id = search_result_id
                                    cycle_name = search_result_name_or_error
            else:
                if is_tc_number(first_arg):
                    test_filter = build_tc_filter(first_arg)
                    if len(remaining_parts) >= 2:
                        second_arg = remaining_parts[1]
                        if is_cycle_id(second_arg):
                            cycle_id = second_arg
                        else:
                            search_result_id, search_result_name_or_error = search_cycle_by_name(second_arg)
                            if search_result_id:
                                cycle_id = search_result_id
                                cycle_name = search_result_name_or_error
                elif is_cycle_id(first_arg):
                    cycle_id = first_arg
                else:
                    search_result_id, search_result_name_or_error = search_cycle_by_name(first_arg)
                    if search_result_id:
                        cycle_id = search_result_id
                        cycle_name = search_result_name_or_error
                    else:
                        test_filter = first_arg

                        if len(remaining_parts) >= 2:
                            second_arg = remaining_parts[1]
                            if is_cycle_id(second_arg):
                                cycle_id = second_arg
                            else:
                                search_result_id, search_result_name_or_error = search_cycle_by_name(second_arg)
                                if search_result_id:
                                    cycle_id = search_result_id
                                    cycle_name = search_result_name_or_error

        devices_key, environment, devices_display = parse_devices_option(devices, platform)

        logger.info(f"[PARSE] platform={platform}, devices={devices_key}, test_filter='{test_filter}', cycle_id={cycle_id}, app_env={app_env}")

        if not cycle_name and cycle_id:
            search_result_id, search_result_name_or_error = search_cycle_by_name(cycle_id)
            if search_result_id:
                cycle_name = search_result_name_or_error

        inputs = {
            "test_filter": test_filter,
            "cycle_id": cycle_id,
            "environment": environment,
            "devices": devices_key,
            "app_env": app_env
        }

        logger.info(f"[START] Triggering workflow: {workflow_file}, inputs: {inputs}")
        success, error = trigger_github_workflow(inputs, workflow_file=workflow_file)
        logger.info(f"[START] Workflow trigger result: success={success}, error={error}")

        if success:
            platform_msg = "Android" if platform == "android" else "iOS"
            cycle_msg = f"`{cycle_name}` (ID: {cycle_id})" if cycle_name else f"`{cycle_id}`"
            env_msg = "Production (정식)" if app_env == "production" else "Staging (베타)"

            notification = f":rocket: *테스트 시작!*\n"
            notification += f"• 요청자: <@{user_id}>\n"
            notification += f"• 플랫폼: `{platform_msg}`\n"
            notification += f"• 환경: `{env_msg}`\n"
            notification += f"• 디바이스: `{devices_display}`\n"
            notification += f"• 테스트 케이스: `{test_filter or '전체'}`\n"
            notification += f"• 테스트 사이클: {cycle_msg}"

            logger.info(f"[START] Sending notification to channel...")
            result = send_to_notification_channel(notification)
            logger.info(f"[START] Notification result: {result}")
            return jsonify({"response_type": "ephemeral", "text": ":white_check_mark: 테스트 시작 요청 완료! 알림 채널을 확인하세요."})
        return jsonify({"response_type": "ephemeral", "text": f":x: 실패: {error}"})

    elif action == "stop":
        runs = get_workflow_runs()
        running = [r for r in runs if r.get("status") in ["queued", "in_progress"]]
        if not running:
            return jsonify({"response_type": "ephemeral", "text": ":warning: 실행 중인 테스트 없음"})
        run = running[0]
        success, error = cancel_workflow_run(run["id"])
        if success:
            send_to_notification_channel(f":octagonal_sign: *테스트 중지!* by <@{user_id}>\n• Run ID: `{run['id']}`")
            return jsonify({"response_type": "ephemeral", "text": ":white_check_mark: 테스트 중지 완료! 알림 채널을 확인하세요."})
        return jsonify({"response_type": "ephemeral", "text": f":x: 중지 실패: {error}"})

    elif action == "status":
        runs = get_workflow_runs()
        if not runs:
            return jsonify({"response_type": "ephemeral", "text": ":warning: 실행 기록 없음"})
        emoji_map = {"completed": ":white_check_mark:", "in_progress": ":runner:", "queued": ":hourglass:", "failure": ":x:"}
        lines = ["*최근 실행:*"]
        for r in runs[:5]:
            e = emoji_map.get(r.get("conclusion") or r.get("status"), ":grey_question:")
            lines.append(f"{e} `{r['id']}` - {r.get('status')} - <{r['html_url']}|보기>")
        return jsonify({"response_type": "ephemeral", "text": "\n".join(lines)})

    elif action == "generate":
        if len(parts) < 2:
            return jsonify({"response_type": "ephemeral", "text": ":x: 예: `/test generate 8112` 또는 `/test generate RC`"})

        cycle_input = parts[1]

        if cycle_input.isdigit():
            cycle_id = cycle_input
        else:
            search_result_id, search_result_name_or_error = search_cycle_by_name(cycle_input)
            if search_result_id:
                cycle_id = search_result_id
                cycle_name = search_result_name_or_error
                logger.info(f"Cycle 검색 성공: '{cycle_input}' -> ID {cycle_id} ({cycle_name})")
            else:
                return jsonify({"response_type": "ephemeral", "text": f":x: Cycle 검색 실패: {search_result_name_or_error}"})

        inputs = {"cycle_id": cycle_id, "slack_user_id": user_id}
        success, error = trigger_github_workflow(inputs, workflow_file="generate-tc.yml")
        if success:
            send_to_notification_channel(f":robot_face: *TC 생성 시작!* by <@{user_id}>\n• Cycle: `{cycle_id}`\n• 모델: 로컬 LLM 파이프라인\n• 로컬 Mac에서 실행 중...")
            return jsonify({"response_type": "ephemeral", "text": f":white_check_mark: TC 생성 시작! Cycle `{cycle_id}`\n알림 채널에서 진행 상황을 확인하세요."})
        return jsonify({"response_type": "ephemeral", "text": f":x: TC 생성 실패: {error}"})

    return jsonify({"response_type": "ephemeral", "text": f":x: 알 수 없는 명령: `{action}`"})


@app.route("/slack/result", methods=["POST"])
def handle_test_result():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data"}), 400

        status = data.get("status", "unknown")
        platform = data.get("platform", "iOS")
        test_filter = data.get("test_filter", "전체")
        cycle_id = data.get("cycle_id", "")
        duration = data.get("duration", "")
        run_url = data.get("run_url", "")
        summary = data.get("summary", {})

        if status == "success":
            emoji = ":white_check_mark:"
            status_text = "성공"
        elif status == "failure":
            emoji = ":x:"
            status_text = "실패"
        elif status == "cancelled":
            emoji = ":octagonal_sign:"
            status_text = "취소됨"
        else:
            emoji = ":grey_question:"
            status_text = status

        notification = f"{emoji} *테스트 완료 - {status_text}*\n"
        notification += f"• 플랫폼: `{platform}`\n"
        notification += f"• 테스트: `{test_filter or '전체'}`\n"
        if cycle_id:
            notification += f"• Cycle: `{cycle_id}`\n"
        if duration:
            notification += f"• 소요 시간: `{duration}`\n"

        if summary:
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)
            total = summary.get("total", passed + failed + skipped)
            notification += f"• 결과: 통과 {passed} / 실패 {failed} / 스킵 {skipped} (총 {total})\n"

        if run_url:
            notification += f"• <{run_url}|GitHub Actions 보기>"

        logger.info(f"[RESULT] Received test result: status={status}, platform={platform}")
        result = send_to_notification_channel(notification)
        logger.info(f"[RESULT] Notification sent: {result}")

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"[RESULT] Error handling test result: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
