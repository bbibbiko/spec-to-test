# -*- coding: utf-8 -*-
import os
import sys
import json
import requests
from typing import Optional, Dict, Any

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


def send_slack_notification(
    webhook_url: str,
    results_file: str,
    environment: str = "iOS",
    test_cycle_key: Optional[str] = None,
    workflow_url: Optional[str] = None,
    devices: Optional[str] = None
) -> bool:
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        print("Sending notification about missing results...")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":warning: {environment} Test Incomplete",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Environment:* {environment}\n*Status:* Test execution failed or did not complete\n*Issue:* No test results file was generated"
                }
            }
        ]

        if test_cycle_key:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Zephyr Test Cycle:* `{test_cycle_key}`"
                }
            })

        if workflow_url:
            blocks.append({
                "type": "actions",
                "elements": [{
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Workflow Run",
                        "emoji": True
                    },
                    "url": workflow_url
                }]
            })

        blocks.append({"type": "divider"})

        payload = {
            "blocks": blocks,
            "attachments": [{
                "color": "#dc3545",
                "blocks": []
            }]
        }

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            print(f"Slack notification sent successfully (no results)")
            return True
        else:
            print(f"Failed to send Slack notification: {response.status_code} - {response.text}")
            return False

    with open(results_file, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    error = summary.get("error", 0)
    skipped = summary.get("skipped", 0)
    duration = data.get("duration", 0)

    pass_rate = (passed / total * 100) if total > 0 else 0

    if failed > 0 or error > 0:
        status_emoji = ":x:"
        status_text = "FAILED"
        color = "#dc3545"
    elif passed == total:
        status_emoji = ":white_check_mark:"
        status_text = "PASSED"
        color = "#28a745"
    else:
        status_emoji = ":warning:"
        status_text = "PARTIAL"
        color = "#ffc107"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} {environment} Test Results",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Status:*\n{status_text}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Pass Rate:*\n{pass_rate:.1f}%"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Environment:*\n{environment}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Devices:*\n{devices or 'default'}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Total Tests:*\n{total}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Duration:*\n{duration:.1f}s"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Passed:*\n:white_check_mark: {passed}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Failed:*\n:x: {failed}"
                }
            ]
        }
    ]

    if test_cycle_key:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Zephyr Test Cycle:* `{test_cycle_key}`"
            }
        })

    if failed > 0:
        tests = data.get("tests", [])
        failed_tests = [t for t in tests if t.get("outcome") == "failed"]

        failed_list = []
        for t in failed_tests[:5]:
            nodeid = t.get("nodeid", "unknown")
            test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
            failed_list.append(f"• `{test_name}`")

        if failed_list:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Failed Tests:*\n" + "\n".join(failed_list)
                }
            })

            if len(failed_tests) > 5:
                blocks.append({
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"_...and {len(failed_tests) - 5} more failures_"
                    }]
                })

    if workflow_url:
        blocks.append({
            "type": "actions",
            "elements": [{
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Workflow Run",
                    "emoji": True
                },
                "url": workflow_url
            }]
        })

    blocks.append({"type": "divider"})

    payload = {
        "blocks": blocks,
        "attachments": [{
            "color": color,
            "blocks": []
        }]
    }

    response = requests.post(
        webhook_url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print(f"Slack notification sent successfully")
        return True
    else:
        print(f"Failed to send Slack notification: {response.status_code} - {response.text}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send test results to Slack")
    parser.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL"), help="Slack webhook URL")
    parser.add_argument("--results", required=True, help="Path to pytest JSON report")
    parser.add_argument("--environment", default="iOS", help="Test environment name")
    parser.add_argument("--cycle", help="Zephyr test cycle key")
    parser.add_argument("--devices", help="Devices tested")
    parser.add_argument("--workflow-url", help="GitHub Actions workflow URL")

    args = parser.parse_args()

    if not args.webhook:
        print("Error: SLACK_WEBHOOK_URL environment variable or --webhook required")
        exit(1)

    success = send_slack_notification(
        webhook_url=args.webhook,
        results_file=args.results,
        environment=args.environment,
        test_cycle_key=args.cycle,
        workflow_url=args.workflow_url,
        devices=args.devices
    )

    exit(0 if success else 1)
