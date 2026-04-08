#!/usr/bin/env bash
# Thin shell shim that execs the Python notification hook.
#
# This file exists so users can wire a single absolute path into their
# Claude Code settings.json `UserPromptSubmit` hook without worrying
# about which python interpreter is on PATH or how the script was
# installed. All real logic lives in notification_hook.py alongside.
#
# Usage in .claude/settings.json:
#
#   {
#     "hooks": {
#       "UserPromptSubmit": [
#         {
#           "hooks": [
#             { "type": "command",
#               "command": "/abs/path/to/scripts/notification-hook.sh" }
#           ]
#         }
#       ]
#     }
#   }
#
# Environment:
#   SUBAGENT_MCP_INBOX_DIR  inbox directory (default: .subagent-inbox)

set -euo pipefail
exec python3 "$(dirname "$0")/notification_hook.py" "$@"
