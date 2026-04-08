"""MCP server that exposes Claude Code skills as tools backed by Ollama subagents."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any, TextIO

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from .config import ServerConfig
from .inbox import Inbox
from .session import Session, SessionStore
from .skills import Skill, discover_skills
from .tools import BUILTIN_TOOLS

logger = logging.getLogger("subagent-mcp")

# Global state initialized at startup
_config: ServerConfig | None = None
_session_store: SessionStore | None = None
_inbox: Inbox | None = None
_skills: list[Skill] = []


@asynccontextmanager
async def _lifespan(_app: FastMCP[None]) -> AsyncIterator[None]:
    """Run async startup tasks before the server begins accepting requests."""
    await _check_ollama(_get_config())
    yield


mcp_server = FastMCP(
    "subagent-mcp",
    instructions=(
        "MCP server that proxies Claude Code skills to local Ollama models. "
        "Each tool corresponds to a discovered skill. Call a skill tool with "
        "your prompt and optionally specify a model and session_id to resume."
    ),
    lifespan=_lifespan,
)


def _get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = ServerConfig.load()
    return _config


def _get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(_get_config().session_dir)
    return _session_store


def _get_inbox() -> Inbox:
    global _inbox
    if _inbox is None:
        _inbox = Inbox(_get_config().inbox_dir)
    return _inbox


def _build_model(model_name: str | None = None) -> OpenAIChatModel:
    """Build a pydantic-ai model pointing at the Ollama endpoint."""
    config = _get_config()
    name = model_name or config.default_model
    return OpenAIChatModel(
        model_name=name,
        provider=OllamaProvider(base_url=f"{config.ollama_base_url}/v1"),
    )


def _build_agent(skill: Skill, model_name: str | None = None) -> Agent[None, str]:
    """Build a pydantic-ai agent for a skill with built-in tools."""
    model = _build_model(model_name)

    # Load skill content as system prompt context
    skill_content = ""
    if skill.source_path.exists():
        skill_content = skill.source_path.read_text()

    system_prompt = (
        f"You are executing the skill '{skill.name}'.\n\n"
        f"Skill definition:\n{skill_content}\n\n"
        "You have access to tools for reading/writing files, searching code, "
        "running shell commands, and more. Use them to accomplish the task.\n"
        "Be thorough but concise in your responses."
    )

    return Agent(
        model,
        system_prompt=system_prompt,
        tools=BUILTIN_TOOLS,  # type: ignore[arg-type]
    )


def _write_trailer(
    log: TextIO,
    status: str,
    exc: BaseException | None = None,
) -> None:
    """Best-effort write of a single-line completion trailer to the streaming log.

    Any I/O failure is swallowed via the module logger so the caller's
    original exception (if any) propagates unmasked. If ``exc`` is None,
    writes ``--- end {status} {iso_ts} ---``; otherwise appends
    ``: {ExceptionType}: {sanitized message}``. Newlines and carriage
    returns in the exception message are replaced with spaces so the
    trailer is always one line. ``str(exc)`` is itself wrapped in a
    try/except in case ``__str__`` raises.
    """
    try:
        ts = datetime.now(UTC).isoformat()
        if exc is None:
            log.write(f"\n--- end {status} {ts} ---\n")
        else:
            try:
                detail = str(exc).replace("\n", " ").replace("\r", " ")
            except Exception:  # noqa: BLE001 — defensive: __str__ may raise
                detail = "<unprintable>"
            log.write(
                f"\n--- end {status} {ts}: {type(exc).__name__}: {detail} ---\n"
            )
        log.flush()
    except Exception:  # noqa: BLE001 — best-effort; never mask caller's exception
        logger.exception("failed to write streaming log %s trailer", status)


async def _run_skill_streaming(
    agent: Agent[None, str],
    prompt: str,
    session: Session,
    store: SessionStore,
) -> tuple[str, list[ModelMessage]]:
    """Run the agent with streaming and write deltas to the session log.

    Opens ``{session_dir}/{session_id}.log`` in append mode, writes a
    timestamped ``--- prompt ---`` / ``--- response ---`` header block,
    then streams text deltas from ``agent.run_stream()`` to the file with
    per-chunk flush so concurrent tail readers see output in real time.

    Each turn is terminated by a trailer line so tail clients can detect
    completion: ``--- end ok {iso_ts} ---`` on success,
    ``--- end error {iso_ts}: {ExceptionType}: {message} ---`` on failure
    (any ``Exception`` escaping ``run_stream``/``stream_text``), or
    ``--- end cancelled {iso_ts}: {ExceptionType}: {message} ---`` on
    cancellation (``CancelledError`` from a client disconnect, or any
    other ``BaseException`` like ``KeyboardInterrupt``/``SystemExit``).
    Trailer writes are best-effort and never mask the original exception:
    if the trailer write itself fails, the failure is logged but the
    caller's exception is what propagates.

    Returns ``(final_output, all_messages)``.
    """
    log_path = store.log_path(session.session_id)
    header = (
        f"\n--- {datetime.now(UTC).isoformat()} prompt ---\n"
        f"{prompt}\n--- response ---\n"
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        try:
            async with agent.run_stream(
                prompt,
                message_history=session.messages or None,
            ) as result:
                async for chunk in result.stream_text(delta=True):
                    log.write(chunk)
                    log.flush()
                output = await result.get_output()
                messages = result.all_messages()
        except Exception as e:
            _write_trailer(log, "error", e)
            raise
        except BaseException as e:
            # Catches CancelledError (client disconnect), KeyboardInterrupt,
            # SystemExit, GeneratorExit. Write a distinct trailer so tail
            # clients can distinguish cancellation from normal completion or
            # error, then re-raise so the runtime tears down cleanly.
            _write_trailer(log, "cancelled", e)
            raise
        else:
            _write_trailer(log, "ok", None)
    return output, messages


def _emit_notification(
    session: Session,
    skill: Skill,
    status: str,
    summary: str,
) -> None:
    """Best-effort write of a completion notification to the inbox.

    Inbox writes must never mask the caller's exit path: they are
    swallowed via the module logger if anything goes wrong. The inbox
    is a separate file per record so a failed write does not corrupt
    the rest of the inbox. ``status`` is the inbox vocabulary
    (``ok``/``error``/``cancelled``), which mirrors the streaming-log
    trailer so observers can correlate.
    """
    try:
        _get_inbox().write(
            session_id=session.session_id,
            skill=skill.name,
            model=session.model,
            status=status,  # type: ignore[arg-type]
            summary=summary,
        )
    except Exception:  # noqa: BLE001 — best-effort; never mask the run result
        logger.exception(
            "failed to write inbox notification for session %s",
            session.session_id,
        )


async def _execute_skill_turn(
    skill: Skill,
    prompt: str,
    session: Session,
    model: str | None,
) -> dict[str, Any]:
    """Run one turn for a session under the per-session lock.

    Status transitions: ``running`` is set on entry and persisted so
    external observers (disk readers, ``list_sessions``) see the
    in-flight state. On success the status flips back to ``idle``; on
    any exception or cancellation it flips to ``failed``. ``last_active``
    is bumped at every transition.

    A completion notification is written to the inbox on every exit
    path (``ok`` / ``error`` / ``cancelled``) so external observers
    (the Phase 4 hook bridge, ``read_inbox`` callers) can detect
    completion without polling the session JSON.

    ``CancelledError`` (and any other ``BaseException``) propagates so
    cancellation tears down cleanly, but the failed status and the
    cancelled notification are persisted first on a best-effort basis.
    """
    store = _get_session_store()
    config = _get_config()

    async with store.lock(session.session_id):
        session.status = "running"
        session.last_active = datetime.now(UTC).isoformat()
        store.save(session)

        agent = _build_agent(skill, model or session.model)

        try:
            if config.streaming:
                output, messages = await _run_skill_streaming(
                    agent, prompt, session, store
                )
            else:
                result = await agent.run(
                    prompt,
                    message_history=session.messages or None,
                )
                output = result.output
                messages = result.all_messages()
        except Exception as e:
            session.status = "failed"
            session.last_active = datetime.now(UTC).isoformat()
            with suppress(Exception):
                store.save(session)
            error_msg = f"Error running skill '{skill.name}': {e}"
            logger.exception(error_msg)
            _emit_notification(session, skill, "error", str(e))
            return {
                "session_id": session.session_id,
                "error": error_msg,
                "model": session.model,
                "skill": skill.name,
                "status": "failed",
            }
        except BaseException as e:
            # CancelledError, KeyboardInterrupt, SystemExit. Persist a
            # failed status so the session is not stuck on "running"
            # forever, then re-raise so the runtime tears down cleanly.
            session.status = "failed"
            session.last_active = datetime.now(UTC).isoformat()
            with suppress(Exception):
                store.save(session)
            _emit_notification(
                session, skill, "cancelled", f"{type(e).__name__}: {e}"
            )
            raise

        session.messages = messages
        session.status = "idle"
        session.last_active = datetime.now(UTC).isoformat()
        store.save(session)
        _emit_notification(session, skill, "ok", output)

        return {
            "session_id": session.session_id,
            "response": output,
            "model": session.model,
            "skill": skill.name,
            "status": "idle",
        }


async def _run_skill(
    skill: Skill,
    prompt: str,
    model: str | None = None,
    session_id: str | None = None,
    run_in_background: bool = False,
) -> dict[str, Any]:
    """Execute a skill, optionally in the background.

    Foreground (default): runs to completion under the per-session lock
    and returns the final response (or error). Background: schedules
    the turn as an ``asyncio.Task`` tracked by the SessionStore and
    returns immediately with ``status="running"`` so the caller can
    poll the log via ``tail_session_log`` and the structured transcript
    via ``get_session_transcript`` once the run completes.

    Resume on a busy session is rejected with ``status="busy"`` —
    proper mailbox queueing arrives in Phase 5.
    """
    store = _get_session_store()
    config = _get_config()

    # Resume or create session. Resolution happens before acquiring the
    # lock so we have a session_id to lock on; for new sessions there can
    # be no concurrent writers anyway since the UUID is fresh.
    session: Session | None = None
    if session_id:
        session = store.get(session_id)
        if session is not None and store.is_running(session.session_id):
            # Fail-fast on busy sessions: do not block on the lock.
            return {
                "session_id": session.session_id,
                "status": "busy",
                "skill": skill.name,
                "model": session.model,
            }
    if session is None:
        effective_model = model or config.default_model
        session = store.create(skill.name, effective_model)

    if run_in_background:
        task = asyncio.create_task(
            _execute_skill_turn(skill, prompt, session, model)
        )
        store.register_task(session.session_id, task)
        return {
            "session_id": session.session_id,
            "status": "running",
            "skill": skill.name,
            "model": session.model,
            "log_path": str(store.log_path(session.session_id)),
        }

    return await _execute_skill_turn(skill, prompt, session, model)


def _register_skill_tool(skill: Skill) -> None:
    """Register an MCP tool for a specific skill."""

    @mcp_server.tool(
        name=f"skill_{skill.name.replace(':', '_').replace('-', '_')}",
        description=f"Run skill '{skill.name}': {skill.description[:200]}",
    )
    async def skill_tool(
        prompt: str,
        model: str = "",
        session_id: str = "",
        run_in_background: bool = False,
    ) -> str:
        result = await _run_skill(
            skill,
            prompt,
            model=model or None,
            session_id=session_id or None,
            run_in_background=run_in_background,
        )
        return json.dumps(result, indent=2)


# -- Session management tools --


@mcp_server.tool(
    name="list_sessions",
    description="List all subagent sessions with their IDs, skills, and message counts.",
)
async def list_sessions() -> str:
    store = _get_session_store()
    sessions = store.list_sessions()
    return json.dumps(sessions, indent=2)


@mcp_server.tool(
    name="get_session_transcript",
    description=(
        "Get the full structured transcript of a session by its UUID. "
        "During an active streaming run the in-flight turn is not yet visible "
        "(the session is only persisted after the stream completes) "
        "-- use tail_session_log to observe in-progress output."
    ),
)
async def get_session_transcript(session_id: str) -> str:
    store = _get_session_store()
    session = store.get(session_id)
    if session is None:
        return json.dumps({"error": f"Session {session_id} not found"})
    return json.dumps(session.to_dict(), indent=2)


@mcp_server.tool(
    name="list_available_skills",
    description="List all discovered skills that can be invoked.",
)
async def list_available_skills() -> str:
    return json.dumps([s.to_dict() for s in _skills], indent=2)


@mcp_server.tool(
    name="tail_session_log",
    description=(
        "Read new output from a running (or completed) session log. "
        "Pass offset=0 for the first call, then feed back the returned "
        "next_offset to poll for new content. Returns JSON with "
        "session_id, text (raw stream output), and next_offset. For the "
        "structured pydantic-ai message history of completed turns, use "
        "get_session_transcript."
    ),
)
async def tail_session_log(session_id: str, offset: int = 0) -> str:
    store = _get_session_store()
    text, next_offset = store.tail(session_id, offset=offset)
    return json.dumps({
        "session_id": session_id,
        "text": text,
        "next_offset": next_offset,
    })


@mcp_server.tool(
    name="read_inbox",
    description=(
        "Drain new completion notifications from the subagent inbox. "
        "Pass since='' on the first call to receive the most recent "
        "notifications (up to limit), then feed back the returned "
        "'head' as 'since' on subsequent calls to advance the cursor "
        "and receive only newer records. Returns JSON with "
        "'notifications' (list of completion records) and 'head' "
        "(the latest notification_id seen, or the input 'since' if no "
        "new records). Each notification carries session_id, skill, "
        "model, status (ok/error/cancelled), timestamp, and a short "
        "summary -- use get_session_transcript with the session_id "
        "for the full structured history."
    ),
)
async def read_inbox(since: str = "", limit: int = 50) -> str:
    inbox = _get_inbox()
    notifications, head = inbox.read(since=since, limit=limit)
    return json.dumps(
        {"notifications": notifications, "head": head},
        indent=2,
    )


@mcp_server.tool(
    name="run_skill_by_name",
    description=(
        "Run any skill by name with a prompt. Use this when you know the skill name "
        "but it wasn't registered as a dedicated tool. Set run_in_background=true "
        "to spawn the run as a background task and return immediately with "
        "status='running'; poll tail_session_log for live output and "
        "get_session_transcript once it completes. Resuming a busy session "
        "returns status='busy' (mailbox queueing arrives in a later phase)."
    ),
)
async def run_skill_by_name(
    skill_name: str,
    prompt: str,
    model: str = "",
    session_id: str = "",
    run_in_background: bool = False,
) -> str:
    matching = [s for s in _skills if s.name == skill_name]
    if not matching:
        return json.dumps({"error": f"Skill '{skill_name}' not found"})
    result = await _run_skill(
        matching[0],
        prompt,
        model=model or None,
        session_id=session_id or None,
        run_in_background=run_in_background,
    )
    return json.dumps(result, indent=2)


async def _check_ollama(config: ServerConfig) -> None:
    """Best-effort check that Ollama is reachable at startup."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{config.ollama_base_url}/api/tags", timeout=5.0
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            logger.info("Ollama reachable, %d models available", len(models))
    except Exception as e:
        logger.warning("Ollama not reachable at %s: %s", config.ollama_base_url, e)


def _initialize() -> None:
    """Discover skills and register them as MCP tools at startup."""
    global _skills, _config, _session_store, _inbox

    _config = ServerConfig.load()
    _session_store = SessionStore(_config.session_dir)
    _inbox = Inbox(_config.inbox_dir)

    logger.info("Discovering skills...")
    _skills = discover_skills()
    logger.info("Found %d skills", len(_skills))

    for skill in _skills:
        _register_skill_tool(skill)
        logger.info("Registered tool for skill: %s", skill.name)


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    _initialize()
    mcp_server.run()


if __name__ == "__main__":
    main()
