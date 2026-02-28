# -*- coding: utf-8 -*-
"""OneBot Channel.

OneBot uses forward WebSocket to connect to NapCatQQ, Go-CQHttp etc.
It allows receiving messages from the WebSocket and sending them back via the same connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from agentscope_runtime.engine.schemas.agent_schemas import (
    RunStatus,
    TextContent,
    ContentType,
)

from ..base import (
    BaseChannel,
    OnReplySent,
    OutgoingContentPart,
    ProcessHandler,
)

logger = logging.getLogger(__name__)


class OneBotChannel(BaseChannel):
    """OneBot Channel: Forward WebSocket implementation."""

    channel = "onebot"

    def __init__(
        self,
        process: ProcessHandler,
        enabled: bool,
        ws_url: str,
        access_token: str = "",
        bot_prefix: str = "",
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
    ):
        super().__init__(
            process,
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
        )
        self.enabled = enabled
        self.ws_url = ws_url
        self.access_token = access_token
        self.bot_prefix = bot_prefix

        self._ws_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ws = None

    @classmethod
    def from_env(
        cls, process: ProcessHandler, on_reply_sent: OnReplySent = None
    ) -> "OneBotChannel":
        import os

        return cls(
            process=process,
            enabled=os.getenv("ONEBOT_CHANNEL_ENABLED", "0") == "1",
            ws_url=os.getenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001"),
            access_token=os.getenv("ONEBOT_ACCESS_TOKEN", ""),
            bot_prefix=os.getenv("ONEBOT_BOT_PREFIX", ""),
            on_reply_sent=on_reply_sent,
        )

    @classmethod
    def from_config(
        cls,
        process: ProcessHandler,
        config: Any,
        on_reply_sent: OnReplySent = None,
        show_tool_details: bool = True,
    ) -> "OneBotChannel":
        return cls(
            process=process,
            enabled=getattr(config, "enabled", False),
            ws_url=getattr(config, "ws_url", "ws://127.0.0.1:3001"),
            access_token=getattr(config, "access_token", ""),
            bot_prefix=getattr(config, "bot_prefix", ""),
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
        )

    async def start(self) -> None:
        """Start the WebSocket client thread."""
        if not self.enabled:
            return
        if self._ws_thread is None or not self._ws_thread.is_alive():
            self._stop_event.clear()
            self._ws_thread = threading.Thread(
                target=self._run_ws_forever,
                name="onebot_ws",
                daemon=True,
            )
            self._ws_thread.start()

    async def stop(self) -> None:
        """Stop the WebSocket client and thread."""
        if not self.enabled:
            return
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._ws_thread:
            self._ws_thread.join(timeout=2)
            self._ws_thread = None

    def _run_ws_forever(self) -> None:
        try:
            import websocket
        except ImportError:
            logger.error(
                "websocket-client not installed. Please run: pip install websocket-client"
            )
            return

        retry_delay = 1

        while not self._stop_event.is_set():
            logger.info("OneBot connecting to %s", self.ws_url)
            try:
                headers = {}
                if self.access_token:
                    headers["Authorization"] = f"Bearer {self.access_token}"
                self._ws = websocket.create_connection(self.ws_url, header=headers)
                retry_delay = 1
                logger.info("OneBot connected.")

                while not self._stop_event.is_set():
                    raw = self._ws.recv()
                    if not raw:
                        break

                    payload = json.loads(raw)
                    self._handle_payload(payload)

            except Exception as e:
                logger.warning("OneBot ws error: %s", e)

            if self._stop_event.is_set():
                break

            logger.info("OneBot ws disconnected, retry in %s seconds...", retry_delay)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

    def _handle_payload(self, payload: dict) -> None:
        """Handle incoming OneBot event payload."""
        post_type = payload.get("post_type")
        if post_type != "message":
            return

        message_type = payload.get("message_type")  # "private" or "group"
        message_id = str(payload.get("message_id", ""))
        sender = payload.get("sender", {})
        user_id = str(sender.get("user_id", ""))
        group_id = str(payload.get("group_id", ""))

        text = ""
        message = payload.get("message", [])
        if isinstance(message, str):
            text = message
        elif isinstance(message, list):
            for seg in message:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    data = seg.get("data", {})
                    text += data.get("text", "")

        text = text.strip()
        if not text:
            # We skip empty text (can be extended for image parsing later)
            return

        if self.bot_prefix and text.startswith(self.bot_prefix):
            return

        meta = {
            "message_type": message_type,
            "message_id": message_id,
            "sender_id": user_id,
            "group_id": group_id,
            "incoming_raw": payload,
        }

        # Determine caller (sender context)
        caller_id = user_id if message_type == "private" else group_id
        session_id = group_id if message_type == "group" else user_id

        request = self.build_agent_request_from_user_content(
            channel_id="onebot",
            sender_id=caller_id,
            session_id=session_id,
            content_parts=[TextContent(type=ContentType.TEXT, text=text)],
            channel_meta=meta,
        )

        if self._enqueue is not None:
            self._enqueue(request)
            logger.info("OneBot enqueued request from %s: %r", caller_id, text[:50])

    async def consume_one(self, payload: Any) -> None:
        """Process one AgentRequest from manager queue, similar to other channels."""
        request = payload
        
        # Debounce mechanism (similar to QQ)
        if getattr(request, "input", None):
            session_id = getattr(request, "session_id", "") or ""
            contents = list(getattr(request.input[0], "content", None) or [])
            should_process, merged = self._apply_no_text_debounce(session_id, contents)
            if not should_process:
                return
            if merged:
                if hasattr(request.input[0], "model_copy"):
                    request.input[0] = request.input[0].model_copy(
                        update={"content": merged},
                    )
                else:
                    request.input[0].content = merged

        try:
            send_meta = getattr(request, "channel_meta", None) or {}
            send_meta.setdefault("bot_prefix", self.bot_prefix)
            to_handle = request.user_id or ""

            accumulated_parts = []

            async for event in self._process(request):
                obj = getattr(event, "object", None)
                status = getattr(event, "status", None)
                if obj == "message" and status == RunStatus.Completed:
                    parts = self._message_to_content_parts(event)
                    accumulated_parts.extend(parts)

            if accumulated_parts:
                await self.send_content_parts(
                    to_handle,
                    accumulated_parts,
                    send_meta,
                )

            if self._on_reply_sent:
                self._on_reply_sent(
                    self.channel,
                    to_handle,
                    request.session_id or f"{self.channel}:{to_handle}",
                )
        except Exception:
            logger.exception("OneBot process failed")
            try:
                fallback_handle = getattr(request, "user_id", "")
                await self.send_content_parts(
                    fallback_handle,
                    [{"type": "text", "text": "An error occurred while processing your request."}],
                    getattr(request, "channel_meta", None) or {},
                )
            except Exception:
                pass


    async def send(
        self, to_handle: str, text: str, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send message via the connected websocket."""
        if not self.enabled or not text.strip() or not self._ws:
            return

        meta = meta or {}
        message_type = meta.get("message_type", "private")

        params = {"message": text.strip()}

        if message_type == "group":
            # If group message, target the group
            params["group_id"] = int(meta.get("group_id", to_handle))
        else:
            # If private message, target the sender
            params["user_id"] = int(meta.get("sender_id", to_handle))

        api_req = {"action": "send_msg", "params": params, "echo": "copaw_reply"}

        try:
            self._ws.send(json.dumps(api_req))
        except Exception:
            logger.exception("OneBot send failed")
