"""The MCP boundary: interoperability, never a shortcut.

Two directions, one rule: a tool call that crosses this boundary takes the same
path as a native one (ToolRegistry.execute): allow-list, exam mode, permissions,
schema checks, idempotency, deadline, cancellation, output checks and an audit
record. Native execution stays the default; nothing here is registered unless an
integrator asks.

**Exporting** (``McpToolServer``): a registered tool is offered as an MCP tool.
The caller's identity is fixed when the server is built, from whatever
authenticated the transport; no request parameter can set or change it.
``tools/list`` shows only what that caller may use; ``tools/call`` runs
``registry.execute``.

**Consuming** (``register_mcp_tools``): an external MCP server's tool becomes a
registered tool only through an explicit ``McpToolPolicy`` written by the
integrator, who supplies what the remote cannot be trusted to: the local name
(always ``mcp_`` prefixed, so it can never shadow a native tool), the description
the model sees, the permissions, and the side-effect class (read or propose only:
an external server never writes through the backend). The remote's own
description is ignored. Its input schema must be one this code can enforce, and
is tightened to ``additionalProperties: false``. Its output is untrusted data:
returned marked ``untrusted``, size-checked, and never evidence, so it cannot be
cited as a source.

The wire shape is a JSON-RPC subset of MCP (``tools/list``, ``tools/call``). There
is no network here: a transport is a Protocol, and the in-process
``LoopbackTransport`` connects a client to a server. A real transport (stdio,
HTTP) is a later, reviewed addition.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.agentic.contracts import (
    CAPABILITY,
    PERMISSION,
    TOOL_NAME,
    Boundary,
    CallerContext,
    Idempotency,
    RetryClass,
    RetryPolicy,
    SideEffect,
    ToolContract,
    canonical_json,
)
from app.core.agentic.registry import RunControl, ToolCall, ToolFailure, ToolRegistry
from app.core.agentic.schema import SchemaError, check_schema

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
_FIXED = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
}
MAX_TEXT = 4_000
LOCAL_PREFIX = "mcp_"


class McpTransportError(Exception):
    """The transport failed. Carries no text."""


class McpTransport(Protocol):
    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None: ...


def _error(request_id: Any, code: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": _FIXED[code]}}


# --- exporting a registered tool ---------------------------------------------------


class McpToolServer:
    """Serves a registry's tools to an MCP client, for one fixed caller."""

    def __init__(
        self,
        registry: ToolRegistry,
        context: CallerContext,
        *,
        allowed: tuple[str, ...] | None = None,
        execution_id: str = "mcp",
        control: RunControl | None = None,
    ) -> None:
        self._registry = registry
        self._context = context
        self._allowed = allowed
        self._execution_id = execution_id
        self._control = control
        self._calls = 0

    async def handle(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST)
        request_id = message.get("id")
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _error(request_id, INVALID_REQUEST)
        if request_id is not None and (
            isinstance(request_id, bool) or not isinstance(request_id, (int, str))
        ):
            return _error(None, INVALID_REQUEST)
        params = message.get("params", {})
        if not isinstance(params, dict):
            return _error(request_id, INVALID_PARAMS)
        method = message["method"]
        if method == "tools/list":
            if params:
                return _error(request_id, INVALID_PARAMS)
            return self._ok(request_id, {"tools": self._listing()})
        if method == "tools/call":
            return await self._call(request_id, params)
        if request_id is None:
            return None  # an unknown notification is ignored
        return _error(request_id, METHOD_NOT_FOUND)

    @staticmethod
    def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any] | None:
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _listing(self) -> list[dict[str, Any]]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "inputSchema": c.input_schema,
            }
            for c in self._registry.available(self._context, self._allowed)
        ]

    async def _call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any] | None:
        # Only these parameters exist. Identity is never one of them.
        if not set(params) <= {"name", "arguments", "_meta"}:
            return _error(request_id, INVALID_PARAMS)
        name, arguments = params.get("name"), params.get("arguments", {})
        meta = params.get("_meta", {})
        if (
            not isinstance(name, str)
            or not isinstance(arguments, dict)
            or not isinstance(meta, dict)
            or self._registry.contract(name) is None
            or (self._allowed is not None and name not in self._allowed)
        ):
            return _error(request_id, INVALID_PARAMS)
        key = meta.get("idempotencyKey")
        if key is not None and (not isinstance(key, str) or not 0 < len(key) <= 128):
            return _error(request_id, INVALID_PARAMS)
        self._calls += 1
        outcome = await self._registry.execute(
            name,
            arguments,
            self._context,
            call_id=f"{self._execution_id}-c{self._calls}",
            execution_id=self._execution_id,
            control=self._control,
            allowed=self._allowed,
            idempotency_key=key,
        )
        if outcome.ok:
            text = canonical_json(outcome.data)
            result: dict[str, Any] = {
                "content": [{"type": "text", "text": text}],
                "structuredContent": outcome.data,
                "isError": False,
            }
        else:
            failure = outcome.failure()
            result = {
                "content": [{"type": "text", "text": canonical_json(failure)}],
                "structuredContent": failure,
                "isError": True,
            }
        return self._ok(request_id, result)


# --- consuming an external tool ---------------------------------------------------------


class McpClient:
    """A minimal client: lists and calls tools over a transport, checking every
    response's shape. Nothing a remote says is trusted beyond that shape."""

    def __init__(self, transport: McpTransport) -> None:
        self._transport = transport
        self._next = 0

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next += 1
        request_id = self._next
        try:
            reply = await self._transport.send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
        except McpTransportError:
            raise
        except Exception:
            raise McpTransportError() from None
        if (
            not isinstance(reply, dict)
            or reply.get("jsonrpc") != "2.0"
            or reply.get("id") != request_id
            or not isinstance(reply.get("result"), dict)
        ):
            raise McpTransportError()
        return reply["result"]  # type: ignore[no-any-return]

    async def list_tools(self) -> list[dict[str, Any]]:
        tools = (await self._request("tools/list", {})).get("tools")
        if not isinstance(tools, list):
            raise McpTransportError()
        return [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], idempotency_key: str | None = None
    ) -> Any:
        """The tool's result data. ToolFailure when the remote reports an error
        (its text is discarded) or answers in a shape that is not a result."""
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        if idempotency_key:
            params["_meta"] = {"idempotencyKey": idempotency_key}
        try:
            result = await self._request("tools/call", params)
        except McpTransportError:
            raise ToolFailure("TOOL_UNAVAILABLE") from None
        if result.get("isError") is not False:
            raise ToolFailure("TOOL_FAILED")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except ValueError:
                    return {"text": text[:MAX_TEXT]}
        raise ToolFailure("TOOL_FAILED")


class LoopbackTransport:
    """In-process: hands each message to a server. For tests and local wiring."""

    def __init__(self, server: McpToolServer) -> None:
        self._server = server

    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        return await self._server.handle(json.loads(json.dumps(message)))


class McpToolPolicy(BaseModel):
    """What an integrator decides about one external tool. The remote decides none of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_name: str = Field(min_length=1, max_length=128)
    local_name: str
    description: str = Field(min_length=10, max_length=600)
    side_effect: SideEffect = SideEffect.READ
    required_permissions: tuple[str, ...] = Field(min_length=1)
    required_capabilities: tuple[str, ...] = ()
    deadline_seconds: float = Field(default=15.0, gt=0, le=60)

    @field_validator("local_name")
    @classmethod
    def _local(cls, value: str) -> str:
        if not TOOL_NAME.match(value) or not value.startswith(LOCAL_PREFIX):
            raise ValueError("an external tool's local name starts with mcp_")
        return value

    @field_validator("side_effect")
    @classmethod
    def _effect(cls, value: SideEffect) -> SideEffect:
        if value is SideEffect.WRITE_THROUGH_BACKEND:
            raise ValueError("an external tool is read or propose only")
        return value

    @field_validator("required_permissions")
    @classmethod
    def _permissions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not PERMISSION.match(p) for p in value):
            raise ValueError("permissions are 'area:action' names")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def _capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not CAPABILITY.match(c) for c in value):
            raise ValueError("capabilities are names")
        return value


class McpRegistration(BaseModel):
    registered: tuple[str, ...]
    # Policies not applied, with a fixed reason: missing, schema, collision.
    skipped: tuple[tuple[str, str], ...]


def _tighten(schema: Any) -> dict[str, Any]:
    check_schema(schema)
    if schema.get("type") != "object":
        raise SchemaError("not an object schema")
    return {**schema, "additionalProperties": False}


async def register_mcp_tools(
    registry: ToolRegistry, client: McpClient, policies: list[McpToolPolicy]
) -> McpRegistration:
    """Register the external tools the policies name, and only those."""
    remote = {t["name"]: t for t in await client.list_tools()}
    registered: list[str] = []
    skipped: list[tuple[str, str]] = []
    for policy in policies:
        tool = remote.get(policy.remote_name)
        if tool is None:
            skipped.append((policy.local_name, "missing"))
            continue
        if registry.contract(policy.local_name) is not None:
            skipped.append((policy.local_name, "collision"))
            continue
        try:
            input_schema = _tighten(tool.get("inputSchema"))
            contract = _contract(policy, input_schema)
        except (SchemaError, ValidationError, ValueError):
            skipped.append((policy.local_name, "schema"))
            continue
        registry.register(contract, _forward(client, policy.remote_name))
        registered.append(policy.local_name)
    return McpRegistration(registered=tuple(registered), skipped=tuple(skipped))


def _contract(policy: McpToolPolicy, input_schema: dict[str, Any]) -> ToolContract:
    propose = policy.side_effect is SideEffect.PROPOSE
    return ToolContract(
        name=policy.local_name,
        version="1.0.0",
        description=policy.description,
        input_schema=input_schema,
        output_schema={
            "type": "object",
            "properties": {"result": {}, "untrusted": {"const": True}},
            "required": ["result", "untrusted"],
            "additionalProperties": False,
        },
        required_permissions=policy.required_permissions,
        required_capabilities=policy.required_capabilities,
        side_effect=policy.side_effect,
        idempotency=Idempotency.KEYED if propose else Idempotency.NATURAL,
        retry=RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=1),
        deadline_seconds=policy.deadline_seconds,
        boundaries=(Boundary.MCP,),
        produces_evidence=False,
        max_output_bytes=32_768,
    )


def _forward(client: McpClient, remote_name: str) -> Any:
    async def handler(call: ToolCall) -> dict[str, Any]:
        result = await client.call_tool(remote_name, call.args, call.idempotency_key)
        return {"result": result, "untrusted": True}

    return handler
