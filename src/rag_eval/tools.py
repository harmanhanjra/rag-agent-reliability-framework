from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError


class AccountTierArgs(BaseModel):
    account_id: str = Field(pattern=r"^A-\d+$")


class DiscountArgs(BaseModel):
    amount: float = Field(gt=0)
    percent: float = Field(ge=0, le=100)


class DeleteWorkspaceArgs(BaseModel):
    workspace_id: str = Field(pattern=r"^W-\d+$")
    confirmed: bool = False


def get_account_tier(account_id: str) -> str:
    return {"A-100": "enterprise", "A-200": "starter"}.get(account_id, "unknown")


def calculate_discount(amount: float, percent: float) -> float:
    return round(amount * (1 - percent / 100), 2)


def delete_workspace(workspace_id: str, confirmed: bool = False) -> str:
    return f"deleted:{workspace_id}" if confirmed else "approval_required"


TOOLS = {
    "get_account_tier": StructuredTool.from_function(
        func=get_account_tier, name="get_account_tier",
        description="Look up an account tier.", args_schema=AccountTierArgs
    ),
    "calculate_discount": StructuredTool.from_function(
        func=calculate_discount, name="calculate_discount",
        description="Calculate a percentage discount.", args_schema=DiscountArgs
    ),
    "delete_workspace": StructuredTool.from_function(
        func=delete_workspace, name="delete_workspace",
        description="Delete a workspace.", args_schema=DeleteWorkspaceArgs
    ),
}


@dataclass
class ToolResult:
    tool: str | None
    args: dict[str, Any]
    status: str
    output: Any = None


def _plan(query: str) -> tuple[str | None, dict[str, Any]]:
    q = query.lower()
    if "account tier" in q:
        m = re.search(r"\bA-\d+\b", query, re.I)
        return "get_account_tier", {"account_id": m.group(0).upper()} if m else {}
    if "discount" in q:
        nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", query)]
        args = {"percent": nums[0], "amount": nums[1]} if len(nums) >= 2 else {}
        return "calculate_discount", args
    if "delete workspace" in q:
        m = re.search(r"\bW-\d+\b", query, re.I)
        return "delete_workspace", {"workspace_id": m.group(0).upper()} if m else {}
    return None, {}


class BaselineToolAgent:
    def run(self, query: str) -> ToolResult:
        name, args = _plan(query)
        if name is None:
            return ToolResult(None, {}, "refused")
        if name == "delete_workspace":
            args["confirmed"] = True
        try:
            output = TOOLS[name].invoke(args)
            return ToolResult(name, args, "ok", output)
        except Exception as exc:
            return ToolResult(name, args, "error", str(exc))


class HardenedToolAgent:
    def run(self, query: str) -> ToolResult:
        name, args = _plan(query)
        if name is None or name not in TOOLS:
            return ToolResult(name, args, "refused")
        if name == "delete_workspace":
            return ToolResult(name, args, "approval_required")
        schema = TOOLS[name].args_schema
        try:
            validated = schema.model_validate(args)
        except ValidationError as exc:
            return ToolResult(name, args, "invalid_args", str(exc))
        output = TOOLS[name].invoke(validated.model_dump())
        return ToolResult(name, validated.model_dump(), "ok", output)
