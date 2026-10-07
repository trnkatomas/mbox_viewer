"""
Minimal agent that wires an Ollama LLM to the mbox-viewer MCP server.

The agent:
  1. Starts mcp_server.py as a subprocess (stdio transport).
  2. Fetches the list of available tools and converts them to Ollama format.
  3. Runs a tool-calling loop until the model stops issuing tool calls.

Requirements (in addition to the project's own deps):
    pip install ollama          # or: uv add ollama

Usage:
    MBOX_FILE_PATH=/path/to/your.mbox python ollama_agent.py "who sent me the most emails?"
    MBOX_FILE_PATH=/path/to/your.mbox python ollama_agent.py --model llama3.1:8b "find emails about invoices"

Tested with models that support tool use: qwen2.5:7b, llama3.1:8b, mistral-nemo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    model: str, initial_query: str | None, verbose: bool = False, max_steps: int = 10
) -> None:
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    server_params = StdioServerParameters(
        command="uv",
        args=["run", server_script],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- discover tools -------------------------------------------
            tools_result = await session.list_tools()
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in tools_result.tools
            ]

            if verbose:
                names = [t.name for t in tools_result.tools]
                print(f"[agent] {len(names)} tools available: {', '.join(names)}\n", flush=True)

            client = ollama.AsyncClient()

            # --- REPL: one MCP session, many queries ----------------------
            first_query = initial_query
            while True:
                if first_query:
                    query = first_query
                    first_query = None
                else:
                    try:
                        query = input("\nQuery (Ctrl-D to quit): ").strip()
                    except EOFError:
                        break
                    if not query:
                        continue

                # Fresh message history per query (stateless between turns)
                messages: list[dict] = [{"role": "user", "content": query}]

                # --- agentic tool-calling loop ----------------------------
                for _ in range(max_steps):
                    response = await client.chat(
                        model=model,
                        messages=messages,
                        tools=ollama_tools,
                    )
                    msg = response.message

                    assistant_entry: dict = {
                        "role": "assistant",
                        "content": msg.content or "",
                    }
                    if msg.tool_calls:
                        assistant_entry["tool_calls"] = msg.tool_calls
                    messages.append(assistant_entry)

                    if not msg.tool_calls:
                        print(msg.content)
                        break

                    for tc in msg.tool_calls:
                        fn = tc.function
                        args = fn.arguments or {}
                        if verbose:
                            print(f"[tool] {fn.name}({json.dumps(args, ensure_ascii=False)})", flush=True)

                        result = await session.call_tool(fn.name, dict(args))
                        first = result.content[0] if result.content else None
                        tool_text = first.text if isinstance(first, TextContent) else ""

                        if verbose:
                            preview = tool_text[:200].replace("\n", " ")
                            print(f"       → {preview}{'…' if len(tool_text) > 200 else ''}\n", flush=True)

                        messages.append({"role": "tool", "tool_name": fn.name, "content": tool_text})
                else:
                    print(
                        f"[agent] stopped after {max_steps} tool-calling rounds without a final answer; "
                        "try a more specific question or raise --max-steps.",
                        file=sys.stderr,
                    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with your mbox archive using Ollama + MCP tools."
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b",
        help="Ollama model name (must support tool use). Default: gemma4:e2b",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print tool calls and response previews.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum tool-calling rounds per query. Default: 10",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural-language query. Prompted interactively if omitted.",
    )
    args = parser.parse_args()

    if not os.environ.get("MBOX_FILE_PATH"):
        print("Error: MBOX_FILE_PATH environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_agent(args.model, args.query or None, verbose=args.verbose, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
