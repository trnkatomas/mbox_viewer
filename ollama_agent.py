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


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

async def run_agent(model: str, user_query: str, verbose: bool = False) -> None:
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

            # --- agentic loop --------------------------------------------
            messages: list[dict] = [{"role": "user", "content": user_query}]
            client = ollama.AsyncClient()

            while True:
                response = await client.chat(
                    model=model,
                    messages=messages,
                    tools=ollama_tools,
                )
                msg = response.message

                # Accumulate the assistant turn (Ollama may return tool_calls=None)
                assistant_entry: dict = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    assistant_entry["tool_calls"] = msg.tool_calls
                messages.append(assistant_entry)

                if not msg.tool_calls:
                    # Final answer
                    print(msg.content)
                    break

                # --- call each tool via MCP --------------------------------
                for tc in msg.tool_calls:
                    fn = tc.function
                    args = fn.arguments or {}
                    if verbose:
                        print(f"[tool] {fn.name}({json.dumps(args, ensure_ascii=False)})", flush=True)

                    result = await session.call_tool(fn.name, args)
                    tool_text = result.content[0].text if result.content else ""

                    if verbose:
                        preview = tool_text[:200].replace("\n", " ")
                        print(f"       → {preview}{'…' if len(tool_text) > 200 else ''}\n", flush=True)

                    messages.append({"role": "tool", "content": tool_text})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with your mbox archive using Ollama + MCP tools."
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama model name (must support tool use). Default: qwen2.5:7b",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print tool calls and response previews.",
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

    query = args.query or input("Query: ").strip()
    if not query:
        print("No query provided.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_agent(args.model, query, verbose=args.verbose))


if __name__ == "__main__":
    main()
