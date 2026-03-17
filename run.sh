#!/bin/bash
# Run the Wakean Word Forge MCP server
# Requires PYTHONUTF8=1 on Windows to handle panphon's IPA data files

export PYTHONUTF8=1

cd "$(dirname "$0")"

# Default: stdio transport (for Claude Code MCP integration)
# Use --transport sse for HTTP server mode
fastmcp run server.py "$@"
