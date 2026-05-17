"""Entry point for running the MCP server."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webnovel_kb.server import run

if __name__ == '__main__':
    run()
