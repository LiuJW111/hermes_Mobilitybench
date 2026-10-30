"""Serve unchanged sandbox handlers once for concurrent isolated Hermes clients."""
import argparse
import os
from pathlib import Path

from starlette.responses import JSONResponse

import sandbox_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--sandbox-dir', type=Path, required=True)
    args = parser.parse_args()
    os.environ['MOBILITY_BENCH_SANDBOX_DIR'] = str(args.sandbox_dir.resolve())
    server = sandbox_tools.mcp
    server.settings.host = '127.0.0.1'
    server.settings.port = args.port
    server.settings.stateless_http = True
    server.settings.json_response = True
    server.settings.log_level = 'WARNING'
    print('Loading shared read-only sandbox data once...', flush=True)
    for name in sandbox_tools.TOOL_NAMES:
        if sandbox_tools._load_impl(name) is None:
            raise RuntimeError(f'Sandbox tool failed to load: {name}')

    @server.custom_route('/health', methods=['GET'])
    async def health(request):
        return JSONResponse({'ready': True, 'pid': os.getpid(), 'tools': sandbox_tools.TOOL_NAMES})

    print(f'Shared sandbox ready at http://127.0.0.1:{args.port}/mcp', flush=True)
    server.run(transport='streamable-http')


if __name__ == '__main__':
    main()
