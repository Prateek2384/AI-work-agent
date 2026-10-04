"""Start both simulated apps. `python -m env.run` to run them standalone (ports 8001 / 8002)."""
from __future__ import annotations

import threading
import time

import httpx
import uvicorn

from env.erp import app as erp_app
from env.vendor_portal import app as portal_app

PORTAL_URL, ERP_URL = "http://127.0.0.1:8001", "http://127.0.0.1:8002"


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def start_all(timeout: float = 10.0) -> list[uvicorn.Server]:
    servers = [_serve(portal_app, 8001), _serve(erp_app, 8002)]
    deadline = time.time() + timeout
    for url in (PORTAL_URL + "/login", ERP_URL + "/bills"):
        while True:
            try:
                httpx.get(url, timeout=1)
                break
            except httpx.HTTPError:
                if time.time() > deadline:
                    raise RuntimeError(f"{url} did not start")
                time.sleep(0.1)
    return servers


def stop_all(servers) -> None:
    for s in servers:
        s.should_exit = True


if __name__ == "__main__":
    start_all()
    print(f"Vendor portal: {PORTAL_URL}  (login ap-team / portal123)\nERP: {ERP_URL}\nCtrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
