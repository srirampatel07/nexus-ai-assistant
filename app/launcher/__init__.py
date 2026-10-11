"""One-click local AI launcher for NEXUS (Windows).

`servers.py` owns llama-server lifecycle (discovery, ownership-gated
start/stop, readiness polling); `menu.py` drives the interactive menu on
top of the existing `run.py` text interface. Nothing here replaces the
assistant — it only ensures a model server is up before delegating.
"""
