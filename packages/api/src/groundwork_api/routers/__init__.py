"""One router module per resource: health, workspaces (upload, list, get,
delete), conversations (create a conversation, ask a question, list its
turns), and eval (read only retrieval and red team results). app.py
mounts every one of them; nothing here imports app.py, so the dependency
only ever runs one direction.
"""
