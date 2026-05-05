"""Minimal no-LLM miner example.

This example wraps :func:`eirel.sample_server.create_app`, which returns a
:class:`eirel.MinerApp` backed by a hardcoded stub handler.  It serves two
purposes:

1. **Reference submission layout.**  See ``submission.yaml`` next to this
   file for the manifest miners ship to the owner-api — agent name / version,
   ``family_id``, and the ``inference`` block that declares the miner's
   preferred provider and model.
2. **Smoke test for the runtime contract.**  Responses are deterministic
   (`eirel compliance` passes against it), so operators can use it to verify
   that a new k8s runtime image boots and speaks the chat/completions
   protocol without needing provider credentials.

This example does NOT demonstrate LLM provider usage.  For a realistic miner
that calls an LLM through the subnet provider-proxy, see
``examples/graph_general_chat/app.py``.
"""
from __future__ import annotations

from eirel.sample_server import create_app

app = create_app()
