"""litellm-compat shim package.

arcllm has no callback / integrations subsystem. This package exists so
code that imports ``litellm.integrations.custom_logger.CustomLogger``
(typically to subclass it as an observability hook) keeps loading after
a ``litellm → arcllm`` import-path swap.

The stubs here are passive — arcllm never invokes them. Code that wants
per-call observability should read ``response.usage`` directly from the
return value of :func:`arcllm.completion` / :func:`arcllm.acompletion`
instead of relying on a callback fired by the SDK.
"""

from __future__ import annotations

from arcllm.integrations.custom_logger import CustomLogger

__all__ = ["CustomLogger"]
