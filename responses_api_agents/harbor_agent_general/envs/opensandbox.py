import ssl
from typing import Any

from harbor.environments.opensandbox import OpenSandboxEnvironment as HarborOpenSandboxEnvironment


class OpenSandboxEnvironment(HarborOpenSandboxEnvironment):
    """
    Harbor environment backed by the OpenSandbox Python SDK.

    Allows turning off TLS verification, while preserving all other behavior.
    """
    def __init__(self, *args: Any, tls_verify: bool = True, **kwargs: Any) -> None:
        self._tls_verify = tls_verify
        super().__init__(*args, **kwargs)

    def _build_connection_config(self, sdk: dict[str, Any]) -> Any:
        config = super()._build_connection_config(sdk)
        if not self._tls_verify:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            config._ssl_context = context
        return config
