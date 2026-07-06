import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

_CONTROL_MESSAGE_KEY = "__openpi_cmd__"
_RESET_COMMAND = "reset"
_RESET_ACK = "reset_ack"


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _assert_control_ack(self, response: object, *, expected_command: str) -> None:
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        payload = msgpack_numpy.unpackb(response)
        if not isinstance(payload, dict) or payload.get(_CONTROL_MESSAGE_KEY) != expected_command:
            raise RuntimeError(f"Unexpected control response from inference server: {payload!r}")

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        payload = dict(obs)
        if "timestamp" not in payload:
            observation = payload.get("observation")
            if not isinstance(observation, dict) or "timestamp" not in observation:
                payload["timestamp"] = time.monotonic()
        data = self._packer.pack(payload)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        self._ws.send(self._packer.pack({_CONTROL_MESSAGE_KEY: _RESET_COMMAND}))
        self._assert_control_ack(self._ws.recv(), expected_command=_RESET_ACK)
