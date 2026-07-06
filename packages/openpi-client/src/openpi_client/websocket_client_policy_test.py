from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _websocket_client_policy


class _FakeConnection:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        return self._responses.pop(0)

    def close(self):
        self.closed = True


def test_reset_sends_control_message_without_reconnect():
    packer = msgpack_numpy.Packer()
    policy = object.__new__(_websocket_client_policy.WebsocketClientPolicy)
    policy._packer = packer
    policy._server_metadata = {"temporal_memory_enabled": True}
    policy._ws = _FakeConnection(
        [packer.pack({_websocket_client_policy._CONTROL_MESSAGE_KEY: _websocket_client_policy._RESET_ACK})]
    )
    policy._wait_for_server = lambda: (_ for _ in ()).throw(AssertionError("reset should not reconnect"))

    policy.reset()

    assert not policy._ws.closed
    assert len(policy._ws.sent) == 1
    assert msgpack_numpy.unpackb(policy._ws.sent[0]) == {
        _websocket_client_policy._CONTROL_MESSAGE_KEY: _websocket_client_policy._RESET_COMMAND
    }
