import asyncio

from openpi_client import msgpack_numpy
from openpi.serving import websocket_policy_server as _websocket_policy_server
import websockets
import websockets.frames


class _DummyPolicy:
    def __init__(self):
        self.reset_calls = 0
        self.infer_inputs = []

    def reset(self):
        self.reset_calls += 1

    def infer(self, obs):
        self.infer_inputs.append(obs)
        return {"actions": [1, 2, 3]}


class _FakeWebsocket:
    def __init__(self, received):
        self._received = list(received)
        self.sent = []
        self.remote_address = ("127.0.0.1", 50000)

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if self._received:
            return self._received.pop(0)
        close = websockets.frames.Close(websockets.frames.CloseCode.NORMAL_CLOSURE, "")
        raise websockets.ConnectionClosedOK(close, close, True)

    async def close(self, code=None, reason=None):
        del code, reason


def test_handler_processes_reset_without_reconnect():
    packer = msgpack_numpy.Packer()
    policy = _DummyPolicy()
    server = _websocket_policy_server.WebsocketPolicyServer(policy, metadata={"server": "ok"})
    websocket = _FakeWebsocket(
        [
            packer.pack(
                {_websocket_policy_server._CONTROL_MESSAGE_KEY: _websocket_policy_server._RESET_COMMAND}
            ),
            packer.pack({"observation/state": [1.0, 2.0, 3.0]}),
        ]
    )

    asyncio.run(server._handler(websocket))

    assert policy.infer_inputs == [{"observation/state": [1.0, 2.0, 3.0]}]
    assert policy.reset_calls == 3
    assert msgpack_numpy.unpackb(websocket.sent[0]) == {"server": "ok"}
    assert msgpack_numpy.unpackb(websocket.sent[1]) == {
        _websocket_policy_server._CONTROL_MESSAGE_KEY: _websocket_policy_server._RESET_ACK
    }
    assert msgpack_numpy.unpackb(websocket.sent[2])["actions"] == [1, 2, 3]
