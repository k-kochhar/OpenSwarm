import asyncio
import websockets
import os
import time
from typing import Dict, Optional, List

HOST = "0.0.0.0"
PORT = 8765

# device_id -> websocket
devices: Dict[str, websockets.WebSocketServerProtocol] = {}

# Keyboard -> command character
KEYMAP = {
    "w": "F",
    "s": "B",
    "a": "L",
    "d": "R",
    " ": "S",   # space = stop immediately
}

AUTO_STOP_AFTER_SEC = 0.25  # if no movement key press within this time -> send S


# ---------- Cross-platform key reader (no extra libraries) ----------
def _getch_blocking() -> str:
    """Return one keypress as a string (best-effort for Win/Linux/macOS)."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getch()
        # Handle special keys (arrows etc.) that come as two bytes
        if ch in (b"\x00", b"\xe0"):
            _ = msvcrt.getch()
            return ""  # ignore special keys
        try:
            return ch.decode("utf-8", errors="ignore")
        except:
            return ""
    else:
        import sys
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def keypress_loop(queue: asyncio.Queue):
    """Runs getch in a thread so asyncio doesn't block."""
    while True:
        ch = await asyncio.to_thread(_getch_blocking)
        if ch:
            await queue.put(ch)


# ---------- WebSocket handler ----------
async def handler(ws):
    print("Client connected. Waiting for ID...")

    device_id = None
    try:
        async for msg in ws:
            msg = msg.strip()

            # First: device registers itself
            if msg.startswith("ID:"):
                device_id = msg[3:].strip()
                if not device_id:
                    await ws.send("ERR:EMPTY_ID")
                    continue

                # If same ID reconnects, replace old
                devices[device_id] = ws
                print(f"Registered: {device_id}  (total={len(devices)})")

                # Acknowledge registration
                await ws.send("REGISTERED")

                # ✅ Immediately send STOP on connect/register
                await ws.send("S")
                print(f"Sent initial STOP -> {device_id}")

                continue

            # Optional: receive logs from ESP32
            if device_id:
                print(f"From {device_id}: {msg}")
            else:
                print("From unknown client:", msg)

    except websockets.ConnectionClosed:
        pass
    finally:
        # Remove only if this websocket is still the active one for that ID
        if device_id and devices.get(device_id) is ws:
            devices.pop(device_id, None)
            print(f"{device_id} disconnected  (total={len(devices)})")
        else:
            print("Client disconnected")


def _sorted_device_ids() -> List[str]:
    return sorted(devices.keys())


async def send_to_target(target: Optional[str], payload: str):
    """
    target:
      - None  => broadcast to all
      - "ESP1" => send only to that device (if connected)
    """
    if target is None:
        if not devices:
            return
        dead = []
        for did, ws in list(devices.items()):
            try:
                await ws.send(payload)
            except:
                dead.append(did)
        for did in dead:
            if devices.get(did) is not None:
                devices.pop(did, None)
        return

    ws = devices.get(target)
    if not ws:
        return
    try:
        await ws.send(payload)
    except:
        # drop if broken
        if devices.get(target) is ws:
            devices.pop(target, None)


async def keyboard_control_loop():
    print("\nControls: w/a/s/d = move, SPACE = stop, [ ] = change robot, 0 = broadcast, q = quit")
    print("Waiting for robots to connect...\n")

    keyq: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(keypress_loop(keyq))

    target: Optional[str] = None  # None => broadcast
    last_sent_cmd = "S"
    last_move_time = time.time()

    while True:
        # Keep target valid as devices connect/disconnect
        if target is not None and target not in devices:
            target = None  # fallback to broadcast if chosen target vanished

        # Non-blocking wait with small timeout so we can auto-stop
        try:
            ch = await asyncio.wait_for(keyq.get(), timeout=0.05)
        except asyncio.TimeoutError:
            ch = ""

        if ch:
            # Quit
            if ch.lower() == "q":
                await send_to_target(target, "S")
                print("\nQuit. Sent STOP.")
                return

            # Broadcast toggle
            if ch == "0":
                target = None
                print("\nTarget: BROADCAST (all robots)")
                continue

            # Cycle targets with [ and ]
            if ch in ("[", "]"):
                ids = _sorted_device_ids()
                if not ids:
                    target = None
                    print("\nNo robots connected.")
                    continue

                if target is None:
                    target = ids[0] if ch == "]" else ids[-1]
                else:
                    i = ids.index(target)
                    target = ids[(i + 1) % len(ids)] if ch == "]" else ids[(i - 1) % len(ids)]
                print(f"\nTarget: {target}")
                continue

            # Movement keys
            cmd = KEYMAP.get(ch.lower())
            if cmd:
                await send_to_target(target, cmd)
                last_sent_cmd = cmd
                if cmd != "S":
                    last_move_time = time.time()

                tgt_label = "BROADCAST" if target is None else target
                print(f"\rSent {cmd} -> {tgt_label}    ", end="", flush=True)

        # Auto-stop if no movement key pressed recently
        now = time.time()
        if last_sent_cmd in ("F", "B", "L", "R") and (now - last_move_time) > AUTO_STOP_AFTER_SEC:
            await send_to_target(target, "S")
            last_sent_cmd = "S"
            tgt_label = "BROADCAST" if target is None else target
            print(f"\rAuto STOP -> {tgt_label}    ", end="", flush=True)


async def main():
    print(f"WebSocket server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT, ping_interval=20, ping_timeout=20):
        await keyboard_control_loop()


if __name__ == "__main__":
    asyncio.run(main())
