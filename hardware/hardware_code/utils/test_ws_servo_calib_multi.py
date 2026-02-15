import asyncio
import websockets

HOST = "0.0.0.0"
PORT = 8765

# device_id -> websocket
devices = {}

async def handler(ws):
    print("Client connected. Waiting for ID...")

    device_id = None

    try:
        async for msg in ws:
            msg = msg.strip()

            # FIRST MESSAGE MUST BE ID
            if msg.startswith("ID:"):
                device_id = msg[3:]
                devices[device_id] = ws
                print(f"Registered device: {device_id}")
                await ws.send("REGISTERED")
                continue

            print(f"From {device_id}: {msg}")

    except websockets.ConnectionClosed:
        pass
    finally:
        if device_id and device_id in devices:
            del devices[device_id]
            print(f"{device_id} disconnected")


async def user_input_loop():
    while True:
        cmd = await asyncio.to_thread(
            input,
            "\nCommand format: DEVICE_ID ANGLE (ex: ESP1 120)\n> "
        )

        parts = cmd.strip().split()

        if len(parts) != 2:
            print("Invalid format")
            continue

        device_id, angle = parts

        try:
            angle = int(angle)
        except:
            print("Angle must be number")
            continue

        if device_id not in devices:
            print("Device not connected")
            continue

        try:
            await devices[device_id].send(str(angle))
            print(f"Sent {angle} to {device_id}")
        except:
            print("Send failed")


async def main():
    print(f"Server running at ws://{HOST}:{PORT}")

    async with websockets.serve(handler, HOST, PORT):
        await user_input_loop()


if __name__ == "__main__":
    asyncio.run(main())
