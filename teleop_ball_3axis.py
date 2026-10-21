import time
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

LEADER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79016111-if00"
FOLLOWER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79014502-if00"

MOVING = (2, 3, 4)
FIXED = (1, 5, 6)
ALL = (1, 2, 3, 4, 5, 6)

LIMITS = {
    1: (845, 3246),
    2: (912, 3185),
    3: (951, 3143),
    4: (962, 3208),
    5: (0, 4095),
    6: (1318, 2754),
}

DURATION = 30
PERIOD = 0.02

leader = PortHandler(LEADER)
follower = PortHandler(FOLLOWER)
sdk = PacketHandler(0)

def read_pos(port, mid):
    for _ in range(3):
        value, result, error = sdk.read2ByteTxRx(port, mid, 56)
        if result == COMM_SUCCESS and error == 0:
            return value
        time.sleep(0.003)
    raise RuntimeError(f"ID {mid}: position read failed")

def write1(mid, address, value):
    result, error = sdk.write1ByteTxRx(follower, mid, address, value)
    if result != COMM_SUCCESS or error:
        raise RuntimeError(f"ID {mid}: write failed at {address}")

def write_goal(mid, value):
    result, error = sdk.write2ByteTxRx(follower, mid, 42, value)
    if result != COMM_SUCCESS or error:
        raise RuntimeError(f"ID {mid}: goal write failed")

enabled = []

try:
    for port in (leader, follower):
        if not port.openPort() or not port.setBaudRate(1000000):
            raise RuntimeError("Serial port connection failed")

    # 開始時のLeader/Follower位置を記録
    leader_start = {i: read_pos(leader, i) for i in MOVING}
    follower_start = {i: read_pos(follower, i) for i in ALL}

    # 初期位置が校正範囲外でも、起動は妨げない
    for i in ALL:
        low, high = LIMITS[i]
        if not low <= follower_start[i] <= high:
            print(
                f"WARNING: ID {i} initial position "
                f"{follower_start[i]} outside [{low}, {high}]"
            )

    # Followerを全軸いったん現在位置保持に設定
    for i in ALL:
        write1(i, 40, 0)
        write_goal(i, follower_start[i])

    # 固定軸ID 1,5,6も保持トルクを有効化
    for i in ALL:
        write1(i, 40, 1)
        enabled.append(i)

    print("Ball-in-Cup 3-axis Teleoperation")
    print("MOVING: ID 2, 3, 4")
    print("FIXED : ID 1, 5, 6")
    print("Duration: 30 seconds")
    print("Press Ctrl+C to stop")
    print("Follower initial positions:", follower_start)

    end_time = time.monotonic() + DURATION

    while time.monotonic() < end_time:
        start = time.monotonic()

        for i in MOVING:
            leader_now = read_pos(leader, i)

            # Leaderの開始位置からの移動量だけFollowerを動かす
            delta = leader_now - leader_start[i]
            target = follower_start[i] + delta

            low, high = LIMITS[i]
            target = max(low, min(high, target))

            write_goal(i, target)

        remaining = PERIOD - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)

    print("TEST COMPLETED")

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    print("Disabling follower torque...")
    for i in ALL:
        try:
            write1(i, 40, 0)
        except Exception as exc:
            print(f"ID {i}: torque OFF failed: {exc}")

    leader.closePort()
    follower.closePort()
