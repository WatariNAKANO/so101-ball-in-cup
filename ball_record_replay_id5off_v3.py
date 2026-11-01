#!/usr/bin/env python3
"""SO-101 3-axis leader recording and follower replay (raw servo counts)."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

LEADER = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79016111-if00'
FOLLOWER = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79014502-if00'
MOVING = (2, 3, 4)
ALL = (1, 2, 3, 4, 5, 6)
LIMITS = {1: (845, 3246), 2: (912, 3185), 3: (951, 3143),
          4: (962, 3208), 5: (0, 4095), 6: (1318, 2754)}
PERIOD = 0.02
POSE_TOLERANCE = 80
POSITION_TOLERANCES = {2: 20, 3: 40, 4: 15}  # ID3 deadband/load allowance
sdk = PacketHandler(0)


def connect(path):
    port = PortHandler(path)
    if not port.openPort():
        raise RuntimeError(f'Cannot open {path}')
    if not port.setBaudRate(1000000):
        port.closePort()
        raise RuntimeError(f'Cannot set baud rate: {path}')
    return port


def read_position(port, mid):
    for _ in range(5):
        value, result, err = sdk.read2ByteTxRx(port, mid, 56)
        if result == COMM_SUCCESS and err == 0:
            return value
        time.sleep(0.01)
    raise RuntimeError(f'ID {mid}: position read failed')


def write1(port, mid, addr, val):
    for _ in range(5):
        result, err = sdk.write1ByteTxRx(port, mid, addr, int(val))
        if result == COMM_SUCCESS and err == 0:
            return
        time.sleep(0.01)
    raise RuntimeError(f'ID {mid}: write failed at {addr} ({result}/{err})')


def goal(port, mid, val):
    for _ in range(5):
        result, err = sdk.write2ByteTxRx(port, mid, 42, int(val))
        if result == COMM_SUCCESS and err == 0:
            return
        time.sleep(0.01)
    raise RuntimeError(f'ID {mid}: goal write failed ({result}/{err})')


def positions(port, ids=ALL):
    return {i: read_position(port, i) for i in ids}


def clamp(mid, pos):
    low, high = LIMITS[mid]
    return max(low, min(high, int(round(pos))))


def torque_off(port, ids=ALL):
    if port is None:
        return
    for mid in ids:
        for attempt in range(3):
            try:
                write1(port, mid, 40, 0)
                break
            except Exception:
                time.sleep(0.02)
        else:
            print(f'WARNING: ID {mid} torque OFF not confirmed')


def prepare_hold(port, initial):
    # ID5 is NEVER powered for holding in this variant.
    # Disable all axes first; never send ID5 a position goal.
    torque_off(port)
    for mid in ALL:
        if mid != 5:
            goal(port, mid, initial[mid])
    # Check ID5 is torque-off before enabling any other axis.
    torque, result, error = sdk.read1ByteTxRx(port, 5, 40)
    if result != COMM_SUCCESS or error != 0 or torque != 0:
        raise RuntimeError(f'ID5 torque-off not verified: value={torque}, result={result}, error={error}')
    try:
        for mid in ALL:
            if mid != 5:
                write1(port, mid, 40, 1)
    except Exception:
        torque_off(port)
        raise


def record(args):
    leader = follower = None
    samples = []
    started = False
    try:
        leader = connect(LEADER)
        follower = connect(FOLLOWER)
        start_l = positions(leader, MOVING)
        start_f = positions(follower)
        print('Leader initial:', start_l)
        print('Follower initial:', start_f)
        for mid in ALL:
            low, high = LIMITS[mid]
            if not low <= start_f[mid] <= high:
                print(f'WARNING ID {mid}: start {start_f[mid]} outside calibration [{low}, {high}]')
        # Reject large out-of-range initial positions rather than sending a large
        # corrective motion immediately (previous tests found small excursions).
        for mid in MOVING:
            low, high = LIMITS[mid]
            if not low - 60 <= start_f[mid] <= high + 60:
                raise RuntimeError(f'ID {mid}: initial pose too far outside calibration limits')
        prepare_hold(follower, start_f)
        started = True
        print(f'Recording {args.duration}s: follower IDs 2-4 follow; IDs 1,5,6 held. Ctrl+C to save.')
        t0 = time.monotonic()
        next_tick = t0
        while time.monotonic() - t0 < args.duration:
            target = {}
            for mid in MOVING:
                raw = start_f[mid] + read_position(leader, mid) - start_l[mid]
                target[mid] = clamp(mid, raw)
            for mid in MOVING:
                goal(follower, mid, target[mid])
            samples.append({'t': round(time.monotonic()-t0, 5),
                            'goals': {str(mid): target[mid] for mid in MOVING}})
            next_tick += PERIOD
            wait = next_tick - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print('\nRecording interrupted; saving samples.')
    finally:
        if follower:
            if started:
                torque_off(follower)
            follower.closePort()
        if leader:
            leader.closePort()
    if samples:
        data = {'created': datetime.now().isoformat(), 'moving_ids': list(MOVING),
                'fixed_ids': [1, 5, 6], 'leader_start': {str(k): v for k, v in start_l.items()},
                'follower_start': {str(k): v for k, v in start_f.items()},
                'samples': samples}
        Path(args.file).write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(f'Saved {args.file} ({len(samples)} samples, {samples[-1]["t"]:.2f}s)')
    else:
        print('No samples saved')


def move_to_recorded_start(port, target, args):
    """Feedback-guided incremental positioning with a persistent setpoint ramp."""
    now = positions(port, MOVING)
    for mid in MOVING:
        low, high = LIMITS[mid]
        if not low <= target[mid] <= high:
            raise RuntimeError(f'ID {mid}: target {target[mid]} outside calibration [{low}, {high}]')
        if not low - 60 <= now[mid] <= high + 60:
            raise RuntimeError(f'ID {mid}: start {now[mid]} outside allowed startup margin')

    # Keep the commanded trajectory persistent. Using now+8 on every cycle
    # can hold a motor inside its position deadband indefinitely.
    commanded = dict(now)
    start_time = last_progress = time.monotonic()
    best_error = sum(abs(now[i] - target[i]) for i in MOVING)
    next_report = start_time
    print('Positioning: 8 counts/0.1s, max 40-count tracking gap. Ctrl+C to abort.')
    while True:
        now = positions(port, MOVING)
        error = sum(abs(now[i] - target[i]) for i in MOVING)
        if all(abs(now[i] - target[i]) <= POSITION_TOLERANCES[i] for i in MOVING):
            # Hold at actual reached positions, do not force an extra final jump.
            for mid in MOVING:
                goal(port, mid, now[mid])
            print('Start pose reached (within tolerances):', now)
            return
        current_time = time.monotonic()
        if error < best_error - 5:
            best_error = error
            last_progress = current_time
        if current_time - last_progress > 6:
            raise RuntimeError(f'No position progress in 6s; stopped at {now}; target {target}')
        if current_time - start_time > 100:
            raise RuntimeError(f'Auto positioning timed out at {now}')
        for mid in MOVING:
            # Do not run ahead of real motor position by more than 40 counts.
            if abs(commanded[mid] - now[mid]) > 40:
                continue
            diff = target[mid] - commanded[mid]
            step = max(-args.position_step, min(args.position_step, diff))
            commanded[mid] = clamp(mid, commanded[mid] + step)
            goal(port, mid, commanded[mid])
        if current_time >= next_report:
            print('  current:', now, 'target:', target, flush=True)
            next_report = current_time + 2
        time.sleep(0.1)


def replay(args):
    data = json.loads(Path(args.file).read_text(encoding='utf-8'))
    if data.get('moving_ids') != list(MOVING) or not data.get('samples'):
        raise RuntimeError('Recording invalid or incompatible')
    recorded_start = {int(k): int(v) for k, v in data['follower_start'].items()}
    follower = None
    try:
        follower = connect(FOLLOWER)
        current = positions(follower)
        difference = {mid: current[mid] - recorded_start[mid] for mid in ALL}
        print('Current vs recorded initial pose offsets:', difference)
        fixed_mismatch = {mid: difference[mid] for mid in (1, 6)
                          if abs(difference[mid]) > POSE_TOLERANCE}
        if fixed_mismatch:
            raise RuntimeError(f'Fixed axes 1,6 do not match recording: {fixed_mismatch}')
        # Recorded initial encoder positions may be a few counts outside the
        # calibration interval. Commands were already clamped when recorded.
        # Use the same clamped position for replay start, without expanding limits.
        for mid in MOVING:
            lo, hi = LIMITS[mid]
            raw_start = recorded_start[mid]
            if not lo - 60 <= raw_start <= hi + 60:
                raise RuntimeError(f'ID {mid}: recorded start {raw_start} too far outside calibrated range [{lo}, {hi}]')
            recorded_start[mid] = clamp(mid, raw_start)
            if raw_start != recorded_start[mid]:
                print(f'ID {mid}: start adjusted {raw_start} -> {recorded_start[mid]} to match recorded commands')
        if args.auto_position:
            print('Auto positioning target:', {i: recorded_start[i] for i in MOVING})
            input('Keep space clear. Enter to START LOW-SPEED positioning (Ctrl+C to abort)... ')
        elif any(abs(difference[mid]) > POSE_TOLERANCE for mid in MOVING):
            raise RuntimeError('Initial pose mismatch. Use --auto-position to move slowly to recorded start.')
        prepare_hold(follower, current)
        if args.auto_position:
            move_to_recorded_start(follower, recorded_start, args)
        else:
            current_m = positions(follower, MOVING)
            if any(abs(current_m[mid] - recorded_start[mid]) > POSE_TOLERANCE for mid in MOVING):
                raise RuntimeError('Initial pose mismatch after torque enabled')
        print(f'Replay at {args.speed:.0%} speed. Keep clear of moving parts.')
        input('Press Enter to start REPLAY... ')
        t0 = time.monotonic()
        for sample in data['samples']:
            target_t = float(sample['t']) / args.speed
            remaining = target_t - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
            for mid in MOVING:
                goal(follower, mid, clamp(mid, sample['goals'][str(mid)]))
        print('Replay completed')
    except KeyboardInterrupt:
        print('\nOperation interrupted')
    finally:
        if follower:
            # Avoid sudden gravity-induced collapse when the loop stops or raises.
            try:
                actual = positions(follower)
                for mid in ALL:
                    if mid != 5:
                        goal(follower, mid, actual[mid])
                print('Follower holding current pose (ID5 torque OFF):', actual, flush=True)
                print('Support the arm before releasing torque.', flush=True)
                try:
                    input('When the arm is SUPPORTED, press Enter to release torque... ')
                except (KeyboardInterrupt, EOFError):
                    print('Input interrupted; releasing torque. Support the arm now.', flush=True)
            except Exception as exc:
                print(f'WARNING: unable to set pose hold: {exc}', flush=True)
                try:
                    input('Support the arm. Press Enter to release torque... ')
                except (KeyboardInterrupt, EOFError):
                    pass
            torque_off(follower)
            follower.closePort()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['record', 'replay'])
    parser.add_argument('--file', default='ball_throw_01.json')
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--auto-position', action='store_true',
                        help='Slowly move IDs 2-4 to recorded replay start first')
    parser.add_argument('--position-step', type=int, default=8,
                        help='Maximum encoder counts per 0.1 s positioning cycle')
    args = parser.parse_args()
    if not 1 <= args.position_step <= 15:
        parser.error('position-step must be 1..15')
    if args.duration <= 0 or not 0 < args.speed <= 1:
        parser.error('duration must be >0 and speed must be in (0,1]')
    if args.mode == 'record':
        record(args)
    else:
        replay(args)

if __name__ == '__main__':
    main()
