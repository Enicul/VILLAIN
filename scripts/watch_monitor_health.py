#!/usr/bin/env python3
"""Watchdog for the VILLAIN sweep monitor."""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/aied_test/VILLAIN')
MONITOR_PID = ROOT / 'outputs-qwen25-cot-sweep-monitor.pid'
MONITOR_LOG = ROOT / 'outputs-qwen25-cot-sweep-monitor.log'
WATCHDOG_LOG = ROOT / 'outputs-qwen25-cot-sweep-watchdog.log'
CHECK_SECONDS = 15 * 60
STALE_SECONDS = 35 * 60
RESTART_COOLDOWN_SECONDS = 20 * 60
last_restart = 0


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def write(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with WATCHDOG_LOG.open('a') as f:
        f.write(line + '\n')


def pid_alive(pid_text):
    try:
        return Path(f'/proc/{int(pid_text.strip())}').exists()
    except Exception:
        return False


def restart_monitor(reason):
    global last_restart
    now_ts = time.time()
    if now_ts - last_restart < RESTART_COOLDOWN_SECONDS:
        write(f'ALERT {reason}; restart suppressed by cooldown')
        return
    last_restart = now_ts
    if MONITOR_PID.exists() and pid_alive(MONITOR_PID.read_text()):
        try:
            os.kill(int(MONITOR_PID.read_text().strip()), 15)
        except Exception:
            pass
    log_file = ROOT / 'outputs-qwen25-cot-sweep-monitor.nohup.log'
    with log_file.open('a') as fh:
        proc = subprocess.Popen(
            ['python', 'scripts/monitor_averimatec_sweep.py'],
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    MONITOR_PID.write_text(str(proc.pid) + '\n')
    write(f'ALERT {reason}; restarted monitor pid={proc.pid}')


def main():
    write('watchdog started: checking monitor process/log freshness every 15 minutes')
    while True:
        alive = False
        if MONITOR_PID.exists():
            alive = pid_alive(MONITOR_PID.read_text())
        if not alive:
            restart_monitor('monitor process is not alive')
        elif not MONITOR_LOG.exists():
            restart_monitor('monitor log does not exist')
        else:
            age = time.time() - MONITOR_LOG.stat().st_mtime
            if age > STALE_SECONDS:
                restart_monitor(f'monitor log stale for {age/60:.1f} minutes')
            else:
                write(f'ok monitor alive; log age {age/60:.1f} minutes')
        time.sleep(CHECK_SECONDS)


if __name__ == '__main__':
    main()
