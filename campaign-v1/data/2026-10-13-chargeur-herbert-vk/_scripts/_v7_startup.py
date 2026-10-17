# _v7_startup.py — one start of one server, timed from the client side: launch the command, poll its health URL
# every 5 ms until it answers 200 (ready), then one request of 1 token (first token). The server's stderr is kept.
# Usage: _v7_startup.py --cmd "..." --health URL --msg URL --model NAME --log FILE  → one JSON line on stdout.
import argparse, json, os, signal, subprocess, time, urllib.request
ap = argparse.ArgumentParser()
for a in ("--cmd", "--health", "--msg", "--model", "--log"): ap.add_argument(a, required=True)
ap.add_argument("--timeout", type=float, default=600)
a = ap.parse_args()
body = json.dumps({"model": a.model, "max_tokens": 1, "stream": False, "messages": [{"role": "user", "content": "Bonjour, qui es-tu ?"}]}).encode()
log = open(a.log, "w")
t0 = time.monotonic()
p = subprocess.Popen(a.cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
ready = first = None; err = None
try:
    while time.monotonic() - t0 < a.timeout:
        if p.poll() is not None: err = f"exited {p.returncode}"; break
        try:
            with urllib.request.urlopen(a.health, timeout=1) as r:
                if r.status == 200: ready = time.monotonic() - t0; break
        except Exception: pass
        time.sleep(0.005)
    if ready is not None:
        req = urllib.request.Request(a.msg, data=body, headers={"content-type": "application/json", "x-api-key": "none", "anthropic-version": "2023-06-01"})
        t1 = time.monotonic()
        with urllib.request.urlopen(req, timeout=a.timeout) as r: r.read()
        first = time.monotonic() - t0; req_s = time.monotonic() - t1
except Exception as e: err = repr(e)
finally:
    try: os.killpg(p.pid, signal.SIGTERM)
    except Exception: pass
    try: p.wait(30)
    except Exception:
        os.killpg(p.pid, signal.SIGKILL); p.wait()
print(json.dumps({"ready_s": ready, "first_token_s": first, "request_s": req_s if first else None, "error": err}), flush=True)
