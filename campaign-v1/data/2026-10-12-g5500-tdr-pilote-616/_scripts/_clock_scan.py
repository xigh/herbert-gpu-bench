# Temperatures of the g5500 clock logs (nvidia-smi, 0.5 s) and the watchdog rule of _watchdog-win.sh replayed
# on each: 90 °C, or 20 identical rows (all fields but the timestamp) outside P8. Usage: _clock_scan.py CSV...
import collections, sys
for f in sys.argv[1:]:
    rows = [l.rstrip('\r\n') for l in open(f, errors='replace')][1:]
    t = [int(r.split(', ')[5]) for r in rows if len(r.split(', ')) >= 6 and r.split(', ')[5].isdigit()]
    bins = collections.Counter(x // 5 * 5 for x in t)
    first = None
    for i in range(19, len(rows)):
        w, last = rows[i - 19:i + 1], rows[i]
        p = last.split(', ')
        if len(p) >= 6 and p[5].isdigit() and int(p[5]) >= 90: first = (last, 'hot'); break
        if ', P8,' not in last and len({r.split(',', 1)[1] for r in w}) == 1: first = (last, 'frozen'); break
    print(f'{f}: {len(rows)} rows, max {max(t)} C, by 5 C: {dict(sorted(bins.items()))}')
    print(f'  watchdog would stop at: {first}')
