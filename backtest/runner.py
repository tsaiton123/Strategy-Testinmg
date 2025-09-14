# backtest/runner.py
import argparse, json, os, itertools, re
from typing import Dict, Any, List, Iterable
from .engine import run_backtest, CostModel

def parse_params(params: str) -> Dict[str, Any]:
    if not params:
        return {}
    try:
        return json.loads(params)
    except json.JSONDecodeError:
        out = {}
        for kv in params.split(','):
            if not kv.strip():
                continue
            if '=' not in kv:
                raise SystemExit(f"Invalid param entry '{kv}'. Use key=value or JSON.")
            k, v = kv.split('=', 1)
            v = v.strip()
            if v.lower() in ('true','false'):
                out[k] = (v.lower() == 'true')
            else:
                try:
                    out[k] = int(v)
                except ValueError:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
        return out

def _split_top_level(s: str, sep: str = ',') -> List[str]:
    """Split by sep but ignore separators inside [...] brackets."""
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]

def _split_items(s: str) -> List[str]:
    """Split list items on comma or pipe, trimming whitespace."""
    return [x.strip() for x in re.split(r"\s*[,|]\s*", s) if x.strip()]

def _cast(v: str):
    vl = v.lower()
    if vl in ('true','false'):
        return vl == 'true'
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def parse_grid(grid: str) -> Iterable[Dict[str, Any]]:
    if not grid:
        return []
    pairs = _split_top_level(grid, ',')
    d = {}
    for part in pairs:
        if '=' not in part:
            raise SystemExit(f"Invalid grid segment '{part}'. Expected key=[v1,v2] or key=[v1|v2].")
        k, arr = part.split('=', 1)
        k, arr = k.strip(), arr.strip()
        if arr.startswith('[') and arr.endswith(']'):
            arr = arr[1:-1]
        items = _split_items(arr)
        if not items:
            raise SystemExit(f"Empty item list for '{k}' in grid.")
        d[k] = [_cast(x) for x in items]
    keys = list(d.keys())
    for combo in itertools.product(*[d[k] for k in keys]):
        yield dict(zip(keys, combo))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Parquet path')
    p.add_argument('--strategy', required=True, help='Import path, e.g., strategies.sma:SMA')
    p.add_argument('--params', default='', help='JSON or key=val pairs')
    p.add_argument('--grid', default='', help='Param grid, e.g., fast=[20,50],slow=[100,200] (pipes | also supported)')
    p.add_argument('--fee_bps', type=float, default=10.0)
    p.add_argument('--slip_bps', type=float, default=0.0)
    p.add_argument('--out', default='artifacts')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cost = CostModel(fee_bps=args.fee_bps, slip_bps=args.slip_bps)

    if args.grid:
        i = 0
        best = None
        for params in parse_grid(args.grid):
            subdir = os.path.join(args.out, f"run_{i:03d}")
            os.makedirs(subdir, exist_ok=True)
            res = run_backtest(args.data, args.strategy, params, cost, subdir)
            score = res['stats'].get('Sharpe', 0.0)
            if best is None or score > best[0]:
                best = (score, subdir, params, res['stats'])
            i += 1
        if best is not None:
            print("Best Sharpe:", best[0], "in", best[1], "params=", best[2])
        else:
            print("No runs executed. Check your --grid format.")
    else:
        params = parse_params(args.params)
        res = run_backtest(args.data, args.strategy, params, cost, args.out)
        print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
