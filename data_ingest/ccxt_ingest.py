import argparse
import time
from typing import List
import pandas as pd
import ccxt
import pyarrow as pa
import os
import pyarrow.parquet as pq
from ccxt.base.errors import ExchangeNotAvailable, BadSymbol

def timeframe_to_ms(tf: str) -> int:
    tf = tf.lower().strip()
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == 'm':
        return n * 60_000
    if unit == 'h':
        return n * 3_600_000
    if unit == 'd':
        return n * 86_400_000
    if unit == 'w':
        return n * 604_800_000
    raise ValueError(f"Unsupported timeframe: {tf}")

def iso_to_ms(iso: str) -> int:
    return int(pd.Timestamp(iso, tz='UTC').to_datetime64().astype('datetime64[ms]').astype(int))

def build_exchange(exchange_id: str):
    if not hasattr(ccxt, exchange_id):
        raise SystemExit(f"Unknown exchange '{exchange_id}'. Known ids: {', '.join(sorted(ccxt.exchanges))}")
    return getattr(ccxt, exchange_id)({'enableRateLimit': True})

def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int, per_call_limit: int = 1000) -> pd.DataFrame:
    tf_ms = timeframe_to_ms(timeframe)
    all_rows: List[list] = []
    cursor = since_ms
    last_progress = None
    while cursor <= until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=per_call_limit)
        if not batch:
            break
        filtered = [row for row in batch if row[0] <= until_ms]
        if filtered:
            all_rows.extend(filtered)
        last_ts = batch[-1][0]
        if last_progress is not None and last_ts <= last_progress:
            break
        last_progress = last_ts
        cursor = last_ts + tf_ms
        time.sleep((getattr(exchange, 'rateLimit', 250)) / 1000.0)
    if not all_rows:
        return pd.DataFrame(columns=['ts','open','high','low','close','volume'])
    df = pd.DataFrame(all_rows, columns=['ms','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ms'], unit='ms', utc=True)
    df = df.drop(columns=['ms']).drop_duplicates(subset=['ts']).sort_values('ts')
    return df[['ts','open','high','low','close','volume']]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exchange', default='coinbase')
    p.add_argument('--symbol', required=True)
    p.add_argument('--timeframe', default='1m')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    ex = build_exchange(args.exchange)
    ex.load_markets()
    if args.symbol not in ex.symbols:
        cand = args.symbol.upper().replace('-', '/')
        if cand in ex.symbols:
            symbol = cand
        else:
            raise SystemExit(f"Symbol '{args.symbol}' not found on {ex.id}." )
    else:
        symbol = args.symbol

    since_ms = iso_to_ms(args.start + ' 00:00:00')
    until_ms = iso_to_ms(args.end + ' 23:59:59')

    try:
        df = fetch_ohlcv(ex, symbol, args.timeframe, since_ms, until_ms, per_call_limit=1000)
    except ExchangeNotAvailable as e:
        raise SystemExit(str(e))

    if df.empty:
        raise SystemExit('No data returned; check symbol/timeframe/date range.')

    df['symbol'] = symbol.replace('/', '')
    df['timeframe'] = args.timeframe
    df['vendor'] = ex.id
    df = df.sort_values('ts').drop_duplicates(subset=['ts'])
    assert df['ts'].is_monotonic_increasing, 'Timestamps not monotonic'
    assert (df[['open','high','low','close','volume']] >= 0).all().all(), 'Negative values found'
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.out)

    tf_ms = timeframe_to_ms(args.timeframe)
    approx_expected = int((until_ms - since_ms) // tf_ms) + 1
    first_ts = df['ts'].iloc[0].isoformat()
    last_ts = df['ts'].iloc[-1].isoformat()
    print(f"Wrote {len(df):,} rows from {ex.id} {symbol} {args.timeframe} to {args.out}")
    print(f"Range covered: {first_ts} → {last_ts} (expected ~{approx_expected} bars if fully continuous)")

if __name__ == '__main__':
    main()
