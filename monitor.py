#!/usr/bin/env python3
"""Pipeline monitor. Refreshes every 10 minutes. Ctrl+C to stop."""
import sqlite3, json, os, subprocess, time
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
DB = BASE / 'etsy_data.db'

def report():
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    c = conn.cursor()

    print('\033[2J\033[H', end='')  # Clear screen
    print('=' * 60)
    print(f'  PIPELINE MONITOR — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # --- DISK ---
    df = subprocess.run(['df', '-h', str(BASE)], capture_output=True, text=True)
    parts = df.stdout.strip().split('\n')[1].split()
    ssd = subprocess.run(['df', '-h', '/Volumes/SSD_120'], capture_output=True, text=True)
    ssd_parts = ssd.stdout.strip().split('\n')[1].split() if ssd.returncode == 0 else None

    print(f'\n  DISK')
    print(f'  ├─ Main:  {parts[3]} free / {parts[1]} ({parts[4]} used)')
    if ssd_parts:
        print(f'  └─ SSD:   {ssd_parts[3]} free / {ssd_parts[1]} ({ssd_parts[4]} used)')

    # --- DATABASE REPAIR ---
    c.execute('''SELECT COUNT(DISTINCT listing_id) FROM image_status
                 WHERE listing_id NOT IN (SELECT listing_id FROM image_status WHERE is_primary = 1)''')
    no_primary = c.fetchone()[0]
    c.execute('SELECT COUNT(DISTINCT listing_id) FROM image_status')
    total_listings = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM image_status WHERE url IS NULL OR url = ''")
    no_url = c.fetchone()[0]

    print(f'\n  DATABASE REPAIR')
    print(f'  ├─ Listings missing primary:  {no_primary:>10,} / {total_listings:,}')
    print(f'  └─ Images missing URL:        {no_url:>10,}')

    # --- SYNC DATA ---
    sync_log = BASE / 'sync_data.log'
    sync_status = 'not running'
    if sync_log.exists():
        lines = sync_log.read_text().strip().split('\n')
        for line in reversed(lines[-10:]):
            if 'offset=' in line:
                sync_status = line.strip()
                break

    c.execute('SELECT COUNT(*) FROM listings_static')
    static = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM listings_dynamic')
    dynamic = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM shops_static')
    shops_static = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM shops_dynamic')
    shops_dynamic = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM reviews')
    reviews_count = c.fetchone()[0]

    print(f'\n  SYNC DATA')
    print(f'  ├─ listings_static:     {static:>10,}')
    print(f'  ├─ listings_dynamic:    {dynamic:>10,}')
    print(f'  ├─ shops_static:        {shops_static:>10,}')
    print(f'  ├─ shops_dynamic:       {shops_dynamic:>10,}')
    print(f'  ├─ reviews:             {reviews_count:>10,}')
    print(f'  └─ Last: {sync_status}')

    # --- DOWNLOADER ---
    c.execute('SELECT download_done, COUNT(*) FROM image_status GROUP BY download_done')
    dl_counts = dict(c.fetchall())
    total = sum(dl_counts.values())

    dl_log = BASE / 'image_downloader.log'
    dl_status = ''
    if dl_log.exists():
        lines = dl_log.read_text().strip().split('\n')
        for line in reversed(lines[-10:]):
            if 'Scan:' in line:
                dl_status = line.strip()
                break

    print(f'\n  DOWNLOADER ({total:,} total images)')
    print(f'  ├─ Needs download (0):  {dl_counts.get(0, 0):>10,}')
    print(f'  ├─ Downloading (1):     {dl_counts.get(1, 0):>10,}')
    print(f'  ├─ Complete (2):        {dl_counts.get(2, 0):>10,}')
    print(f'  └─ Last: {dl_status}')

    # --- EMBEDDINGS & CONSISTENCY ---
    emb_dir = BASE / 'embeddings'
    idx_file = emb_dir / 'image_index.json'
    idx_count = 0
    if idx_file.exists():
        idx_count = len(json.loads(idx_file.read_text()))

    faiss_counts = {}
    if emb_dir.exists():
        for f in sorted(emb_dir.glob('*.faiss')):
            import faiss
            index = faiss.read_index(str(f))
            faiss_counts[f.name] = index.ntotal

    print(f'\n  EMBEDDINGS')
    print(f'  ├─ image_index.json:    {idx_count:>10,}')
    if faiss_counts:
        for name, cnt in faiss_counts.items():
            sym = '├' if name != list(faiss_counts.keys())[-1] else '└'
            print(f'  {sym}─ {name:25s} {cnt:>10,}')
    else:
        print(f'  └─ No FAISS files yet')

    # Consistency
    img_counts = set(v for k, v in faiss_counts.items() if '_text' not in k)
    txt_counts = set(v for k, v in faiss_counts.items() if '_text' in k)
    all_match = len(img_counts) <= 1 and len(txt_counts) <= 1
    idx_match = not img_counts or (list(img_counts)[0] == idx_count)
    consistent = all_match and idx_match
    print(f'\n  CONSISTENCY: {"OK" if consistent else "MISMATCH!"}')
    if not consistent:
        if not all_match:
            print(f'    FAISS files have different counts!')
        if not idx_match:
            print(f'    image_index ({idx_count}) != FAISS ({list(img_counts)[0] if img_counts else 0})')

    # --- API ---
    qps_file = BASE / 'qps_config.json'
    if qps_file.exists():
        qps = json.loads(qps_file.read_text())
        used = qps.get('limit', 0) - qps.get('remaining', 0)
        print(f'\n  API: {used:,} / {qps.get("limit", 0):,} used today')

    # --- RUNNING ---
    ps = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    scripts = {'sync_data': False, 'image_downloader': False, 'embed_orchestrator': False}
    for line in ps.stdout.split('\n'):
        if 'grep' not in line and 'python' in line:
            for s in scripts:
                if s in line:
                    scripts[s] = True
    running = [s for s, v in scripts.items() if v]
    stopped = [s for s, v in scripts.items() if not v]
    print(f'\n  RUNNING: {", ".join(running) if running else "NONE"}')
    if stopped:
        print(f'  STOPPED: {", ".join(stopped)}')

    print('\n' + '=' * 60)
    print('  Next refresh in 10 minutes. Ctrl+C to stop.')
    print('=' * 60, flush=True)
    conn.close()


if __name__ == '__main__':
    try:
        while True:
            report()
            time.sleep(600)
    except KeyboardInterrupt:
        print('\nMonitor stopped.')
