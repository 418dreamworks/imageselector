# Etsy Furniture Image Pipeline

## CRITICAL RULES

**NEVER DO THESE:**
- `pkill` or `kill -9` on any script — Use KILL files instead
- Use system python — Use `venv/bin/python3`
- Propose or execute deletion of files/data unless user explicitly asks
- Create, move, or rename files/folders at the project root level
- Use `venv/bin/python` (symlink doesn't exist) — Use `venv/bin/python3`

**ALWAYS DO THESE:**
- Use KILL files to stop scripts gracefully (KILL_SD, KILL_DL, KILL_ORCH)
- Use `PYTHONUNBUFFERED=1` when running Python scripts in background
- Write temporary/one-off scripts to `scratch/`, never at the top level
- Check what was done before — don't invent new approaches
- Backup format: `backups/data_TIMESTAMP/` with `chmod 444` on files

---

## Project Overview

Ensemble image search system for Etsy furniture listings. Multiple embedding models, matches scoring well across models are more reliable.

```
sync_data.py ──► image_downloader.py ──► embed_orchestrator.py ──► tar_images.py ──► update_primary.py
     │                   │                         │                      │                  │
  Etsy API          Etsy CDN              3 workers (iMac, MBP,     imageembedded →      rebuild
  (5 QPS)          (8 workers)             Sleight)                 imagetarred      imageprimary
```

---

## Scripts

### Core Pipeline (bin/)

| File | Purpose |
|------|---------|
| `bin/sync_data.py` | Crawls Etsy API (5 QPS), discovers listings, fetches shop/review data |
| `bin/image_downloader.py` | Downloads images from Etsy CDN (8 workers, pauses at 5GB free) |
| `bin/image_db.py` | Shared database helpers with `@_retry_on_lock` decorator |
| `bin/tar_images.py` | Archives 10K batches from imageembedded → imagetarred (exits when done) |
| `bin/update_primary.py` | Weekly full rebuild: clear imageprimary/, extract all primaries, tar all in 10K batches (no loose files) |
| `bin/backup_db.py` | Backs up entire data/ folder to HDD1TB |
| `bin/embedding/embed_orchestrator.py` | Distributes batches to 3 workers, imports to FAISS (exits when no work left) |
| `bin/embedding/embed_worker.py` | BG removal + 5-model embedding (overwrites JPGs with bg-removed, produces .npy; after import, JPGs move to images/imageembedded/) |
| `bin/embedding/models.py` | Model definitions and lazy loading for CLIP/DINO |

### Kill Files

| Kill File | Stops |
|-----------|-------|
| `KILL_SD` | sync_data.py |
| `KILL_DL` | image_downloader.py |
| `KILL_ORCH` | embed_orchestrator.py |

---

## Database

SQLite with WAL mode at `data/db/etsy_data.db`

### `image_status` — Image Pipeline Tracking
| Column | Type | Description |
|--------|------|-------------|
| listing_id | INTEGER | PK with image_id |
| image_id | INTEGER | PK with listing_id |
| is_primary | INTEGER | 1 if primary image for listing |
| download_done | INTEGER | -1=dead URL, 0=needs download, 1=marker created, 2=complete |
| url | TEXT | Etsy CDN URL |

### Other Tables
- `listings` — static listing data (title, description, tags, materials, price, dimensions, taxonomy_id)
- `listings_dynamic` — time-series metrics (state, quantity, num_favorers, views, price)
- `shops` — static shop data (create_date, url, location)
- `shops_dynamic` — time-series shop metrics (listing_active_count, num_favorers, review_count)
- `reviews` — shop reviews (rating, review text, buyer_user_id)
- `sync_state` — key-value crawler state

### Embedding Tracking
- Tracked via `data/embeddings/image_index.json` (not in DB)
- Row alignment: Row N in ALL FAISS files = same image. `image_index.json[N]` = `[listing_id, image_id]`

---

## Data & Image Layout

```
data/
  db/             etsy_data.db, qps_config.json, taxonomy configs
  embeddings/     FAISS indexes (7 files) + image_index.json

images/
  imagedownload/    Raw downloaded images (primary SSD)
  imageembedded/     BG-removed images waiting to be tarred (primary SSD)
  imagetarred/    Symlink → HDD1TB/images/imagetarred
  imageprimary/     Symlink → SSD500GB; contains primary image tars (no loose files)
```

---

## Embedding Models (5 Total)

| Model | Dimension | Library | Inputs |
|-------|-----------|---------|--------|
| clip_vitb32 | 512 | open_clip | image + text |
| clip_vitl14 | 768 | open_clip | image + text |
| dinov2_base | 768 | transformers | image only |
| dinov2_large | 1024 | transformers | image only |
| dinov3_base | 768 | transformers | image only |

---

## Weekly Cron Schedule

| Day | Time | Job |
|-----|------|-----|
| Every day | every 3h | tar_images.py |
| Sunday | 0:00 | sync_data (5 QPS) |
| Sunday | 4:00 | image_downloader (8 workers) |
| Sunday | 8:00 | embed_orchestrator |
| Sunday | 22:00 | tar_images |
| Monday | 0:00 | update_primary (full rebuild) |
| Monday | 4:00 | backup_db |
| Monday | 5:00 | rsync data/ → SSD500GB |
| Monday | 6:00 | HDD1TB → HDD3TB |
| Monday | 10:00 | SSD500GB → HDD500GB |

---

## Storage

### Physical Drives

| Location | What lives here |
|----------|----------------|
| **Primary SSD (228GB)** | `data/db/` (live DB — only permanent data here), `images/imagedownload/` (staging), `images/imageembedded/` (staging), `embed_exports/` (temporary) |
| **SSD500GB** (external) | `data/embeddings/` (5 FAISS shard dirs, symlinked), `images/imageprimary/` (primary image tars, symlinked), `data/db/` mirror (weekly copy) |
| **HDD1TB** (external) | `images/imagetarred/` (BG-removed tars, symlinked), `backups/` (timestamped snapshots, symlinked) |
| **HDD500GB** | Latest mirror of SSD500GB (no history) |
| **HDD3TB** | Latest mirror of HDD1TB (no history) |

### Symlinks (project root → actual location)

| Project path | Points to |
|-------------|-----------|
| `data/embeddings/` | `/Volumes/SSD500GB/imageselector/data/embeddings` |
| `images/imagetarred/` | `/Volumes/HDD1TB/images/imagetarred` |
| `images/imageprimary/` | `/Volumes/SSD500GB/imageselector/images/imageprimary` |
| `backups/` | `/Volumes/HDD1TB/backups` |

### Backup Strategy

Two chains — **timestamped snapshots** for rollback, **mirrors** for redundancy:

1. `backup_db.py` → HDD1TB/backups/ — **timestamped** DB snapshots (SQLite backup API) + shard snapshots. Can roll back to any backup date.
2. `rsync data/ → SSD500GB` — updates the **latest** DB copy on SSD500GB (no history).
3. `HDD1TB → HDD3TB` — redundancy mirror of the timestamped backups + tars.
4. `SSD500GB → HDD500GB` — redundancy mirror of the live embeddings + DB copy.

Result: DB exists in 5 places (Primary SSD live, SSD500GB mirror, HDD1TB timestamped, HDD500GB mirror, HDD3TB mirror). FAISS shards in 4 places (SSD500GB live, HDD1TB timestamped, HDD500GB mirror, HDD3TB mirror).

---

## Embed Workers

- 3 workers: imac (localhost), mbp (mbp.local), sleight (192.168.68.117)
- Orchestrator rsyncs batches to workers, workers run embed_worker.py
- To verify workers are running, check worker.log on each machine — NOT ps/orchestrator log
  - imac: `ssh embed@localhost "tail -c 200 /Users/embed/imageselector/embed_exports/BATCH_NAME/worker.log"`
  - mbp: `ssh embed@mbp.local "tail -c 200 /Users/embed/imageselector/embed_exports/BATCH_NAME/worker.log"`
  - sleight: `ssh embed@192.168.68.117 "powershell -Command \"Get-Content 'C:\\Users\\embed\\imageselector\\embed_exports\\BATCH_NAME\\worker.log' -Tail 3\""`
- DINOv3 requires HuggingFace token (`~/.huggingface/token` or `C:\Users\embed\.huggingface\token` on Windows)
- Models cached in `~/.cache/huggingface/hub/`; once downloaded, token not needed for subsequent runs
- Sleight (Windows): rsync is `swrsync`, not `rsync`

### Starting Workers Manually
Sleight (Windows):
```bash
ssh embed@192.168.68.117 'cd /d "C:\Users\embed\imageselector\bin\embedding" && "C:\Users\embed\imageselector\venv\Scripts\python.exe" embed_worker.py --input "C:\Users\embed\imageselector\embed_exports\BATCH_NAME" > "C:\Users\embed\imageselector\embed_exports\BATCH_NAME\worker.log" 2>&1' &
```
Mac workers (imac/mbp):
```bash
ssh embed@HOST "nohup PYTHON_PATH WORKER_SCRIPT --input BATCH_PATH > BATCH_PATH/worker.log 2>&1 &"
```

---

## Quick Commands

```bash
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector

# Check status
venv/bin/python3 bin/image_db.py

# Stop scripts gracefully
touch KILL_SD     # sync_data
touch KILL_DL     # image_downloader
touch KILL_ORCH   # orchestrator

# Run in background
PYTHONUNBUFFERED=1 nohup venv/bin/python3 bin/SCRIPT.py > logs/SCRIPT.log 2>&1 &

# Watch logs
tail -f logs/*.log
```

---

## Project Layout

```
bin/              Active scripts
  sync_data.py, tar_images.py, image_db.py, image_downloader.py
  update_primary.py, backup_db.py
  embedding/      embed_orchestrator.py, embed_worker.py, models.py
  init/           fetch_taxonomy.py, calculate_price_breaks.py
data/
  db/             etsy_data.db, qps_config.json, taxonomy configs
  embeddings/     FAISS indexes + image_index.json
images/
  imagedownload/  Raw downloaded images (primary SSD)
  imageembedded/   BG-removed images waiting to be tarred (primary SSD)
  imagetarred/  Symlink → HDD1TB
  imageprimary/   Symlink → SSD500GB (primary image tars)
scripts/          Inactive/legacy scripts
scratch/          Temporary/debug scripts
logs/             All log files
backups/          Symlink → HDD1TB/backups
embed_exports/    Batch staging for workers
venv/             Python virtual environment
```
