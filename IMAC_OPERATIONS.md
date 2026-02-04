# iMac Operations Guide

Complete guide for running the Etsy image pipeline directly on the iMac terminal with Claude Code.

---

## Quick Reference

```bash
# Project location
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector

# Always use venv python
venv/bin/python <script>

# Check pipeline status
venv/bin/python image_db.py

# Stop everything gracefully
touch KILL
```

---

## 1. Environment Setup

### Project Path
```
/Users/tzuohannlaw/Documents/418Dreamworks/imageselector
```

### Python Environment
**CRITICAL**: Always use the virtual environment, never system Python.

```bash
# Correct
venv/bin/python script.py

# WRONG - will fail with missing packages
python script.py
python3 script.py
```

### Required Environment Variables
The `.env` file contains:
```
ETSY_API_KEY=<your-api-key>
```

This is loaded automatically by scripts using `python-dotenv`.

---

## 2. Directory Structure

```
imageselector/
├── images/                    # Downloaded images (945K+ files)
├── images_nobg/               # Background-removed images (deprecated, not used)
├── embeddings/                # FAISS index files per model
│   ├── clip_vitb32.faiss
│   ├── clip_vitb32_text.faiss
│   ├── clip_vitl14.faiss
│   ├── clip_vitl14_text.faiss
│   ├── dinov2_base.faiss
│   ├── dinov2_large.faiss
│   ├── dinov3_base.faiss
│   └── image_index.json       # Maps row index → (listing_id, image_id)
├── etsy_data.db               # Main SQLite database (WAL mode)
├── etsy_data.db-wal           # WAL file (don't delete while running)
├── etsy_data.db-shm           # Shared memory file
├── qps_config.json            # API rate limit config (auto-adjusted)
├── pipeline.log               # Pipeline monitor log
├── sync_data.log              # Sync script log
├── image_downloader.log       # Downloader log
├── bg_remover.log             # Background remover log
├── embed.log                  # Embedding script log
└── scripts/
    ├── pipeline_monitor.py    # Manages all 4 scripts
    ├── image_downloader.py    # Downloads images from Etsy CDN
    ├── bg_remover.py          # Removes backgrounds (GPU)
    └── cleanup_images.py      # Moves non-primary images to HDD
```

---

## 3. The Pipeline

### Flow Diagram
```
sync_data.py ──────► image_downloader.py ──────► bg_remover.py ──────► embed.py
     │                      │                         │                    │
     ▼                      ▼                         ▼                    ▼
 Discovers            Downloads from            Removes BG,          Generates
 listings via         Etsy CDN to              overwrites in        embeddings
 Etsy API             images/                  images/              for 5 models
     │                      │                         │                    │
     ▼                      ▼                         ▼                    ▼
 Sets                 Sets                      Sets                 Sets
 to_download=1        download_done=1          bg_removed=1         embed_{model}=1
```

### Scripts

| Script | Purpose | Kill File | Log File |
|--------|---------|-----------|----------|
| `sync_data.py` | Crawls Etsy API, discovers listings, adds to DB | `KILL` | `sync_data.log` |
| `scripts/image_downloader.py` | Downloads images from Etsy CDN | `KILL_DL` | `image_downloader.log` |
| `scripts/bg_remover.py` | Removes backgrounds using rembg + CoreML GPU | `KILL_BG` | `bg_remover.log` |
| `image_search_v2/embed.py` | Generates embeddings for all 5 models | `KILL_EMBED` | `embed.log` |
| `scripts/pipeline_monitor.py` | Manages all scripts, auto-restarts on crash | `KILL` | `pipeline.log` |

---

## 4. Starting the Pipeline

### Option A: Pipeline Monitor (Recommended)
Manages all 4 scripts, auto-restarts them if they crash, adjusts API rate limits.

```bash
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector
nohup venv/bin/python scripts/pipeline_monitor.py > pipeline.log 2>&1 &
```

To run in foreground (see output directly):
```bash
venv/bin/python scripts/pipeline_monitor.py
```

### Option B: Run Scripts Individually
For debugging or running specific scripts:

```bash
# Terminal 1: Sync data
venv/bin/python sync_data.py

# Terminal 2: Download images
venv/bin/python scripts/image_downloader.py --watch

# Terminal 3: Background removal
venv/bin/python scripts/bg_remover.py --watch --gpu

# Terminal 4: Embeddings
venv/bin/python image_search_v2/embed.py --model all --batch-size 8
```

---

## 5. Stopping the Pipeline

### Graceful Stop (ALWAYS USE THIS)
```bash
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector
touch KILL
```

This signals pipeline_monitor.py to:
1. Create individual kill files (KILL_DL, KILL_BG, KILL_EMBED)
2. Wait up to 30 seconds for each script to exit
3. Shut down cleanly

### Stop Individual Scripts
```bash
touch KILL      # Stops sync_data.py (and pipeline_monitor)
touch KILL_DL   # Stops image_downloader.py
touch KILL_BG   # Stops bg_remover.py
touch KILL_EMBED # Stops embed.py
```

### NEVER DO THIS
```bash
# WRONG - can corrupt database or lose embeddings
pkill -f sync_data
kill -9 <pid>
killall python
```

---

## 6. Monitoring Progress

### Quick Status Check
```bash
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector
venv/bin/python image_db.py
```

Output:
```
=== Image Status ===

Total images:      1,234,567
To download:       1,234,567
Download done:       945,632
BG removed:          277,558
Embedded clip32:      79,664
Embedded clip14:           0
Embedded dino2b:           0
Embedded dino2l:           0
Embedded dino3b:           0
```

### Check Running Processes
```bash
ps aux | grep -E "(sync_data|image_downloader|bg_remover|embed|pipeline)" | grep -v grep
```

### Watch Logs in Real-Time
```bash
# Pipeline monitor (overview)
tail -f pipeline.log

# Individual scripts
tail -f sync_data.log
tail -f image_downloader.log
tail -f bg_remover.log
tail -f embed.log

# All logs at once
tail -f *.log
```

### Check API Rate Limits
```bash
cat qps_config.json
```

---

## 7. Database Schema

### Main Tables

#### `image_status` - Tracks every image through the pipeline
```sql
CREATE TABLE image_status (
    listing_id INTEGER,
    image_id INTEGER,
    shop_id INTEGER,
    hex TEXT,                    -- CDN hex path component
    suffix TEXT,                 -- CDN suffix component
    is_primary INTEGER,          -- 1 if primary image for listing
    when_made TEXT,              -- "2020_2024", "before_2000", etc.
    price REAL,
    to_download INTEGER DEFAULT 1,
    download_done INTEGER DEFAULT 0,
    bg_removed INTEGER DEFAULT 0,
    embed_clip_vitb32 INTEGER DEFAULT 0,
    embed_clip_vitl14 INTEGER DEFAULT 0,
    embed_dinov2_base INTEGER DEFAULT 0,
    embed_dinov2_large INTEGER DEFAULT 0,
    embed_dinov3_base INTEGER DEFAULT 0,
    PRIMARY KEY (listing_id, image_id)
);
```

#### `listings` - Static listing data
```sql
-- listing_id, shop_id, title, description, url, tags, materials, when_made, etc.
```

#### `listings_dynamic` - Time-series listing metrics
```sql
-- listing_id, snapshot_timestamp, state, quantity, num_favorers, views, price
```

#### `shops` - Static shop data
```sql
-- shop_id, create_date, url, location info
```

#### `shops_dynamic` - Time-series shop metrics
```sql
-- shop_id, snapshot_timestamp, listing_active_count, num_favorers, review_count, etc.
```

#### `reviews` - Shop reviews
```sql
-- shop_id, listing_id, buyer_user_id, rating, review, create_timestamp
```

### Useful Queries

```bash
# Open SQLite shell
venv/bin/python -c "import sqlite3; c=sqlite3.connect('etsy_data.db'); c.execute('PRAGMA journal_mode=WAL'); import code; code.interact(local=locals())"
```

Or use sqlite3 directly:
```bash
sqlite3 etsy_data.db
```

```sql
-- Pipeline progress
SELECT
    COUNT(*) as total,
    SUM(download_done) as downloaded,
    SUM(bg_removed) as bg_done,
    SUM(embed_clip_vitb32) as clip32,
    SUM(embed_clip_vitl14) as clip14,
    SUM(embed_dinov2_base) as dino2b,
    SUM(embed_dinov2_large) as dino2l,
    SUM(embed_dinov3_base) as dino3b
FROM image_status;

-- Images ready for cleanup (all 5 models embedded, not primary)
SELECT COUNT(*) FROM image_status
WHERE is_primary = 0
  AND embed_clip_vitb32 = 1
  AND embed_clip_vitl14 = 1
  AND embed_dinov2_base = 1
  AND embed_dinov2_large = 1
  AND embed_dinov3_base = 1;

-- Listings with complete embeddings
SELECT COUNT(DISTINCT listing_id) FROM image_status
WHERE embed_clip_vitb32 = 1
  AND embed_clip_vitl14 = 1
  AND embed_dinov2_base = 1
  AND embed_dinov2_large = 1
  AND embed_dinov3_base = 1;

-- Recent shops
SELECT shop_id, url FROM shops ORDER BY shop_id DESC LIMIT 10;

-- Listings per shop
SELECT shop_id, COUNT(*) as cnt FROM listings GROUP BY shop_id ORDER BY cnt DESC LIMIT 10;
```

---

## 8. Embedding Models

| Model | Dimension | Library | Inputs | FAISS File |
|-------|-----------|---------|--------|------------|
| clip_vitb32 | 512 | open_clip | image + text | `clip_vitb32.faiss`, `clip_vitb32_text.faiss` |
| clip_vitl14 | 768 | open_clip | image + text | `clip_vitl14.faiss`, `clip_vitl14_text.faiss` |
| dinov2_base | 768 | transformers | image only | `dinov2_base.faiss` |
| dinov2_large | 1024 | transformers | image only | `dinov2_large.faiss` |
| dinov3_base | 768 | transformers | image only | `dinov3_base.faiss` |

### Image Index Format
`embeddings/image_index.json` maps FAISS row index to (listing_id, image_id):
```json
[
    [1234567, 111111],
    [1234567, 222222],
    [9876543, 333333],
    ...
]
```

Row 0 in ALL FAISS files corresponds to listing_id=1234567, image_id=111111.

---

## 9. Backup & Cleanup

### Moving Non-Primary Images to HDD
After all 5 models have embedded an image, non-primary images can be moved to the 1TB HDD:

```bash
# Dry run (see what would be moved)
venv/bin/python scripts/cleanup_images.py --dry-run

# Actually move files
venv/bin/python scripts/cleanup_images.py

# Watch mode (continuous)
venv/bin/python scripts/cleanup_images.py --watch
```

Backup location: `/Volumes/HDD_1000/embedded_backup/images/`

### What Gets Kept on SSD
- Primary image (`is_primary=1`) for each listing stays in `images/`
- All FAISS embeddings stay in `embeddings/`
- Database stays on SSD

---

## 10. Troubleshooting

### Database Lock Errors
The scripts use WAL mode and retry logic, but occasional lock errors can still occur:
```
sqlite3.OperationalError: database is locked
```

**Solution**: Scripts have `@_retry_on_lock` decorator with 30 retries and exponential backoff. If a script crashes, pipeline_monitor will restart it.

### Script Won't Start
Check if kill file exists:
```bash
ls -la KILL*
rm KILL KILL_DL KILL_BG KILL_EMBED 2>/dev/null
```

### Embedding Progress Stuck
Embed.py processes models sequentially. Check which model is active:
```bash
tail -50 embed.log
```

### Check Disk Space
```bash
df -h /Users/tzuohannlaw
df -h /Volumes/HDD_1000
```

### Memory Issues
If bg_remover or embed.py runs out of memory:
```bash
# Reduce batch size for embed
venv/bin/python image_search_v2/embed.py --model all --batch-size 4

# bg_remover uses fixed batch of 10000 from SQL, processes one at a time
```

---

## 11. Git Workflow

The codebase is synced via git between MacBook Pro (development) and iMac (production).

### On iMac (pull changes)
```bash
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector
git pull origin main
```

### On MacBook Pro (push changes)
```bash
cd /Users/tzuohann/Documents/Claude/imageselector
git add -A && git commit -m "message" && git push origin main
```

**NEVER** use scp/rsync directly between machines - always use git.

---

## 12. Expected Processing Rates

| Stage | Rate | Hardware |
|-------|------|----------|
| Sync data | ~5 QPS | API limited |
| Image download | ~100/sec | 8 parallel CDN workers |
| BG removal | ~56/min (~3,400/hr) | CoreML GPU |
| Embedding | ~32/batch | MPS GPU, batch-size 8 |

### Time to 100K embeddings per model
- Bottleneck: bg_remover at ~3,400/hour
- ~700K images backlogged
- Estimated: ~8 days to clear backlog
- However, embed.py can process the 277K already bg-removed images

---

## 13. Common Commands Cheat Sheet

```bash
# Go to project
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector

# Check status
venv/bin/python image_db.py

# Start pipeline
nohup venv/bin/python scripts/pipeline_monitor.py > pipeline.log 2>&1 &

# Stop pipeline gracefully
touch KILL

# Watch all logs
tail -f *.log

# Check processes
ps aux | grep python | grep -v grep

# Pull latest code
git pull origin main

# Database shell
sqlite3 etsy_data.db "SELECT COUNT(*) FROM image_status WHERE bg_removed=1;"

# Check embeddings directory
ls -lh embeddings/

# Check images count
ls images/ | wc -l

# Disk usage
du -sh images/ embeddings/ etsy_data.db
```

---

## 14. After 100K Embeddings

Once all 5 models have 100K+ embeddings:

1. **Stop pipeline**: `touch KILL`
2. **Verify counts**: `venv/bin/python image_db.py`
3. **Run cleanup**: `venv/bin/python scripts/cleanup_images.py` (moves non-primary to HDD)
4. **Build search UI**: Implement ensemble search algorithm
5. **Restart pipeline**: Keep collecting more data

### Ensemble Search Algorithm (TODO)
- Query image → embed with all 5 models
- Search each FAISS index for top-K matches
- Aggregate scores by listing_id (max score per listing)
- Rank by consensus (images matching across multiple models are more reliable)
- Display primary image for each matched listing

---

## 15. Documentation

- **`IMAC_OPERATIONS.md`** - This guide (complete operations reference)
- **`CLAUDE.md`** - Detailed codebase overview and next steps
- **`data_structure.md`** - Database schema reference
