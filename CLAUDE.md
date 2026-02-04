# Etsy Furniture Image Pipeline

## CRITICAL RULES - READ FIRST

**NEVER DO THESE:**
- ❌ `pkill` or `kill -9` on any script → Use KILL files instead
- ❌ Use system python on iMac → Use `venv/bin/python`
- ❌ Wrong iMac path → Correct: `/Users/tzuohannlaw/Documents/418Dreamworks/imageselector`
- ❌ Direct scp/rsync to remote → Use git push/pull instead

**ALWAYS DO THESE:**
- ✅ Use KILL files to stop scripts gracefully
- ✅ Confirm YES:GIT before git push/pull
- ✅ Read `IMAC_OPERATIONS.md` for detailed operations guide

---

## Project Overview

This project builds an **ensemble image search system** for Etsy furniture listings. The goal is to find similar furniture items using multiple embedding models, where matches that score well across multiple models are more reliable.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   sync_data.py ──► image_downloader.py ──► bg_remover.py ──► embed.py     │
│        │                   │                    │                │          │
│        ▼                   ▼                    ▼                ▼          │
│   Etsy API            Etsy CDN              CoreML GPU        MPS GPU      │
│   (~5 QPS)          (8 workers)            (~56/min)       (5 models)      │
│        │                   │                    │                │          │
│        ▼                   ▼                    ▼                ▼          │
│   SQLite DB           images/               images/          FAISS         │
│   (listings,        (945K files)          (overwrites)      indexes        │
│   shops, reviews)                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Image-level search**: Search matches individual images, not listings
2. **Listing deduplication**: Multiple images from same listing → keep best score
3. **Primary image display**: Always show `is_primary=1` image in results
4. **Ensemble scoring**: Matches appearing in multiple models are more reliable
5. **SQLite for state**: `image_status` table tracks each image through pipeline
6. **FAISS for search**: Separate index file per model (different dimensions)
7. **Shared image_index.json**: Row N in ALL FAISS files = same (listing_id, image_id)

---

## Codebase Structure

### Core Scripts

| File | Purpose |
|------|---------|
| `sync_data.py` | Crawls Etsy API taxonomy, discovers listings, fetches shop/review data |
| `scripts/image_downloader.py` | Downloads images from Etsy CDN (8 parallel workers) |
| `scripts/bg_remover.py` | Removes backgrounds using rembg + CoreML GPU acceleration |
| `image_search_v2/embed.py` | Generates embeddings for 5 models, saves to FAISS indexes |
| `scripts/pipeline_monitor.py` | Manages all 4 scripts, auto-restarts on crash |
| `scripts/cleanup_images.py` | Moves non-primary images to HDD after embedding complete |

### Support Modules

| File | Purpose |
|------|---------|
| `image_db.py` | Shared database helpers with `@_retry_on_lock` decorator |
| `image_search_v2/models.py` | Model definitions and lazy loading for CLIP/DINO |
| `image_search_v2/search.py` | Search functions (to be implemented) |

### Kill Files (Graceful Shutdown)

| Kill File | Stops |
|-----------|-------|
| `KILL` | sync_data.py and pipeline_monitor.py |
| `KILL_DL` | image_downloader.py |
| `KILL_BG` | bg_remover.py |
| `KILL_EMBED` | embed.py |

---

## Database: image_status Table

The `image_status` table is the **single source of truth** for pipeline state:

```sql
CREATE TABLE image_status (
    listing_id INTEGER,
    image_id INTEGER,
    shop_id INTEGER,
    hex TEXT,                    -- CDN path component
    suffix TEXT,                 -- CDN suffix
    is_primary INTEGER,          -- 1 = main listing image
    when_made TEXT,              -- "2020_2024", etc.
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

Each script atomically updates its own flag:
- `sync_data.py` → sets `to_download=1`
- `image_downloader.py` → sets `download_done=1`
- `bg_remover.py` → sets `bg_removed=1`
- `embed.py` → sets `embed_{model}=1` for each model

---

## Embedding Models (5 Total)

| Model | Dimension | Library | Inputs | Notes |
|-------|-----------|---------|--------|-------|
| clip_vitb32 | 512 | open_clip | image + text | Fast, good general-purpose |
| clip_vitl14 | 768 | open_clip | image + text | Higher quality than B/32 |
| dinov2_base | 768 | transformers | image only | Good for visual similarity |
| dinov2_large | 1024 | transformers | image only | Best visual quality |
| dinov3_base | 768 | transformers | image only | Latest DINO version |

**CLIP models** also generate text embeddings from listing title + materials.

### FAISS Index Files

```
embeddings/
├── clip_vitb32.faiss          # Image embeddings (N, 512)
├── clip_vitb32_text.faiss     # Text embeddings (N, 512)
├── clip_vitl14.faiss          # Image embeddings (N, 768)
├── clip_vitl14_text.faiss     # Text embeddings (N, 768)
├── dinov2_base.faiss          # Image embeddings (N, 768)
├── dinov2_large.faiss         # Image embeddings (N, 1024)
├── dinov3_base.faiss          # Image embeddings (N, 768)
└── image_index.json           # Row → (listing_id, image_id)
```

**Row alignment**: Row 42 in ALL files = same image. `image_index.json[42]` = `[listing_id, image_id]`

---

## Current Status

As of last session:

| Metric | Count |
|--------|-------|
| Total images | ~1.2M |
| Download done | ~945K |
| BG removed | ~277K |
| Embedded clip_vitb32 | ~80K |
| Other models | 0 (processed sequentially) |

### Processing Rates

| Stage | Rate |
|-------|------|
| Etsy API | ~5 QPS (rate limited) |
| CDN download | ~100/sec (8 workers) |
| BG removal | ~56/min (~3,400/hr) with CoreML GPU |
| Embedding | ~32 images/batch with MPS GPU |

### Bottleneck

BG removal is the bottleneck. With ~700K backlog at 3,400/hr, clearing takes ~8 days.

---

## Next Steps

### Phase 1: Reach 100K Embeddings (Current)

1. **Keep pipeline running** until all 5 models have 100K+ embeddings
2. **Monitor progress**: `venv/bin/python image_db.py`
3. **Check logs**: `tail -f *.log`

### Phase 2: Implement Ensemble Search

Once 100K embeddings exist:

1. **Build search function** in `image_search_v2/search.py`:
   - Load all 5 FAISS indexes
   - Query image → embed with all 5 models
   - Search each index for top-K (e.g., K=100)
   - Aggregate by listing_id, keeping max score per listing
   - Rank by ensemble consensus

2. **Scoring approaches** to experiment with:
   - Simple: Sum of ranks across models
   - Weighted: Weight by model quality
   - Consensus: Count how many models have listing in top-K
   - Hybrid: Combination of above

3. **Display**: Show primary image (`is_primary=1`) for each matched listing

### Phase 3: Build UI

1. **Streamlit app** in `image_search_v2/app.py`
2. Upload query image → show top matches
3. Tabs for each model's results vs ensemble
4. Highlight matches that appear in multiple models

### Phase 4: Cleanup & Scale

1. **Run cleanup_images.py** to move non-primary images to HDD
2. **Keep pipeline running** to accumulate more data
3. **Periodic re-embedding** as models improve

---

## Quick Commands

```bash
# Go to project
cd /Users/tzuohannlaw/Documents/418Dreamworks/imageselector

# Check status
venv/bin/python image_db.py

# Start pipeline (manages all 4 scripts)
nohup venv/bin/python scripts/pipeline_monitor.py > pipeline.log 2>&1 &

# Stop pipeline gracefully
touch KILL

# Watch logs
tail -f *.log

# Check running processes
ps aux | grep python | grep -v grep

# Pull code changes from MacBook Pro
git pull origin main
```

---

## Documentation

- **`IMAC_OPERATIONS.md`** - Complete operations guide (start/stop, troubleshooting, SQL queries)
- **`data_structure.md`** - Full database schema reference
