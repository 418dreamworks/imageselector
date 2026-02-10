# Etsy Furniture Image Pipeline

## CRITICAL RULES - READ FIRST

**NEVER DO THESE:**
- `pkill` or `kill -9` on any script → Use KILL files instead
- Use system python on iMac → Use `venv/bin/python3`
- Wrong iMac path → Correct: `/Users/tzuohannlaw/Documents/418Dreamworks/imageselector`
- Direct scp/rsync to remote → Use git push/pull instead

**ALWAYS DO THESE:**
- Use KILL files to stop scripts gracefully
- Check `lsof data/db/etsy_data.db` before any kill/SIG commands to verify no DB write conflicts
- Confirm YES:GIT before git push/pull
- Read `docs/IMAC_OPERATIONS.md` for detailed operations guide

---

## Project Overview

This project builds an **ensemble image search system** for Etsy furniture listings. The goal is to find similar furniture items using multiple embedding models, where matches that score well across multiple models are more reliable.

### Architecture

```
sync_data.py ──► image_downloader.py ──► embed_orchestrator.py (bg removal + embedding)
     │                   │                         │
     ▼                   ▼                         ▼
  Etsy API          Etsy CDN                 3 workers (iMac, MBP, Sleight)
  (~5 QPS)        (8 workers)              bg removal + 5 model embeddings
     │                   │                         │
     ▼                   ▼                         ▼
  data/db/         images/                   data/embeddings/
  etsy_data.db     imagedownload/            FAISS indexes
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

### Core Scripts (bin/)

| File | Purpose |
|------|---------|
| `bin/sync_data.py` | Crawls Etsy API taxonomy, discovers listings, fetches shop/review data |
| `bin/image_downloader.py` | Downloads images from Etsy CDN (8 parallel workers) |
| `bin/image_db.py` | Shared database helpers with `@_retry_on_lock` decorator |
| `bin/tar_images.py` | Archives 10K image batches from imageall_new → imageall_tars |
| `bin/embedding/embed_orchestrator.py` | Distributes batches to 3 workers, imports results to FAISS |
| `bin/embedding/embed_worker.py` | BG removal + 5-model embedding (runs on workers) |
| `bin/embedding/models.py` | Model definitions and lazy loading for CLIP/DINO |

### Maintenance Scripts (bin/)

| File | Purpose |
|------|---------|
| `bin/update_primary.py` | Extracts primary images from tar archives to imageprimary/ |
| `bin/backup_db.py` | Backs up entire data/ folder to HDD1TB |
| `bin/init/fetch_taxonomy.py` | One-time taxonomy bootstrap |
| `bin/init/calculate_price_breaks.py` | One-time price break calculation |

### Kill Files (Graceful Shutdown)

| Kill File | Stops |
|-----------|-------|
| `KILL_SD` | sync_data.py |
| `KILL_DL` | image_downloader.py |
| `KILL_ORCH` | embed_orchestrator.py |

---

## Data Layout

```
data/
├── db/
│   ├── etsy_data.db                 # Main SQLite database
│   ├── qps_config.json              # API rate config
│   ├── furniture_taxonomy_config.json
│   ├── furniture_taxonomy_ids.json
│   └── furniture_leaf_taxonomy_ids.json
└── embeddings/
    ├── clip_vitb32.faiss            # Image embeddings (N, 512)
    ├── clip_vitb32_text.faiss       # Text embeddings (N, 512)
    ├── clip_vitl14.faiss            # Image embeddings (N, 768)
    ├── clip_vitl14_text.faiss       # Text embeddings (N, 768)
    ├── dinov2_base.faiss            # Image embeddings (N, 768)
    ├── dinov2_large.faiss           # Image embeddings (N, 1024)
    ├── dinov3_base.faiss            # Image embeddings (N, 768)
    └── image_index.json             # Row → (listing_id, image_id)
```

**Row alignment**: Row 42 in ALL FAISS files = same image. `image_index.json[42]` = `[listing_id, image_id]`

---

## Image Layout

```
images/
├── imagedownload/    # Raw downloaded images (primary SSD)
├── imageall_new/     # BG-removed images waiting to be tarred
├── imageall_tars/    # Symlink → HDD1TB/images/imageall_tars
└── imageprimary/     # One bg-removed primary image per listing
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

## Storage & Backup

| Location | Contents | Update Frequency |
|----------|----------|-----------------|
| Primary SSD | Everything (data/, images/, bin/) | Live |
| SSD500GB | data/ + images/imageprimary/ | Hourly rsync (server-ready) |
| HDD500GB | Mirror of SSD500GB | Nightly rsync |
| HDD1TB | images/imageall_tars/ + backups/ | Live (tars) + 6h (backups) |
| HDD3TB | Mirror of HDD1TB | Nightly rsync |

### Cron Schedule

| Time | Job |
|------|-----|
| `:00` every 12h | tar_images.py && update_primary.py |
| `:15` hourly | rsync data/ + imageprimary/ → SSD500GB |
| `:30` every 6h | backup_db.py (data/ → HDD1TB) |
| 4am nightly | HDD1TB → HDD3TB |
| 6am nightly | SSD500GB → HDD500GB |

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

# Watch logs
tail -f logs/*.log

# Check running processes
ps aux | grep python | grep -v grep
```

---

## Documentation

- **`docs/IMAC_OPERATIONS.md`** - Complete operations guide
- **`docs/data_structure.md`** - Full database schema reference
