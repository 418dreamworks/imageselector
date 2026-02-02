# iMac Production Files

Files that must be present on the iMac at `~/Documents/418Dreamworks/imageselector/`.

## Core Scripts

| File | Purpose |
|------|---------|
| `sync_images.py` | Main image crawler (run continuously) |
| `sync_data.py` | Shop/listing/review data collector |

## Configuration

| File | Purpose |
|------|---------|
| `.env` | API key (`ETSY_API_KEY`) |
| `.env.example` | Template for `.env` |
| `furniture_taxonomy_config.json` | Price-range intervals per taxonomy ID |
| `furniture_taxonomy_ids.json` | Parent furniture taxonomy IDs |
| `furniture_leaf_taxonomy_ids.json` | Leaf furniture taxonomy IDs |
| `requirements.txt` | Python dependencies |

## Runtime Data

| File | Purpose |
|------|---------|
| `image_metadata.json` | Listing metadata (shop_id, image_id, hex, suffix) |
| `sync_progress.json` | Crawler progress (completed intervals) |
| `etsy_data.db` | SQLite database (shops, listings, reviews) |
| `images/` | Downloaded listing images (~65k files) |

## Support Scripts (not run continuously)

| File | Purpose |
|------|---------|
| `calculate_price_breaks.py` | Generate price-range intervals |
| `count_listings.py` | Count listings per taxonomy ID |
| `fetch_taxonomy.py` | Fetch Etsy taxonomy tree |
| `find_price_breaks.py` | Find optimal price breaks |

## Other

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project documentation |
| `.gitignore` | Git ignore rules |
| `.git/` | Git repository |
| `venv/` | Python virtual environment |

## Running Rules

- **MAX 5 QPS total** — Combined API queries across ALL running scripts must never exceed 5 per second.
- Only one script should use the API at a time to stay within the limit.
- **Deploy via git only** — Always `git push` locally then `git pull` on iMac. Never use `scp` to copy files directly.

## External Disks

| Device | Name | Size | Purpose |
|--------|------|------|---------|
| disk6 | SSD_120 | 120 GB | Project data (fast access) |
| disk4 | HDD_320 | 320 GB | Project data (bulk storage) |
| disk5 | HDD_1000 | 1 TB | Backup (rsync mirror of SSD_120 + HDD_320) |

**Backup strategy**: HDD_1000 is an rsync backup of the other two disks. Run periodically:
```bash
rsync -av /Volumes/SSD_120/ /Volumes/HDD_1000/SSD_120_backup/
rsync -av /Volumes/HDD_320/ /Volumes/HDD_1000/HDD_320_backup/
```

**Disk health**: Run full read checks periodically to detect bad sectors:
```bash
sudo dd if=/dev/disk6 of=/dev/null bs=1m status=progress  # SSD_120
sudo dd if=/dev/disk4 of=/dev/null bs=1m status=progress  # HDD_320
sudo dd if=/dev/disk5 of=/dev/null bs=1m status=progress  # HDD_1000
```

## Can Be Cleaned Up

| File | Notes |
|------|-------|
| `image_metadata_backup_*.json` | Old backups (keep most recent only) |
| `backup_image_metadata_*.json.gz` | Failed/empty backup |
| `sync_output.log` | Grows over time, can truncate |
| `image_metadata_test.json` | Test data |
| `images_test/` | Test images |
| `sync_progress_test.json` | Test progress |
| `.DS_Store` | macOS artifact |
| `fix_metadata.py` | Deprecated, no longer needed |
