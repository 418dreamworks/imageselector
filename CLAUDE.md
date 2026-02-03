# Etsy Furniture Image Sync

## CRITICAL RULES - READ FIRST

**NEVER DO THESE:**
- ❌ `pkill` or `kill -9` on any script → Use KILL files instead (see below)
- ❌ Use system python on iMac → Use `venv/bin/python`
- ❌ Wrong iMac path → Correct: `/Users/tzuohannlaw/Documents/418Dreamworks/imageselector`
- ❌ Direct scp/rsync to remote → Use git push/pull instead

**ALWAYS DO THESE:**
- ✅ Read this file after every context compaction
- ✅ Use KILL files to stop scripts gracefully (see Kill Files section)
- ✅ Confirm YES:RS before running remote scripts
- ✅ Confirm YES:GIT before git push/pull

## Required Reading After Compaction

**IMPORTANT**: After any context compaction, read these files BEFORE doing anything else:
- `IMAC_FILES.md` - iMac setup, venv location, file locations
- `data_structure.md` - DB schema and field definitions
- `imagesearch.md` - Image search implementation plan
- `general_rules.md` - Points to parent directory rules

## Remote Server (iMac)

```bash
ssh imac
# or: ssh tzuohannlaw@192.168.68.200
```

- **Path**: `/Users/tzuohannlaw/Documents/418Dreamworks/imageselector`
- **Python**: `venv/bin/python` (not system python)
- No password required from this machine

## Project State

Two main scripts share the Etsy API quota:

1. **sync_images.py** - Downloads furniture listing images from Etsy CDN
   - Run continuously in slow mode on remote server
   - Uses API to discover listings, CDN to download images
   - Stores images + metadata (shop_id, listing_id, image URLs)
   - One major flaw of this program right now is that it grabs listings 1 at a time. We should change it so the crawler grabs lis

2. **sync_data.py** - Collects shop/listing/review data to SQLite
   - Run after image sync completes (shares API quota)
   - Reads `image_metadata.json` to know which shops to sync
   - Skips shops synced within last 14 days
   - This should be retired. For this main design, take elements from this that get the shop static and dynamic data and integrate into sync_images. The reason is we can't get new shops, so we are stuck getting new listings, and linking them back to shops which means using the crawler.

## Merge sync_data with sync_images. The structure is largely sync_images, but we will call the final product sync_data.
# High level process
1. Crawler assumes that all items in the meta_data have a corresponding image file and all image files have a corresponding meta_data entry. This is true on imac where this script is actually run. On the macbookpro where the script is developed, this is not the case but it is alright.
2. We are only interested in listings where when_made >= 2000. A listing with when_made < 2000 continues to stay in meta_data, but no further action is taken other than an entry in meta_data following the format that is in meta_data. This means that the image does not get downloaded, and the database does not get populated.
3. A listing that has when_made >= 2000 can either be new or existing. 
   1. If it is new, an entry is made for it in the database under static and dynamic. If the shop for the listing is already in the database, no further action. If the shop is not in the database, an entry for shop is made as well under static and dynamic.
   2. If it is existing, update the dynamic entry if the dynamic entry is greater than 1 week old. Same for the shop. If a shop has multiple listings, the first listing would update this shop and the rest would skip since the entry would be fresh.
4. When the crawler has exhausted all the taxonomy categories, do a batch fetch for any remaining listing that are somehow missed in the taxonomy crawl and apply (3). These would be listings that are in the metadata that did not show up in the taxonomy crawl. This should finish up shops as well.
5. Check that the dataset is fully in sync.
6. Loop through all the shops and pull in all the reviews at the shop level. Filter the reviews so that only reviews that were created after the last datasnapshot are given. If there are no reviews, ask for everything from year 2000. Loop as many offsets as needed.

## Keep things very clean for now. No flags. The one rule we need to stick to is 5QPS max, and 1 download worker. That's all we need. I suspect this will run very quickly. We test this locally. I want to completely go through 2 taxonomy categories here on this computer before going into production on the imac.

## Kill Files - Graceful Script Shutdown

Each script has its own kill file. Create the file to stop the script gracefully:

| Script | Kill File | Command (iMac) |
|--------|-----------|----------------|
| sync_data.py | `KILL` | `touch KILL` |
| bg_remover.py | `KILL_BG` | `touch KILL_BG` |
| embed.py | `KILL_EMBED` | `touch KILL_EMBED` |

Full paths on iMac:
```bash
touch /Users/tzuohannlaw/Documents/418Dreamworks/imageselector/KILL        # sync_data
touch /Users/tzuohannlaw/Documents/418Dreamworks/imageselector/KILL_BG     # bg_remover
touch /Users/tzuohannlaw/Documents/418Dreamworks/imageselector/KILL_EMBED  # embed
```

Each script checks for its kill file at each loop iteration and exits cleanly, saving all state first.

**NEVER use `pkill` or `kill -9`** - always use kill files instead.