# Etsy Furniture Tracker

## IMPORTANT: API Rate Limits (Standing Instruction)

**This account has:**
- **100,000 requests/day** (rolling 24h window)
- **150 requests/second** (QPS)

**Allocation:**
- `sync_images.py` uses **90,000/day at 95 QPS**
- Development/testing uses remaining **10,000/day at 5 QPS max**

**When developing:**
- Always throttle API calls to ~5 QPS (0.2s delay between calls)
- Check `x-remaining-today` header before bulk operations
- Never run sync and heavy development simultaneously

**Check remaining quota:**
```python
response.headers.get("x-remaining-today")
```

---

## Project Goal
Track Etsy furniture listings over time to learn about them. Collect listing data, images, shop info, and reviews at intervals.

## Universe
- **Taxonomy ID 967** (Furniture) under Home & Living (891)
- ~535,000 listings
- Max 3 levels deep in taxonomy

## Etsy API Reference

### API Calls Per Listing
| Data | Endpoint | Calls |
|------|----------|-------|
| Listing details | `/listings/{id}` | 1 |
| Images | `/listings/{id}/images` | 1 |
| Shop info | `/shops/{shop_id}` | 1 (reuse per shop) |
| Shop reviews | `/shops/{shop_id}/reviews` | ceil(reviews/100) |

### Listing Fields
**Static:** `listing_id`, `shop_id`, `title`, `description`, `tags`, `taxonomy_id`, `who_made`, `when_made`, `original_creation_timestamp`

**Dynamic (track over time):** `views`, `num_favorers`, `quantity`, `price`, `state`, `updated_timestamp`

### Shop Fields
**Static:** `shop_id`, `shop_name`, `create_date`, `shipping_from_country_iso`

**Dynamic:** `review_average`, `review_count`, `transaction_sold_count`, `num_favorers`, `listing_active_count`

### Reviews
- Shop-level, not listing-level
- Each review has `listing_id` to associate back
- Fields: `rating` (1-5), `review` (text), `image_url_fullxfull`

### Search/Filter
- `taxonomy_id` parameter to filter by category
- `sort_on`: `score`, `created`, `updated`, `price` (no sort by views/favorers)
- Max 100 results per call, use `offset` for pagination

### Images
- Use `url_170x135` for thumbnails
- Track `listing_image_id` to detect changes without re-downloading
- Image downloads from CDN (`i.etsystatic.com`) - not counted as API calls
- Throttle CDN requests (~0.2-0.5s delay)

## Files
- `etsy_taxonomy.json` - Full Etsy category taxonomy
- `.env` - Contains `ETSY_API_KEY`
- `corpus/` - Downloaded images and metadata

## Environment
```bash
source venv/bin/activate
```

## TODO

### Image Sync Script (`sync_images.py`)
Single script for initial download and updates. Run every ~6 months.

**Logic:**
1. Get all listing IDs under taxonomy 967 (Furniture)
2. For each listing ID:
   - Check if image exists in database (by `listing_id`)
   - If not: download first image (`url_170x135`), store with `listing_id` and `listing_image_id`
   - If exists: compare stored `listing_image_id` with API response
     - If different: download new image, update record
     - If same: skip
3. Throttle CDN requests (~0.2-0.5s delay)

**Storage:**
- Images stored as files, retrievable by `listing_id`
- Metadata (e.g., `listing_image_id`) in JSON or SQLite for change detection

**Estimated:**
- ~535,000 listings
- ~5 GB initial download (10 KB per thumbnail)
- Subsequent runs: only new/changed images
