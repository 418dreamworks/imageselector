# Database Schema

SQLite database: `etsy_data.db` (WAL mode for concurrent access)

---

## Core Tables

### `image_status` - Image Pipeline Tracking

Tracks every image through the download → bg_removal → embedding pipeline.

| Column | Type | Description |
|--------|------|-------------|
| `listing_id` | INTEGER | Listing ID (PK with image_id) |
| `image_id` | INTEGER | Image ID (PK with listing_id) |
| `shop_id` | INTEGER | Shop ID |
| `hex` | TEXT | CDN hex path component |
| `suffix` | TEXT | CDN suffix component |
| `is_primary` | INTEGER | 1 if primary image for listing |
| `when_made` | TEXT | "2020_2024", "before_2000", etc. |
| `price` | REAL | Listing price |
| `to_download` | INTEGER | 1 if queued for download (default 1) |
| `download_done` | INTEGER | 1 if downloaded (default 0) |
| `bg_removed` | INTEGER | 1 if background removed (default 0) |
| `embed_clip_vitb32` | INTEGER | 1 if embedded with CLIP ViT-B/32 |
| `embed_clip_vitl14` | INTEGER | 1 if embedded with CLIP ViT-L/14 |
| `embed_dinov2_base` | INTEGER | 1 if embedded with DINOv2 base |
| `embed_dinov2_large` | INTEGER | 1 if embedded with DINOv2 large |
| `embed_dinov3_base` | INTEGER | 1 if embedded with DINOv3 base |

**CDN URL reconstruction:**
```
https://i.etsystatic.com/il/{hex}/{image_id}/il_570xN.{image_id}_{suffix}.jpg
```

---

### `shops` - Static Shop Data

Written once when shop is first discovered.

| Column | Type | Description |
|--------|------|-------------|
| `shop_id` | INTEGER | Primary key |
| `snapshot_timestamp` | TEXT | When first synced |
| `create_date` | INTEGER | Unix timestamp of shop creation |
| `url` | TEXT | Shop URL |
| `is_shop_us_based` | INTEGER | 1 if US-based |
| `shipping_from_country_iso` | TEXT | Country code |
| `shop_location_country_iso` | TEXT | Country code |

### `shops_dynamic` - Shop Metrics Over Time

Appended each sync (every 2 weeks).

| Column | Type | Description |
|--------|------|-------------|
| `shop_id` | INTEGER | Foreign key to shops |
| `snapshot_timestamp` | TEXT | When synced |
| `update_date` | INTEGER | Last update timestamp |
| `listing_active_count` | INTEGER | Active listings |
| `accepts_custom_requests` | INTEGER | 1 if accepts custom |
| `num_favorers` | INTEGER | Favorites count |
| `transaction_sold_count` | INTEGER | Total sales |
| `review_average` | REAL | Average rating (1-5) |
| `review_count` | INTEGER | Total reviews |

---

### `listings` - Static Listing Data

Written once when listing is first discovered.

| Column | Type | Description |
|--------|------|-------------|
| `listing_id` | INTEGER | Primary key |
| `snapshot_timestamp` | TEXT | When first synced |
| `shop_id` | INTEGER | Foreign key to shops |
| `title` | TEXT | Listing title |
| `description` | TEXT | Full description |
| `creation_timestamp` | INTEGER | When created |
| `url` | TEXT | Listing URL |
| `is_customizable` | INTEGER | 1 if customizable |
| `is_personalizable` | INTEGER | 1 if personalizable |
| `listing_type` | TEXT | physical, digital, etc. |
| `tags` | TEXT | JSON array of tags |
| `materials` | TEXT | JSON array of materials |
| `processing_min` | INTEGER | Min processing days |
| `processing_max` | INTEGER | Max processing days |
| `who_made` | TEXT | i_did, collective, someone_else |
| `when_made` | TEXT | made_to_order, 2020_2024, etc. |
| `item_weight` | REAL | Weight |
| `item_weight_unit` | TEXT | oz, lb, g, kg |
| `item_length` | REAL | Length |
| `item_width` | REAL | Width |
| `item_height` | REAL | Height |
| `item_dimensions_unit` | TEXT | in, ft, mm, cm, m |
| `should_auto_renew` | INTEGER | Auto-renew |
| `language` | TEXT | Listing language |
| `price_amount` | INTEGER | Price amount |
| `price_divisor` | INTEGER | Price divisor |
| `price_currency` | TEXT | Currency code |
| `taxonomy_id` | INTEGER | Category ID |
| `production_partners` | TEXT | JSON array |

### `listings_dynamic` - Listing Metrics Over Time

Appended each sync.

| Column | Type | Description |
|--------|------|-------------|
| `listing_id` | INTEGER | Foreign key to listings |
| `snapshot_timestamp` | TEXT | When synced |
| `state` | TEXT | active, inactive, etc. |
| `ending_timestamp` | INTEGER | Expiration |
| `quantity` | INTEGER | Available quantity |
| `num_favorers` | INTEGER | Favorites count |
| `views` | INTEGER | View count |
| `price_amount` | INTEGER | Current price |
| `price_divisor` | INTEGER | Price divisor |
| `price_currency` | TEXT | Currency |

---

### `reviews` - Shop Reviews

Append-only. Fetches only new reviews since last sync.

| Column | Type | Description |
|--------|------|-------------|
| `snapshot_timestamp` | TEXT | When synced |
| `shop_id` | INTEGER | Shop ID |
| `listing_id` | INTEGER | Listing reviewed |
| `buyer_user_id` | INTEGER | Reviewer's user ID |
| `rating` | INTEGER | 1-5 stars |
| `review` | TEXT | Review text |
| `language` | TEXT | Language code |
| `create_timestamp` | INTEGER | When review was created |

---

### `sync_state` - Crawler State

Key-value store for sync progress.

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT | State key |
| `value` | TEXT | JSON value |

---

## Embedding Files

Located in `embeddings/`:

| File | Shape | Description |
|------|-------|-------------|
| `clip_vitb32.faiss` | (N, 512) | CLIP ViT-B/32 image embeddings |
| `clip_vitb32_text.faiss` | (N, 512) | CLIP ViT-B/32 text embeddings |
| `clip_vitl14.faiss` | (N, 768) | CLIP ViT-L/14 image embeddings |
| `clip_vitl14_text.faiss` | (N, 768) | CLIP ViT-L/14 text embeddings |
| `dinov2_base.faiss` | (N, 768) | DINOv2 base embeddings |
| `dinov2_large.faiss` | (N, 1024) | DINOv2 large embeddings |
| `dinov3_base.faiss` | (N, 768) | DINOv3 base embeddings |
| `image_index.json` | - | Maps row index → (listing_id, image_id) |

**Row alignment:** Row 42 in ALL FAISS files corresponds to the same (listing_id, image_id) pair in image_index.json.

---

## Runtime Files (gitignored)

| Path | Description |
|------|-------------|
| `etsy_data.db` | Main SQLite database |
| `etsy_data.db-wal` | WAL file (don't delete while running) |
| `etsy_data.db-shm` | Shared memory file |
| `images/` | Downloaded images (~1M files) |
| `embeddings/` | FAISS index files |
| `qps_config.json` | API rate limit config |
| `*.log` | Script logs |
