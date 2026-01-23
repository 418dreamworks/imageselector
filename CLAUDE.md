# Etsy Furniture Image Sync

## Remote Server

This code runs on a remote computer accessible via:
```bash
ssh imac
# or: ssh tzuohannlaw@192.168.68.106
```
No password or passphrase required from this machine.

## Project State

Two main scripts share the Etsy API quota:

1. **sync_images.py** - Downloads furniture listing images from Etsy CDN
   - Run continuously in slow mode on remote server
   - Uses API to discover listings, CDN to download images
   - Stores images + metadata (shop_id, listing_id, image URLs)

2. **sync_data.py** - Collects shop/listing/review data to SQLite
   - Run after image sync completes (shares API quota)
   - Reads `image_metadata.json` to know which shops to sync
   - Skips shops synced within last 14 days

## Algorithm (sync_images.py)

```
(A) Every 30 days, re-sync all known shops from metadata
(B) For each shop: get all furniture listings, queue new images
(C) Crawl each (taxonomy_id, min_price, max_price) interval from furniture_taxonomy_config.json
(D) Background workers download queued images from CDN
```

## Usage

### Image Sync (sync_images.py)

```bash
# Production: run with caffeinate to prevent Mac sleep
cd ~/Documents/418Dreamworks/imageselector
nohup caffeinate -i ./venv/bin/python -u sync_images.py --slow > sync_output.log 2>&1 &

# Fast mode: use when API quota is available and need quick sync
nohup caffeinate -i ./venv/bin/python -u sync_images.py --fast > sync_output.log 2>&1 &

# Other options
./venv/bin/python sync_images.py --reset-shops  # Clear synced_shops list, then run
./venv/bin/python sync_images.py --test         # Use test folders instead of production
./venv/bin/python sync_images.py --limit N      # Limit to N listings (for testing)
./venv/bin/python sync_images.py ids.txt        # Sync specific listing IDs from file
```

**Important:** Use `caffeinate -i` to prevent Mac from sleeping while running.

**Recommendation:** Run `--slow` (default) for daily operation. Only use `--fast` when you need to catch up or rebuild data quickly.

### Data Sync (sync_data.py)

```bash
# Initial collection: run fast to get all data quickly
./venv/bin/python sync_data.py --fast         # ~90 QPS, sync all shops once

# Ongoing: run continuously in slow mode (with caffeinate)
nohup caffeinate -i ./venv/bin/python -u sync_data.py --continuous > data_sync.log 2>&1 &

# Other options
python sync_data.py --slow         # ~1 QPS, one-shot sync
python sync_data.py --top N        # Only sync top N shops by listing count
python sync_data.py --test         # Use test database
```

**Workflow:**
1. First run `--fast` to collect all data quickly (uses full API quota)
2. Then run `--continuous` which runs forever in slow mode (~1 QPS), checking for shops that need updating (>14 days since last sync)

**Sync Logic:**
- New shop (not in DB): fetch all data, insert static + dynamic
- Existing shop (last sync > 14 days): fetch all data, insert dynamic only
- Existing shop (last sync < 14 days): skip entirely (no API call)

**Output:** SQLite database `etsy_data.db` with tables:
- `shops` - Static shop data (written once)
- `shops_dynamic` - Dynamic shop data (appended each sync)
- `listings` - Static listing data (written once)
- `listings_dynamic` - Dynamic listing data (appended each sync)
- `reviews` - Reviews (append-only, incremental fetch)
- `sync_state` - Tracks last sync times per shop

## Data Collection

### Data Structure

Three datasets, all append-only with `snapshot_timestamp`:

1. **Shops** - All selected fields + `snapshot_timestamp`. Append new row each sync (every 2 weeks for dynamic fields like `num_favorers`, `transaction_sold_count`; full refresh every 6 months for static fields).

2. **Listings** - All selected fields + `snapshot_timestamp`. Append new row each sync (every 2 weeks for `num_favorers`, `views`; full refresh every 6 months for static fields).

3. **Reviews** - All selected fields, append-only. Only fetch new reviews since last sync (by `create_timestamp`). Derive `last_review_timestamp` per shop via `max(create_timestamp) group by shop_id`.

Forward-fill when joining data for analysis.

### Selected Shop Fields

| Field | Type | Description |
|-------|------|-------------|
| `shop_id` | int | Unique shop identifier |
| `create_date` | int | Unix timestamp of shop creation |
| `update_date` | int | Unix timestamp of last update |
| `listing_active_count` | int | Number of active listings |
| `accepts_custom_requests` | bool | Whether shop accepts custom orders |
| `url` | string | Full URL to shop page |
| `num_favorers` | int | Number of users who favorited shop |
| `is_shop_us_based` | bool | Shop is US-based |
| `transaction_sold_count` | int | Total number of sales |
| `shipping_from_country_iso` | string | Country code where items ship from |
| `shop_location_country_iso` | string | Country code where shop is located |
| `review_average` | float | Average review rating (1-5) |
| `review_count` | int | Total number of reviews |

### Available Shop Fields

| Field | Type | Description |
|-------|------|-------------|
| `shop_id` | int | Unique shop identifier |
| `shop_name` | string | Shop's URL-friendly name |
| `user_id` | int | Owner's user ID |
| `create_date` | int | Unix timestamp of shop creation |
| `created_timestamp` | int | Same as create_date |
| `title` | string | Shop title displayed at top of pages |
| `announcement` | string | Message displayed on shop homepage |
| `currency_code` | string | Shop's default currency (e.g., "USD", "MAD") |
| `is_vacation` | bool | Whether shop is on vacation mode |
| `vacation_message` | string | Message shown when on vacation |
| `sale_message` | string | Message sent to buyers after purchase |
| `digital_sale_message` | string | Message for digital item purchases |
| `update_date` | int | Unix timestamp of last update |
| `updated_timestamp` | int | Same as update_date |
| `listing_active_count` | int | Number of active listings |
| `digital_listing_count` | int | Number of digital listings |
| `login_name` | string | Internal login identifier |
| `accepts_custom_requests` | bool | Whether shop accepts custom orders |
| `vacation_autoreply` | string | Auto-reply message during vacation |
| `url` | string | Full URL to shop page |
| `image_url_760x100` | string | Shop banner image URL |
| `num_favorers` | int | Number of users who favorited shop |
| `languages` | array | Languages shop supports |
| `icon_url_fullxfull` | string | Shop icon/logo URL |
| `is_using_structured_policies` | bool | Using Etsy's structured policies |
| `has_onboarded_structured_policies` | bool | Onboarded to structured policies |
| `include_dispute_form_link` | bool | Shows dispute form link |
| `is_direct_checkout_onboarded` | bool | Direct checkout enabled |
| `is_etsy_payments_onboarded` | bool | Etsy Payments enabled |
| `is_opted_in_to_buyer_promise` | bool | Opted into buyer promise program |
| `is_calculated_eligible` | bool | Eligible for calculated shipping |
| `is_shop_us_based` | bool | Shop is US-based |
| `transaction_sold_count` | int | Total number of sales |
| `shipping_from_country_iso` | string | Country code where items ship from |
| `shop_location_country_iso` | string | Country code where shop is located |
| `policy_welcome` | string | Welcome policy text |
| `policy_payment` | string | Payment policy text |
| `policy_shipping` | string | Shipping policy text |
| `policy_refunds` | string | Refund policy text |
| `policy_additional` | string | Additional policy text |
| `policy_seller_info` | string | Seller info policy |
| `policy_update_date` | int | When policies were last updated |
| `policy_has_private_receipt_info` | bool | Has private receipt info |
| `has_unstructured_policies` | bool | Has old-style policies |
| `policy_privacy` | string | Privacy policy text |
| `review_average` | float | Average review rating (1-5) |
| `review_count` | int | Total number of reviews |

### Selected Listing Fields

| Field | Type | Description |
|-------|------|-------------|
| `listing_id` | int | Unique listing identifier |
| `shop_id` | int | Shop identifier |
| `title` | string | Listing title |
| `description` | string | Full listing description |
| `state` | string | Listing state (active, inactive, etc.) |
| `creation_timestamp` | int | Unix timestamp when created |
| `ending_timestamp` | int | When listing expires |
| `quantity` | int | Available quantity |
| `url` | string | Full URL to listing |
| `num_favorers` | int | Number of users who favorited |
| `is_customizable` | bool | Whether item can be customized |
| `is_personalizable` | bool | Whether item can be personalized |
| `listing_type` | string | Type: physical, digital, etc. |
| `tags` | array | Search tags (up to 13) |
| `materials` | array | Materials used |
| `processing_min` | int | Min processing days |
| `processing_max` | int | Max processing days |
| `who_made` | string | Who made it (i_did, collective, someone_else) |
| `when_made` | string | When made (made_to_order, 2020s, etc.) |
| `item_weight` | float | Item weight |
| `item_weight_unit` | string | Weight unit (oz, lb, g, kg) |
| `item_length` | float | Item length |
| `item_width` | float | Item width |
| `item_height` | float | Item height |
| `item_dimensions_unit` | string | Dimension unit (in, ft, mm, cm, m) |
| `should_auto_renew` | bool | Auto-renew when expires |
| `language` | string | Listing language |
| `price` | object | Price with `amount`, `divisor`, `currency_code` |
| `taxonomy_id` | int | Category taxonomy ID |
| `production_partners` | array | Production partner IDs |
| `views` | int | Number of views |

### Available Listing Fields

| Field | Type | Description |
|-------|------|-------------|
| `listing_id` | int | Unique listing identifier |
| `user_id` | int | Owner's user ID |
| `shop_id` | int | Shop identifier |
| `title` | string | Listing title |
| `description` | string | Full listing description |
| `state` | string | Listing state (active, inactive, etc.) |
| `creation_timestamp` | int | Unix timestamp when created |
| `created_timestamp` | int | Same as creation_timestamp |
| `ending_timestamp` | int | When listing expires |
| `original_creation_timestamp` | int | Original creation time |
| `last_modified_timestamp` | int | Last modification time |
| `updated_timestamp` | int | Same as last_modified_timestamp |
| `state_timestamp` | int | When state last changed |
| `quantity` | int | Available quantity |
| `shop_section_id` | int | Shop section this belongs to |
| `featured_rank` | int | Featured position in shop |
| `url` | string | Full URL to listing |
| `num_favorers` | int | Number of users who favorited |
| `non_taxable` | bool | Whether item is non-taxable |
| `is_taxable` | bool | Whether item is taxable |
| `is_customizable` | bool | Whether item can be customized |
| `is_personalizable` | bool | Whether item can be personalized |
| `personalization_is_required` | bool | Personalization required |
| `personalization_char_count_max` | int | Max chars for personalization |
| `personalization_instructions` | string | Instructions for personalization |
| `listing_type` | string | Type: physical, digital, etc. |
| `tags` | array | Search tags (up to 13) |
| `materials` | array | Materials used |
| `shipping_profile_id` | int | Shipping profile ID |
| `return_policy_id` | int | Return policy ID |
| `processing_min` | int | Min processing days |
| `processing_max` | int | Max processing days |
| `who_made` | string | Who made it (i_did, collective, someone_else) |
| `when_made` | string | When made (made_to_order, 2020s, etc.) |
| `is_supply` | bool | Whether it's a craft supply |
| `item_weight` | float | Item weight |
| `item_weight_unit` | string | Weight unit (oz, lb, g, kg) |
| `item_length` | float | Item length |
| `item_width` | float | Item width |
| `item_height` | float | Item height |
| `item_dimensions_unit` | string | Dimension unit (in, ft, mm, cm, m) |
| `is_private` | bool | Whether listing is private |
| `style` | array | Style tags |
| `file_data` | string | Digital file data |
| `has_variations` | bool | Whether has variations |
| `should_auto_renew` | bool | Auto-renew when expires |
| `language` | string | Listing language |
| `price` | object | Price with `amount`, `divisor`, `currency_code` (actual price = amount/divisor) |
| `taxonomy_id` | int | Category taxonomy ID |
| `production_partners` | array | Production partner IDs |
| `skus` | array | SKU codes |
| `views` | int | Number of views |

### Selected Review Fields

| Field | Type | Description |
|-------|------|-------------|
| `shop_id` | int | Shop that received the review |
| `listing_id` | int | Listing being reviewed |
| `buyer_user_id` | int | User ID of reviewer |
| `rating` | int | Rating 1-5 stars |
| `review` | string | Review text |
| `language` | string | Language code (en, fr, etc.) |
| `create_timestamp` | int | When review was created |

**Strategy:** API returns reviews sorted by `create_timestamp` DESC (newest first). Fetch page 1, stop when hitting a `create_timestamp` older than last sync. Ignore review edits - only store initial review.

### Available Review Fields (via getReviewsByShop)

| Field | Type | Description |
|-------|------|-------------|
| `shop_id` | int | Shop that received the review |
| `listing_id` | int | Listing being reviewed |
| `transaction_id` | int | Transaction/order ID |
| `buyer_user_id` | int | User ID of reviewer |
| `rating` | int | Rating 1-5 stars |
| `review` | string | Review text |
| `language` | string | Language code (en, fr, etc.) |
| `image_url_fullxfull` | string | Review image URL (if attached) |
| `create_timestamp` | int | When review was created |
| `created_timestamp` | int | Same as create_timestamp |
| `update_timestamp` | int | When review was last updated |
| `updated_timestamp` | int | Same as update_timestamp |
