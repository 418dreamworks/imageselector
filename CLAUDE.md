# Etsy Furniture Image Sync

## API Rate Limits
- **100,000 requests/day** (rolling 24h)
- **150 QPS**
- `sync_images.py` uses 90,000/day at 95 QPS
- Development: 10,000/day at 5 QPS max

## Algorithm

```
(A) Every 30 days, clear synced_shops to re-check for new listings

AT STARTUP (before crawling):
  (B)/(C) Fix existing data first:
    - For any listing without shop_id, call API to get shop_id
    - For any shop not in synced_shops, sync all its FURNITURE listings

For each leaf taxonomy:
  (D1) Crawl with sort=relevance until offset=10k or exhausted
  (D2) Reset offset=0, crawl with sort=created_desc until 10k or exhausted
  (D3) Reset offset=0, crawl with sort=created_asc until 10k or exhausted
  (D4) Reset offset=0, crawl with sort=price_desc until 10k or exhausted
  (D5) Reset offset=0, crawl with sort=price_asc until 10k or exhausted

  (B)/(C) AFTER all sorts exhausted for this leaf:
    - Fix any new listings missing shop_id
    - Sync any new unsynced shops

  Move to next leaf
```

## Data Files

- `images/` - JPG files named `{listing_id}.jpg`
- `image_metadata.json` - `{listing_id: {image_id, shop_id}}`
- `sync_progress.json` - crawl state, synced_shops list, API call count

## Key Points

- Only sync **furniture** listings (check `taxonomy_id` against `FURNITURE_TAXONOMY_IDS`)
- `image_id` = Etsy's `listing_image_id`, useful for detecting image updates
- `synced_shops` persists and only clears every 30 days
- Filename `listing_id.jpg` IS the link between image and listing
