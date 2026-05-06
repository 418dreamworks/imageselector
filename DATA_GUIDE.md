# Data Guide

How all data in the system relates to each other.

## The Core Identifier

Every image is identified by `(listing_id, image_id)`. This pair is the key that connects everything — the database, the FAISS embeddings, the tar archives, and the primary index.

## Database (data/db/etsy_data.db)

The database is the source of truth for what exists and what state it's in.

### image_status table
Tracks every image through the pipeline:
```
listing_id + image_id  →  (is_primary, download_done, url)
```
- `download_done` values: -1=dead, 0=pending, 1=marker, 2=downloaded, 3=tarred, 4=shard finalized
- `is_primary=1` means this is the main display image for the listing

### listings_static table
Listing metadata (title, description, tags, materials, price, taxonomy_id, shop_id). Used by CLIP text embeddings — the title+materials get embedded alongside the image.

### Other tables
- `listings_dynamic` — time-series metrics (views, favorites, price changes)
- `shops_static` / `shops_dynamic` — shop info and metrics
- `reviews` — shop reviews
- `sync_state` — crawler bookkeeping

## Loose Images (Staging Only)

The only place loose JPGs exist:
```
images/imagedownload/    {listing_id}_{image_id}.jpg    max ~10K at a time
images/imageembedded/    {listing_id}_{image_id}.jpg    max ~10K at a time
```
These are temporary staging areas. Images flow through: download → embed → tar → delete loose file.

## FAISS Embeddings (data/embeddings/)

Symlink → /Volumes/SSD500GB/imageselector/data/embeddings

### Shard structure
```
data/embeddings/
  shard_0000/          500K images (finalized, immutable)
  shard_0001/          500K images (finalized, immutable)
  ...
  shard_0010/          <500K images (active, still receiving writes)
```

Each shard contains:
```
shard_NNNN/
  clip_vitb32.faiss         512-dim vectors
  clip_vitb32_text.faiss    512-dim vectors (from listing title+materials)
  clip_vitl14.faiss         768-dim vectors
  clip_vitl14_text.faiss    768-dim vectors (from listing title+materials)
  dinov2_base.faiss         768-dim vectors
  dinov2_large.faiss        1024-dim vectors
  dinov3_base.faiss         768-dim vectors
  image_index.json          maps FAISS row → (listing_id, image_id)
```

### Row alignment
**Row N in every .faiss file in a shard = the same image.** The identity of that image is in `image_index.json[N]`.

### image_index.json format
```json
[
  [listing_id, image_id, faiss_row, hash1, hash2, hash3, hash4, hash5, hash6, hash7],
  ...
]
```
- Elements 0-1: the (listing_id, image_id) pair
- Element 2: row number in the FAISS indexes
- Elements 3-9: MD5 hashes of each model's vector (integrity checks)

### How to search
```python
import faiss, json

shard = "data/embeddings/shard_0000"
index = faiss.read_index(f"{shard}/clip_vitb32.faiss")
image_index = json.load(open(f"{shard}/image_index.json"))

D, I = index.search(query_vector.reshape(1, -1), k=10)
for row in I[0]:
    listing_id, image_id = image_index[row][0], image_index[row][1]
    # Now use tar_index or primary_index to get the actual image bytes
```

## Tarred Images — All (images/imagetarred/)

Symlink → /Volumes/HDD1TB/images/imagetarred

Every background-removed image, packed in 10K batches. This is the complete archive.

```
images/imagetarred/
  imageall_00000.tar    10K images each
  imageall_00001.tar
  ...
  imageall_00533.tar
  tar_index.json        byte-offset index for O(1) access
```

### tar_index.json format
```json
{
  "tars": {
    "imageall_00000.tar": {
      "1234567_8901234.jpg": 0,
      "2345678_9012345.jpg": 28672
    }
  },
  "reverse": {
    "1234567_8901234": ["imageall_00000.tar", 0],
    "2345678_9012345": ["imageall_00000.tar", 28672]
  }
}
```
- `tars`: tar name → {filename: byte_offset}
- `reverse`: listing_image key → [tar_name, byte_offset]

### How to retrieve any image
```python
import json, tarfile

idx = json.load(open("images/imagetarred/tar_index.json"))
key = f"{listing_id}_{image_id}"
tar_name, offset = idx["reverse"][key]

with tarfile.open(f"images/imagetarred/{tar_name}", "r") as tf:
    tf.fileobj.seek(offset)
    member = tarfile.TarInfo.fromtarfile(tf)
    image_bytes = tf.extractfile(member).read()
```

## Tarred Images — Primary Only (images/imageprimary/)

Symlink → /Volumes/SSD500GB/imageselector/images/imageprimary

A subset: only the primary (is_primary=1) image per listing. Append-only — update_primary.py adds new listings' primaries to new tars each week. Use `--full` for a complete rebuild. No loose files.

```
images/imageprimary/
  imageprimary_00000.tar    10K images each
  imageprimary_00001.tar
  ...
  primary_index.json        byte-offset index for O(1) access
```

### primary_index.json format
```json
{
  "tars": {
    "imageprimary_00000.tar": {
      "1234567.jpg": 0,
      "2345678.jpg": 56832
    }
  },
  "reverse": {
    "1234567": ["imageprimary_00000.tar", 0],
    "2345678": ["imageprimary_00000.tar", 56832]
  }
}
```
- Files inside tars are named `{listing_id}.jpg` (no image_id — one primary per listing)
- `reverse` is keyed by listing_id → [tar_name, byte_offset]

### How to retrieve a primary image
```python
import json, tarfile

idx = json.load(open("images/imageprimary/primary_index.json"))
listing_id = "1234567"
tar_name, offset = idx["reverse"][listing_id]

with tarfile.open(f"images/imageprimary/{tar_name}", "r") as tf:
    tf.fileobj.seek(offset)
    member = tarfile.TarInfo.fromtarfile(tf)
    image_bytes = tf.extractfile(member).read()
```

## How Everything Connects

```
FAISS search result (row N)
    ↓
image_index.json[N] → (listing_id, image_id)
    ↓
Two options to get the actual image:
    ├── primary_index.json  → primary tar  (if you only need the listing's main image)
    └── tar_index.json      → all-image tar (any image by listing_id + image_id)
    ↓
Byte-offset seek into tar → image bytes

Listing metadata:
    listing_id → listings_static (title, price, tags, materials, taxonomy)
    listing_id → listings_dynamic (views, favorites over time)
    listing_id → shops_static/shops_dynamic (via shop_id)
```

## Physical Locations

| Data | Drive | Path |
|------|-------|------|
| Database | Primary SSD | data/db/etsy_data.db |
| Loose staging images | Primary SSD | images/imagedownload/, images/imageembedded/ |
| FAISS + image_index | SSD500GB | data/embeddings/shard_*/ |
| Primary image tars | SSD500GB | images/imageprimary/ |
| All-image tars | HDD1TB | images/imagetarred/ |
| Backups | HDD1TB | backups/ |
