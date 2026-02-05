#!/usr/bin/env python3
"""
Distributed Embedding Orchestrator

Manages distributed embedding across multiple workers:
- iMac (local): embed@localhost, MPS
- MBP: embed@mbp.local, MPS
- Windows: embed@192.168.68.117, CUDA (GTX 1070)

Architecture:
1. Export batches of images for each worker
2. SSH to start worker processes
3. Rsync to poll for completion and retrieve results
4. Import results into FAISS indexes
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import sqlite3

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from image_db import DB_FILE, get_connection, get_images_for_embedding

# Worker configurations
@dataclass
class Worker:
    name: str
    host: str
    user: str
    work_dir: str
    python_path: str
    models: List[str]  # Models this worker should process
    speed_factor: float  # Relative speed (higher = faster)

WORKERS = {
    'imac': Worker(
        name='imac',
        host='localhost',
        user='embed',
        work_dir='/Users/embed/imageselector',
        python_path='/Users/embed/imageselector/venv/bin/python',
        models=['clip_vitb32', 'clip_vitl14', 'dinov2_base', 'dinov2_large', 'dinov3_base'],
        speed_factor=1.0,
    ),
    'mbp': Worker(
        name='mbp',
        host='mbp.local',
        user='embed',
        work_dir='/Users/embed/imageselector',
        python_path='/Users/embed/imageselector/venv/bin/python',
        models=['clip_vitb32', 'clip_vitl14', 'dinov2_base', 'dinov2_large', 'dinov3_base'],
        speed_factor=2.1,  # ~2x faster than iMac based on benchmark
    ),
    'windows': Worker(
        name='windows',
        host='192.168.68.117',
        user='embed',
        work_dir='E:/embed_work/imageselector',
        python_path='E:/embed_work/imageselector/venv/Scripts/python.exe',
        models=['clip_vitb32', 'clip_vitl14', 'dinov2_base', 'dinov2_large', 'dinov3_base'],
        speed_factor=2.4,  # Fastest based on benchmark
    ),
}

# All available models
ALL_MODELS = ['clip_vitb32', 'clip_vitl14', 'dinov2_base', 'dinov2_large', 'dinov3_base']


class Orchestrator:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.exports_dir = base_dir / 'embed_exports'
        self.exports_dir.mkdir(exist_ok=True)
        self.db_path = DB_FILE

    def get_pending_images(self, model: str, limit: int = 1000) -> List[Tuple[int, int]]:
        """Get images that need embedding for a specific model."""
        conn = get_connection()
        cursor = conn.cursor()

        # Get images where bg_removed=1 but embed_{model}=0
        cursor.execute(f"""
            SELECT listing_id, image_id
            FROM image_status
            WHERE bg_removed = 1 AND embed_{model} = 0
            LIMIT ?
        """, (limit,))

        results = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_pending_by_all_models(self, limit: int = 1000) -> Dict[str, List[Tuple[int, int]]]:
        """Get pending images grouped by model."""
        return {model: self.get_pending_images(model, limit) for model in ALL_MODELS}

    def export_batch(self, batch_name: str, images: List[Tuple[int, int]],
                     models: List[str]) -> Path:
        """Export a batch of images for processing."""
        batch_dir = self.exports_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        images_dir = batch_dir / 'images'
        images_dir.mkdir(exist_ok=True)

        # Get listing info for text embeddings
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        listings_data = {}
        manifest = []

        for listing_id, image_id in images:
            # Get listing info
            cursor.execute("""
                SELECT title, materials
                FROM listings
                WHERE listing_id = ?
            """, (listing_id,))
            row = cursor.fetchone()

            if row:
                title, materials = row
                text = f"{title}. {materials}" if materials else title
                listings_data[str(listing_id)] = text

            # Copy image
            src_path = self.base_dir / 'images' / f"{listing_id}_{image_id}.jpg"
            if src_path.exists():
                dst_path = images_dir / f"{listing_id}_{image_id}.jpg"
                if not dst_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)
                manifest.append([listing_id, image_id])

        conn.close()

        # Write manifest
        with open(batch_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f)

        # Write listings
        with open(batch_dir / 'listings.json', 'w') as f:
            json.dump(listings_data, f)

        # Write models to process
        with open(batch_dir / 'models.json', 'w') as f:
            json.dump(models, f)

        print(f"Exported {len(manifest)} images to {batch_dir}")
        return batch_dir

    def rsync_to_worker(self, worker: Worker, batch_dir: Path) -> bool:
        """Rsync batch to worker."""
        batch_name = batch_dir.name

        if worker.host == 'localhost':
            target = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}"
        elif worker.host == '192.168.68.117':
            # Windows with cwrsync uses /cygdrive/e/ style paths
            target = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}"
        else:
            target = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}"

        cmd = ['rsync', '-av', '--delete', f"{batch_dir}/", target]
        print(f"Syncing to {worker.name}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Rsync failed: {result.stderr}")
            return False
        return True

    def start_worker(self, worker: Worker, batch_name: str, models: List[str]) -> bool:
        """Start worker process via SSH."""
        worker_script = f"{worker.work_dir}/image_search_v2/embed_worker.py"
        batch_path = f"{worker.work_dir}/embed_exports/{batch_name}"

        # Worker uses --input and --model (one at a time)
        # Run each model sequentially
        for model in models:
            if worker.host == '192.168.68.117':
                # Windows paths
                remote_cmd = f'"{worker.python_path}" "{worker_script}" --input "{batch_path}" --output "{batch_path}" --model {model}'
            else:
                # Mac (localhost or mbp)
                remote_cmd = f'{worker.python_path} {worker_script} --input {batch_path} --output {batch_path} --model {model}'

            # All workers use SSH (including localhost via embed@localhost)
            ssh_host = 'localhost' if worker.host == 'localhost' else worker.host
            ssh_cmd = ['ssh', f'{worker.user}@{ssh_host}', remote_cmd]
            print(f"Starting {worker.name} ({model}): {' '.join(ssh_cmd)}")

            subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"Started worker on {worker.name} for models: {models}")
        return True

    def check_completion_local(self, batch_name: str, models: List[str]) -> Dict[str, bool]:
        """Check which model outputs exist locally (after rsync pull)."""
        results = {}
        batch_dir = self.exports_dir / batch_name
        for model in models:
            npy_file = batch_dir / f"{model}.npy"
            results[model] = npy_file.exists()
        return results

    def rsync_results(self, worker: Worker, batch_name: str) -> bool:
        """Rsync only .npy results back from worker (fast - ignores images)."""
        local_batch = self.exports_dir / batch_name

        if worker.host == 'localhost':
            src = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}/"
        elif worker.host == '192.168.68.117':
            src = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}/"
        else:
            src = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}/"

        # Only sync .npy files - fast even with 50k images in directory
        cmd = ['rsync', '-av', '--include=*.npy', '--exclude=*', src, f"{local_batch}/"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Rsync failed: {result.stderr}")
            return False
        return True

    def get_images_needing_models(self, models: List[str], limit: int = 1000) -> List[Tuple[int, int]]:
        """Get images that need embedding for ANY of the specified models."""
        conn = get_connection()
        cursor = conn.cursor()

        # Get images where ANY of the specified model flags are 0
        model_conditions = ' OR '.join(f'embed_{m} = 0' for m in models)
        cursor.execute(f"""
            SELECT listing_id, image_id
            FROM image_status
            WHERE bg_removed = 1 AND ({model_conditions})
            LIMIT ?
        """, (limit,))

        results = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        return results

    def distribute_work(self, total_images: int, workers: List[str],
                       models: List[str], assignments: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> Dict[str, Dict]:
        """
        Distribute images across workers. Each worker processes ALL models for their images.

        Args:
            total_images: Total images to process
            workers: List of worker names to use
            models: Models each worker should process
            assignments: Optional explicit {worker: [(listing_id, image_id), ...]} assignments

        Returns dict of worker_name -> {images: [(listing_id, image_id), ...], models: [...]}
        """
        # Get active workers
        active_workers = [WORKERS[w] for w in workers if w in WORKERS]

        if not active_workers:
            print("No workers specified!")
            return {}

        # If explicit assignments provided, use those
        if assignments:
            distribution = {}
            for worker_name, images in assignments.items():
                if worker_name in WORKERS:
                    worker = WORKERS[worker_name]
                    worker_models = [m for m in models if m in worker.models]
                    distribution[worker_name] = {
                        'images': images,
                        'models': worker_models,
                    }
            return distribution

        # Auto-distribute: get images needing any of the specified models
        pending_list = self.get_images_needing_models(models, total_images)

        if not pending_list:
            print("No pending images found!")
            return {}

        # Calculate total speed factor
        total_speed = sum(w.speed_factor for w in active_workers)

        # Distribute images proportionally by speed
        distribution = {}
        start_idx = 0

        for i, worker in enumerate(active_workers):
            # Last worker gets remainder
            if i == len(active_workers) - 1:
                worker_images = pending_list[start_idx:]
            else:
                proportion = worker.speed_factor / total_speed
                count = int(len(pending_list) * proportion)
                worker_images = pending_list[start_idx:start_idx + count]
                start_idx += count

            # Each worker processes ALL models for their images
            worker_models = [m for m in models if m in worker.models]

            distribution[worker.name] = {
                'images': worker_images,
                'models': worker_models,
            }

        return distribution

    def assign_specific_images(self, worker_name: str, listing_image_pairs: List[Tuple[int, int]],
                                models: List[str]) -> Dict[str, Dict]:
        """Assign specific images to a specific worker."""
        if worker_name not in WORKERS:
            print(f"Unknown worker: {worker_name}")
            return {}

        worker = WORKERS[worker_name]
        worker_models = [m for m in models if m in worker.models]

        return {
            worker_name: {
                'images': listing_image_pairs,
                'models': worker_models,
            }
        }


def cmd_status(args):
    """Show current embedding status."""
    orch = Orchestrator(Path(args.base_dir))

    conn = get_connection()
    cursor = conn.cursor()

    print("Embedding status by model:")
    print(f"  {'Model':<15} {'Embedded':>10} {'Pending':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")

    for model in ALL_MODELS:
        cursor.execute(f"SELECT COUNT(*) FROM image_status WHERE embed_{model} = 1")
        embedded = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM image_status WHERE bg_removed = 1 AND embed_{model} = 0")
        pending = cursor.fetchone()[0]
        print(f"  {model:<15} {embedded:>10,} {pending:>10,}")

    conn.close()

    print("\nWorkers:")
    for name, worker in WORKERS.items():
        print(f"  {name}: {worker.host} ({worker.speed_factor}x speed)")


def parse_assignment(assignment_str: str, orch: 'Orchestrator') -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """
    Parse assignment string like:
      'windows:1000:clip_vitb32,clip_vitl14'  -> worker, count, models
      'mbp:500'                                -> worker, count, all models
      'imac:2000:dinov2_base'                  -> worker, count, specific model

    Returns (worker_name, images, models)
    """
    parts = assignment_str.split(':')
    worker_name = parts[0]

    if len(parts) < 2:
        raise ValueError(f"Assignment must have at least worker:count, got: {assignment_str}")

    count = int(parts[1])

    # Models are optional - default to all
    if len(parts) >= 3:
        models = parts[2].split(',')
    else:
        models = ALL_MODELS

    # Get pending images for this worker's models
    images = orch.get_images_needing_models(models, count)

    return worker_name, images, models


def cmd_run(args):
    """Run distributed embedding job."""
    orch = Orchestrator(Path(args.base_dir))

    # Check for explicit assignments
    if args.assign:
        distribution = {}
        used_images = set()  # Track images to avoid duplicates across workers

        for assignment_str in args.assign:
            worker_name, images, models = parse_assignment(assignment_str, orch)

            # Filter out already-assigned images
            images = [(lid, iid) for lid, iid in images if (lid, iid) not in used_images]
            used_images.update(images)

            if worker_name in WORKERS:
                distribution[worker_name] = {
                    'images': images,
                    'models': models,
                }

        print("Explicit assignments:")
        for worker_name, work in distribution.items():
            print(f"  {worker_name}: {len(work['images'])} images, models={work['models']}")
    else:
        # Auto-distribute mode
        workers = args.workers.split(',') if args.workers else list(WORKERS.keys())
        models = args.models.split(',') if args.models else ALL_MODELS

        print(f"Workers: {workers}")
        print(f"Models: {models}")
        print(f"Images per batch: {args.batch_size}")

        # Distribute work
        distribution = orch.distribute_work(args.batch_size, workers, models)

    if not distribution:
        print("No work to distribute!")
        return

    # Show distribution
    print("\nWork distribution:")
    for worker_name, work in distribution.items():
        print(f"  {worker_name}: {len(work['images'])} images, models={work['models']}")

    if args.dry_run:
        print("\nDry run - not executing")
        return

    # Export and start each worker
    batch_id = f"batch_{int(time.time())}"
    worker_batches = {}

    for worker_name, work in distribution.items():
        if not work['images']:
            continue

        batch_name = f"{batch_id}_{worker_name}"
        worker = WORKERS[worker_name]

        # Export batch
        batch_dir = orch.export_batch(batch_name, work['images'], work['models'])

        # Rsync to worker
        if not orch.rsync_to_worker(worker, batch_dir):
            print(f"Failed to sync to {worker_name}")
            continue

        # Start worker
        orch.start_worker(worker, batch_name, work['models'])
        worker_batches[worker_name] = batch_name

    if not worker_batches:
        print("No workers started!")
        return

    # Poll for completion by pulling results periodically
    print("\nWaiting for workers to complete (pulling results periodically)...")
    poll_interval = 30  # seconds
    timeout = 3600  # 1 hour max
    start_time = time.time()

    completed = {w: set() for w in worker_batches}

    while time.time() - start_time < timeout:
        all_done = True

        for worker_name, batch_name in worker_batches.items():
            worker = WORKERS[worker_name]
            work = distribution[worker_name]

            # Pull results from worker (rsync/scp)
            orch.rsync_results(worker, batch_name)

            # Check locally what files arrived
            status = orch.check_completion_local(batch_name, work['models'])

            for model, done in status.items():
                if done and model not in completed[worker_name]:
                    completed[worker_name].add(model)
                    print(f"  {worker_name}: {model} complete")

            if not all(status.values()):
                all_done = False

        if all_done:
            print("\nAll workers complete!")
            break

        elapsed = int(time.time() - start_time)
        print(f"  [{elapsed}s] Waiting...")
        time.sleep(poll_interval)
    else:
        print("\nTimeout waiting for workers!")

    # Import results into FAISS/DB
    print("\nImporting results into FAISS/DB...")
    for worker_name, batch_name in worker_batches.items():
        work = distribution[worker_name]
        batch_dir = orch.exports_dir / batch_name
        models_str = ','.join(work['models'])

        print(f"  Importing {worker_name} ({batch_name})...")
        import_cmd = [
            sys.executable,
            str(Path(__file__).parent / 'embed_manager.py'),
            'import',
            '--results-dir', str(batch_dir),
            '--models', models_str
        ]
        result = subprocess.run(import_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Import failed: {result.stderr}")
        else:
            print(f"    {result.stdout.strip().split(chr(10))[-1]}")  # Last line of output

    # Cleanup: remove batch directories locally and on workers
    print("\nCleaning up batch directories...")
    for worker_name, batch_name in worker_batches.items():
        worker = WORKERS[worker_name]
        batch_dir = orch.exports_dir / batch_name

        # Remove local batch directory
        if batch_dir.exists():
            import shutil
            shutil.rmtree(batch_dir)
            print(f"  Removed local: {batch_dir}")

        # Remove on worker using rsync --delete with empty dir
        empty_dir = orch.exports_dir / '.empty'
        empty_dir.mkdir(exist_ok=True)

        if worker.host == 'localhost':
            target = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}"
        elif worker.host == '192.168.68.117':
            target = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}"
        else:
            target = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}"

        # rsync empty dir with --delete removes all files in target
        cmd = ['rsync', '-av', '--delete', f"{empty_dir}/", target]
        subprocess.run(cmd, capture_output=True)
        print(f"  Cleaned worker: {worker_name}")

        # Also remove the now-empty dir on worker
        if worker.host == 'localhost':
            rm_cmd = ['ssh', 'embed@localhost', f'rmdir /Users/embed/imageselector/embed_exports/{batch_name}']
        elif worker.host == '192.168.68.117':
            rm_cmd = ['ssh', f'{worker.user}@{worker.host}', f'cmd /c rmdir E:\\embed_work\\imageselector\\embed_exports\\{batch_name}']
        else:
            rm_cmd = ['ssh', f'{worker.user}@{worker.host}', f'rmdir {worker.work_dir}/embed_exports/{batch_name}']
        subprocess.run(rm_cmd, capture_output=True)

    # Show final status
    print("\nFinal status:")
    subprocess.run([sys.executable, str(Path(__file__)), 'status'])


def cmd_export(args):
    """Export a batch for manual processing."""
    orch = Orchestrator(Path(args.base_dir))

    models = args.models.split(',') if args.models else ALL_MODELS

    # Get pending images
    all_pending = set()
    for model in models:
        pending = orch.get_pending_images(model, args.count)
        all_pending.update(pending)

    images = list(all_pending)[:args.count]

    if not images:
        print("No pending images!")
        return

    batch_dir = orch.export_batch(args.name, images, models)
    print(f"Exported to: {batch_dir}")

    if args.worker:
        worker = WORKERS.get(args.worker)
        if worker:
            orch.rsync_to_worker(worker, batch_dir)
        else:
            print(f"Unknown worker: {args.worker}")


def main():
    parser = argparse.ArgumentParser(description='Distributed Embedding Orchestrator')
    parser.add_argument('--base-dir', default='/Users/tzuohannlaw/Documents/418Dreamworks/imageselector',
                       help='Base directory')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show embedding status')
    status_parser.set_defaults(func=cmd_status)

    # Run command
    run_parser = subparsers.add_parser('run', help='Run distributed embedding')
    run_parser.add_argument('--workers', help='Comma-separated worker names (default: all)')
    run_parser.add_argument('--models', help='Comma-separated model names (default: all)')
    run_parser.add_argument('--batch-size', type=int, default=1000, help='Total images to process')
    run_parser.add_argument('--assign', action='append', metavar='WORKER:COUNT[:MODELS]',
                           help='Explicit assignment: worker:count or worker:count:model1,model2. Can be repeated.')
    run_parser.add_argument('--dry-run', action='store_true', help='Show distribution without running')
    run_parser.set_defaults(func=cmd_run)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export batch for manual processing')
    export_parser.add_argument('name', help='Batch name')
    export_parser.add_argument('--count', type=int, default=100, help='Number of images')
    export_parser.add_argument('--models', help='Comma-separated model names')
    export_parser.add_argument('--worker', help='Worker to sync to')
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
