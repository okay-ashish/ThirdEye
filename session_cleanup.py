import os
import shutil
import time
import uuid
import logging
from flask import session, current_app

logger = logging.getLogger(__name__)

# Base configuration constants
DEFAULT_RETENTION_HOURS = 6

# Explicit protected directories that MUST NEVER be deleted or modified by cleanup
PROTECTED_PATHS = {
    'static/person_db',
    'static/sketch_parts',
    'static/img',
    'model',
    'templates',
    'static/css',
    'static/js'
}


def is_path_safe(target_path):
    """
    Validates that a path is safe for deletion and does NOT intersect
    with any protected permanent application directory.
    """
    if not target_path:
        return False

    abs_target = os.path.abspath(target_path)
    base_dir = os.path.abspath(os.getcwd())

    # Must be within project directory
    if not abs_target.startswith(base_dir):
        return False

    # Check against protected directories
    for prot in PROTECTED_PATHS:
        abs_prot = os.path.abspath(os.path.join(base_dir, prot))
        if abs_target == abs_prot or abs_target.startswith(abs_prot + os.sep):
            logger.error(f"[SECURITY ALERT] Attempted cleanup on protected directory: {target_path}")
            return False

    # Target must be specifically within an allowed temporary root
    allowed_temp_roots = [
        os.path.abspath(os.path.join(base_dir, 'static', 'generated_faces')),
        os.path.abspath(os.path.join(base_dir, 'static', 'generated_variations')),
        os.path.abspath(os.path.join(base_dir, 'static', 'generated_posters')),
        os.path.abspath(os.path.join(base_dir, 'static', 'uploads'))
    ]

    is_in_allowed_root = any(abs_target.startswith(root + os.sep) for root in allowed_temp_roots)
    return is_in_allowed_root


def get_or_create_session_id():
    """
    Retrieves the current user's session identifier from the Flask session.
    If not present, generates a secure random UUID hex identifier.
    Guarantees no username or raw user input is ever used in filesystem paths.
    """
    if 'session_id' not in session or not session['session_id']:
        session['session_id'] = uuid.uuid4().hex[:16]
    return session['session_id']


def get_session_dir(category, session_id=None):
    """
    Returns the absolute path to a session-specific directory for a given category.
    Creates the directory if it does not already exist.

    Categories:
    - 'faces': static/generated_faces/session_<session_id>
    - 'variations': static/generated_variations/session_<session_id>
    - 'posters': static/generated_posters/session_<session_id>
    - 'uploads': static/uploads/temp_<session_id>
    """
    if not session_id:
        session_id = get_or_create_session_id()

    # Sanitize session_id to alphanumeric characters only
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c == '_')
    if not safe_session_id:
        safe_session_id = uuid.uuid4().hex[:16]

    base_dir = os.path.abspath(os.getcwd())

    if category == 'faces':
        target = os.path.join(base_dir, 'static', 'generated_faces', f"session_{safe_session_id}")
    elif category == 'variations':
        target = os.path.join(base_dir, 'static', 'generated_variations', f"session_{safe_session_id}")
    elif category == 'posters':
        target = os.path.join(base_dir, 'static', 'generated_posters', f"session_{safe_session_id}")
    elif category == 'uploads':
        target = os.path.join(base_dir, 'static', 'uploads', f"temp_{safe_session_id}")
    else:
        raise ValueError(f"Unknown category for session directory: {category}")

    os.makedirs(target, exist_ok=True)
    return target


def cleanup_temp_file(filepath):
    """
    Safely removes an individual temporary file (such as a probe sketch upload).
    Catches Windows file locking errors (PermissionError, OSError) gracefully.
    """
    if not filepath or not os.path.exists(filepath):
        return

    if not is_path_safe(filepath):
        logger.warning(f"Skipping deletion: Path {filepath} is not verified as safe temporary storage.")
        return

    try:
        os.remove(filepath)
        logger.info(f"Temporary file cleaned: {filepath}")
    except PermissionError as pe:
        logger.warning(f"File locked on Windows (PermissionError), deferred deletion: {filepath} - {pe}")
    except OSError as oe:
        logger.warning(f"Could not delete temporary file {filepath}: {oe}")


def _on_rmtree_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree on Windows.
    Tries clearing read-only flags and retrying after a tiny delay.
    """
    import stat
    time.sleep(0.05)
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logger.warning(f"Could not remove locked path {path} on Windows: {e}")


def cleanup_session_artifacts(session_id):
    """
    Deletes all temporary generated directories and files associated with a specific session_id.
    Invoked during user logout.

    Target directories:
    - static/generated_faces/session_<session_id>
    - static/generated_variations/session_<session_id>
    - static/generated_posters/session_<session_id>
    - static/uploads/temp_<session_id>
    """
    if not session_id:
        return 0

    safe_session_id = "".join(c for c in session_id if c.isalnum() or c == '_')
    if not safe_session_id:
        return 0

    base_dir = os.path.abspath(os.getcwd())
    target_dirs = [
        os.path.join(base_dir, 'static', 'generated_faces', f"session_{safe_session_id}"),
        os.path.join(base_dir, 'static', 'generated_variations', f"session_{safe_session_id}"),
        os.path.join(base_dir, 'static', 'generated_posters', f"session_{safe_session_id}"),
        os.path.join(base_dir, 'static', 'uploads', f"temp_{safe_session_id}")
    ]

    deleted_count = 0
    for tdir in target_dirs:
        if os.path.exists(tdir) and is_path_safe(tdir):
            try:
                for root, dirs, files in os.walk(tdir):
                    deleted_count += len(files)
                shutil.rmtree(tdir, ignore_errors=False, onerror=_on_rmtree_error)
                logger.info(f"Cleaned session directory: {tdir}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not remove session directory {tdir}: {e}")

    return deleted_count


def cleanup_stale_sessions(retention_hours=DEFAULT_RETENTION_HOURS):
    """
    Scans temporary roots for abandoned session folders older than retention_hours.
    Compares folder/file modification times against (now - retention_seconds).
    Windows file locking is handled with graceful error suppression.
    """
    base_dir = os.path.abspath(os.getcwd())
    temp_roots = [
        os.path.join(base_dir, 'static', 'generated_faces'),
        os.path.join(base_dir, 'static', 'generated_variations'),
        os.path.join(base_dir, 'static', 'generated_posters'),
        os.path.join(base_dir, 'static', 'uploads')
    ]

    now = time.time()
    cutoff_time = now - (retention_hours * 3600)
    pruned_count = 0

    for root_dir in temp_roots:
        if not os.path.exists(root_dir):
            continue

        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)

            # Only target session subdirectories or temp uploads
            if os.path.isdir(entry_path):
                if entry.startswith('session_') or entry.startswith('temp_'):
                    if not is_path_safe(entry_path):
                        continue
                    try:
                        mtime = os.path.getmtime(entry_path)
                        if mtime < cutoff_time:
                            shutil.rmtree(entry_path)
                            pruned_count += 1
                            logger.info(f"Pruned stale session directory: {entry_path}")
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Could not prune stale directory {entry_path}: {e}")
            elif os.path.isfile(entry_path):
                # Loose temporary files in uploads like temp_sketch_*
                if entry.startswith('temp_'):
                    if not is_path_safe(entry_path):
                        continue
                    try:
                        mtime = os.path.getmtime(entry_path)
                        if mtime < cutoff_time:
                            os.remove(entry_path)
                            pruned_count += 1
                            logger.info(f"Pruned stale temporary file: {entry_path}")
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Could not prune stale file {entry_path}: {e}")

    return pruned_count


def cleanup_benchmark_artifacts(
    benchmark_dir="static/benchmark_eval",
    max_files_per_phase=20,
    max_age_days=7,
    dry_run=False
):
    """
    Phase 4.4: Separate Cleanup Policy for Development Benchmark Artifacts.
    Retains newest `max_files_per_phase` files per phase/prefix.
    Prunes files older than `max_age_days` (default: 7 days) if exceeding retention.
    Never deletes permanent assets, source test faces, or models.
    Returns audit dictionary with list of removed files, bytes freed, and retention stats.
    """
    base_dir = os.path.abspath(os.getcwd())
    bench_abs = os.path.abspath(os.path.join(base_dir, benchmark_dir))

    if not os.path.exists(bench_abs):
        return {'removed': [], 'bytes_freed': 0, 'retained': 0}

    now = time.time()
    cutoff_time = now - (max_age_days * 86400)

    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(bench_abs):
        for f in files:
            fpath = os.path.join(root, f)
            all_files.append(fpath)

    # Group files by phase/prefix tag
    # e.g., p43, p422, p421, phase4, bench, mask_debug, scratch, general
    groups = {}
    for fpath in all_files:
        fname = os.path.basename(fpath).lower()
        tag = 'general'
        for prefix in ['p46', 'p44', 'p43', 'p422', 'p421', 'phase4_2', 'phase4_1', 'phase4', 'mask_debug', 'bench', 'scratch']:
            if fname.startswith(prefix):
                tag = prefix
                break
        if tag not in groups:
            groups[tag] = []
        try:
            mtime = os.path.getmtime(fpath)
            size = os.path.getsize(fpath)
        except OSError:
            mtime = 0
            size = 0
        groups[tag].append({'path': fpath, 'mtime': mtime, 'size': size})

    removed = []
    bytes_freed = 0
    retained_count = 0

    for tag, file_list in groups.items():
        # Sort newest first
        file_list.sort(key=lambda x: x['mtime'], reverse=True)
        for idx, item in enumerate(file_list):
            fpath = item['path']
            mtime = item['mtime']
            size = item['size']

            # Retention criteria:
            # Within max_files_per_phase quota AND not older than max_age_days -> RETAIN
            # If exceeding max_files_per_phase OR older than max_age_days -> CANDIDATE FOR CLEANUP
            # However, always retain at least the 5 newest files regardless of age
            should_prune = False
            if idx >= max_files_per_phase:
                should_prune = True
            elif idx >= 5 and mtime < cutoff_time:
                should_prune = True

            if should_prune:
                if not dry_run:
                    try:
                        os.remove(fpath)
                        logger.info(f"[BENCHMARK CLEANUP] Removed: {fpath} ({size} bytes)")
                        removed.append(fpath)
                        bytes_freed += size
                    except (PermissionError, OSError) as e:
                        logger.warning(f"[BENCHMARK CLEANUP] Could not remove {fpath}: {e}")
                else:
                    removed.append(fpath)
                    bytes_freed += size
            else:
                retained_count += 1

    return {
        'removed': removed,
        'removed_count': len(removed),
        'bytes_freed': bytes_freed,
        'retained_count': retained_count,
        'dry_run': dry_run
    }
