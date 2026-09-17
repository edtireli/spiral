"""Unverified working files for explicit same-task continuation.

A checkpoint is data, never completion evidence or a replayed tool action. Only
managed project transactions participate; Git-backed archives remain manual.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from spiral.context_store import ContextStore
from spiral.transactions import _file_signature, _managed_paths


class CheckpointUnavailable(ValueError):
    pass


def _path(root, relative):
    if root.is_symlink():
        raise CheckpointUnavailable('checkpoint root cannot be a symbolic link')
    value = Path(relative)
    if (not isinstance(relative, str) or not relative or value.is_absolute()
            or '..' in value.parts or value.as_posix() != relative):
        raise CheckpointUnavailable('checkpoint path is not normalized and relative')
    current = root
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise CheckpointUnavailable('checkpoint paths cannot traverse symbolic links')
    current.resolve().relative_to(root.resolve())
    return current


def _editable(root, relative):
    from spiral.safety_kernel import protected_relative_path, protection_active
    from spiral.transactions import _SNAPSHOT_SKIP
    if any(part in _SNAPSHOT_SKIP for part in Path(relative).parts):
        raise CheckpointUnavailable('checkpoint contains harness, Git or cache state')
    if protection_active(root) and protected_relative_path(root, root / relative):
        raise CheckpointUnavailable('checkpoint contains protected source')


def seal(transaction, archive):
    """Bind archived bytes and the complete pre-task baseline while both exist."""
    if not transaction.managed or archive is None or transaction.snapshot_dir is None:
        return None
    root = transaction.root
    _path(root, archive.relative_to(root).as_posix())
    _path(root, (transaction.snapshot_dir / 'tree').relative_to(root).as_posix())
    manifest = json.loads((archive / 'manifest.json').read_text())
    paths = sorted(transaction.snapshot_paths or ())
    if len(paths) > 2048 or len(manifest['changed']) > 2048:
        raise CheckpointUnavailable('working checkpoint exceeds its file bound')
    baseline = {}
    for rel in paths:
        source = _path(transaction.snapshot_dir / 'tree', rel)
        baseline[rel] = _file_signature(source)
    changed = {}
    total = 0
    for rel in manifest['changed']:
        _editable(root, rel)
        source = _path(archive / 'changed', rel)
        if not source.is_file():
            raise CheckpointUnavailable('working checkpoint requires regular files')
        total += source.stat().st_size
        if total > 32 * 1024 * 1024:
            raise CheckpointUnavailable('working checkpoint exceeds its byte bound')
        changed[rel] = _file_signature(source)
    for rel in manifest['deleted']:
        _editable(root, rel)
    value = {'schema': 'spiral.working-checkpoint.v1',
        'archive': archive.relative_to(root).as_posix(), 'baseline': baseline,
        'changed': changed, 'deleted': manifest['deleted'], 'verified': False}
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded.encode()) > 512_000:
        raise CheckpointUnavailable('working checkpoint manifest is too large')
    return ContextStore(root).save(encoded, kind='working-checkpoint')


def restore(transaction, reference):
    """Prevalidate every byte before touching the same unchanged project baseline.

    I/O failures during publication propagate to the owning transaction's normal
    rollback. Rejected/legacy checkpoints perform no writes.
    """
    if not transaction.managed:
        raise CheckpointUnavailable('working continuation requires a managed transaction')
    match = re.fullmatch(r'\.spiral/context/working-checkpoint-([a-f0-9]{64})\.txt', reference)
    if not match:
        raise CheckpointUnavailable('working checkpoint has no content identity')
    root = transaction.root
    try:
        source = _path(root, reference)
        if source.stat().st_size > 512_000:
            raise CheckpointUnavailable('working checkpoint manifest is too large')
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != match[1]:
            raise CheckpointUnavailable('working checkpoint manifest changed')
        data = json.loads(raw)
        if data.get('schema') != 'spiral.working-checkpoint.v1' or data.get('verified') is not False:
            raise CheckpointUnavailable('unsupported working checkpoint')
        baseline, changed, deleted = data['baseline'], data['changed'], data['deleted']
        if (not isinstance(baseline, dict) or not isinstance(changed, dict)
                or not isinstance(deleted, list) or len(baseline) > 2048 or len(changed) > 2048
                or len(deleted) > 2048 or set(changed) & set(deleted)):
            raise CheckpointUnavailable('invalid working checkpoint inventory')
        if set(_managed_paths(root)) != set(baseline):
            raise CheckpointUnavailable('project baseline paths changed')
        for rel, signature in baseline.items():
            if _file_signature(_path(root, rel)) != signature:
                raise CheckpointUnavailable('project baseline contents changed')
        archive_rel = data['archive']
        if not re.fullmatch(r'\.spiral/recovery/[a-zA-Z0-9_-]+', archive_rel):
            raise CheckpointUnavailable('archive is not a task recovery directory')
        archive = _path(root, archive_rel)
        prepared = []
        total = 0
        for rel, signature in changed.items():
            _editable(root, rel)
            destination = _path(root, rel)
            original = _path(archive / 'changed', rel)
            if not original.is_file() or _file_signature(original) != signature:
                raise CheckpointUnavailable('archived candidate bytes changed')
            total += original.stat().st_size
            if total > 32 * 1024 * 1024:
                raise CheckpointUnavailable('working checkpoint exceeds its byte bound')
            raw = original.read_bytes()
            # Read once; validate the actual bytes/mode being published too.
            mode = original.stat().st_mode & 0o777
            digest = hashlib.sha256(b'file\0' + str(mode).encode() + raw).hexdigest()
            if digest != signature:
                raise CheckpointUnavailable('candidate changed during reading')
            prepared.append((destination, raw, mode))
        removals = []
        for rel in deleted:
            _editable(root, rel)
            if rel not in baseline:
                raise CheckpointUnavailable('deleted path was not in the original baseline')
            removals.append(_path(root, rel))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointUnavailable(str(exc)) from exc
    # All identity, ownership, content and path checks precede publication.
    for destination, raw, mode in prepared:
        _path(root, destination.relative_to(root).as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as staged:
            temporary = Path(staged.name)
            try:
                staged.write(raw); staged.flush(); os.fchmod(staged.fileno(), mode)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    for destination in removals:
        _path(root, destination.relative_to(root).as_posix())
        destination.unlink()
    return {'changed': len(prepared), 'deleted': len(removals), 'verified': False}
