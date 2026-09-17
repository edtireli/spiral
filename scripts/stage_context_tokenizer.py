#!/usr/bin/env python3
"""Stage a relocatable macOS vocabulary helper; never install or load a model.

Accepts an explicitly built executable, copies its non-system dylib closure,
rewrites copied load commands and ad-hoc signs the copies. Original libraries,
user settings and existing outputs are never modified. Release signing belongs
to the application's ordinary packaging pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(*argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30).stdout


def dependencies(path):
    lines = run('/usr/bin/otool', '-L', str(path)).splitlines()
    return [line.strip().split(' (compatibility version', 1)[0] for line in lines[1:] if line.startswith('\t')]


def system_library(name):
    return name.startswith(('/usr/lib/', '/System/Library/'))


def resolve_library(name, owner):
    if name.startswith(('@rpath/', '@loader_path/')):
        path = owner.parent / name.split('/', 1)[1]
    elif name.startswith('/'):
        path = Path(name)
    else:
        raise ValueError(f'unsupported library address: {name}')
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix != '.dylib':
        raise ValueError(f'not a dylib: {path}')
    return path


def stage(binary, output):
    if sys.platform != 'darwin':
        raise ValueError('this staging command requires macOS')
    binary = binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('an explicit executable is required')
    if output.exists() or output.is_symlink():
        raise ValueError('output already exists; preserve earlier runtime evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix='.tokenizer-stage-', dir=output.parent))
    try:
        graph, pending, names = {}, [binary], {}
        while pending:
            source = pending.pop()
            if source in graph:
                continue
            if len(graph) >= 64:
                raise ValueError('library closure exceeds its bound')
            edges = []
            for name in dependencies(source):
                if system_library(name):
                    continue
                target = resolve_library(name, source)
                if target == source:  # dylib's own install id
                    continue
                if target.name in names and names[target.name] != target:
                    raise ValueError('colliding dylib basenames')
                names[target.name] = target
                edges.append((name, target))
                pending.append(target)
            graph[source] = edges
        if sum(source.stat().st_size for source in graph) > 256 * 1024 * 1024:
            raise ValueError('library closure exceeds 256 MiB')
        targets = {source: staged / ('spiral-tokenize' if source == binary else 'lib/' + source.name)
                   for source in graph}
        for source, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            target.chmod(0o755)
        for source, edges in graph.items():
            target = targets[source]
            if source != binary:
                run('/usr/bin/install_name_tool', '-id', '@rpath/' + source.name, str(target))
            for name, dependency in edges:
                relative = ('@loader_path/lib/' if source == binary else '@loader_path/') + dependency.name
                run('/usr/bin/install_name_tool', '-change', name, relative, str(target))
            # Remove build-machine search paths; the copied closure uses exact
            # @loader_path addresses and must not accidentally borrow Homebrew.
            lines = run('/usr/bin/otool', '-l', str(target)).splitlines()
            for index, line in enumerate(lines):
                if line.strip() == 'cmd LC_RPATH':
                    for entry in lines[index+1:index+4]:
                        if entry.strip().startswith('path '):
                            rpath = entry.strip()[5:].split(' (offset ', 1)[0]
                            run('/usr/bin/install_name_tool', '-delete_rpath', rpath, str(target))
            run('/usr/bin/codesign', '--force', '--sign', '-', '--timestamp=none', str(target))
            run('/usr/bin/codesign', '--verify', str(target))
            for name in dependencies(target):
                if system_library(name) or (source != binary and name == '@rpath/' + source.name):
                    continue
                if not name.startswith('@loader_path/'):
                    raise ValueError('staged runtime still has an external dependency')
                resolved = (target.parent / name.removeprefix('@loader_path/')).resolve(strict=True)
                resolved.relative_to(staged.resolve())
        receipt = {'schema': 'spiral.context-tokenizer-runtime.v1', 'platform': sys.platform,
                   'executable': 'spiral-tokenize', 'scope': 'vocabulary-only helper; no model or inference bundled',
                   'files': [{'path': target.relative_to(staged).as_posix(),
                              'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                              'bytes': target.stat().st_size} for target in sorted(targets.values())]}
        (staged / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
        staged.rename(output)
        return receipt
    finally:
        if staged.exists():
            shutil.rmtree(staged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.binary, args.output.absolute()), indent=2))
