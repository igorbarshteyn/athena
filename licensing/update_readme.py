#!/usr/bin/env python3
"""Preview or apply the prepared README licensing section, offline and fail-closed.

No third-party dependencies. Python 3.9 or newer. The initial edit requires the
reviewed Git blob hash. Already-managed notices must match the supplied block.
No source files, Git history, remote repositories, or runtime settings are edited.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Optional, Sequence

START = '<!-- ATHENA-NCRD-LICENSE-START -->'
END = '<!-- ATHENA-NCRD-LICENSE-END -->'
LEGACY = ('MIT licensed. Orpheus weights: Llama 3.2 community license. '
          'SNAC codec: MIT. Qwen3.5: Apache-2.0.')
REPOSITORIES = {'igorbarshteyn/athena', 'igorbarshteyn/athena-consciousness'}


class UpdateError(Exception):
    """A validation failure; never silently force an edit."""


def blob_hash(data: bytes) -> str:
    """Compute Git's SHA-1 blob identity, not the repository commit hash."""
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise UpdateError(f'Required regular, non-symlink file is missing: {path}')
    return path.read_bytes()


def headings(text: str) -> list[tuple[int, int, str]]:
    """Return ATX headings outside fenced code, with exact character offsets."""
    result = []
    fence: Optional[tuple[str, int]] = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip('\r\n')
        fm = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', stripped)
        if fence is None:
            if fm:
                fence = (fm.group(1)[0], len(fm.group(1)))
            else:
                hm = re.match(r'^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*$', stripped)
                if hm:
                    title = re.sub(r'[ \t]+#+[ \t]*$', '', hm.group(2)).strip()
                    result.append((offset, len(hm.group(1)), title))
        elif fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] and not fm.group(2).strip():
            fence = None
        offset += len(line)
    return result


def license_headings(text: str) -> list[tuple[int, int, str]]:
    return [h for h in headings(text) if h[2].casefold() in {'license', 'licence', 'licensing'}]


def plan(root: Path) -> tuple[Path, bytes, bytes]:
    """Validate inputs and compute the exact candidate; never write files."""
    readme = root / 'README.md'
    raw = read_regular(readme)
    try:
        meta = json.loads(read_regular(root / 'licensing' / 'PACKAGE-METADATA.json'))
        snippet_raw = read_regular(root / 'licensing' / 'README-LICENSE-SECTION.md')
        text = raw.decode('utf-8')
        snippet = snippet_raw.decode('utf-8').replace('\r\n', '\n')
    except (UnicodeError, ValueError) as exc:
        raise UpdateError(f'Invalid UTF-8 or metadata JSON: {exc}') from exc
    if not isinstance(meta, dict) or meta.get('repository') not in REPOSITORIES:
        raise UpdateError('The metadata does not identify a supported Athena repository.')
    expected = meta.get('reviewed_readme_blob')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{40}', expected):
        raise UpdateError('The metadata has no valid reviewed README blob hash.')
    if '\r' in text.replace('\r\n', ''):
        raise UpdateError('Bare CR line endings are not supported; merge manually.')
    if '\r\n' in text and '\n' in text.replace('\r\n', ''):
        raise UpdateError('Mixed README line endings; merge manually.')
    newline = '\r\n' if '\r\n' in text else '\n'
    snippet = snippet.strip('\n') + '\n'
    if not snippet.startswith('## License\n') or snippet.count(START) != 1 or snippet.count(END) != 1:
        raise UpdateError('The supplied section does not contain one valid managed License block.')
    if snippet.index(START) >= snippet.index(END) or not snippet.rstrip().endswith(END):
        raise UpdateError('The supplied managed section has invalid marker order or trailing content.')
    replacement = snippet.replace('\n', newline)
    n_start, n_end = text.count(START), text.count(END)
    if n_start or n_end:
        if (n_start, n_end) != (1, 1) or text.index(START) >= text.index(END):
            raise UpdateError('Incomplete, duplicated, or reversed managed markers; merge manually.')
        # Permit unrelated edits after adoption, but never overwrite an altered notice.
        hs = license_headings(text)
        if len(hs) != 1 or hs[0][1:] != (2, 'License'):
            raise UpdateError('The managed notice has a missing or conflicting License heading.')
        begin = hs[0][0]
        stop = text.index(END) + len(END)
        if text[begin:stop] != replacement.rstrip('\r\n'):
            raise UpdateError('The managed licensing section differs from the supplied text; merge manually.')
        return readme, raw, raw
    actual = blob_hash(raw)
    if actual != expected:
        raise UpdateError(f'README has changed since review (expected {expected}, found {actual}). '
                          'Merge licensing/README-LICENSE-SECTION.md manually; no files changed.')
    hs = license_headings(text)
    if meta['repository'] == 'igorbarshteyn/athena':
        if len(hs) != 1 or hs[0][1:] != (2, 'License'):
            raise UpdateError('Expected exactly one level-two License heading in the baseline README.')
        begin = hs[0][0]
        following = [h[0] for h in headings(text) if h[0] > begin and h[1] <= 2]
        stop = min(following) if following else len(text)
        old = text[begin:stop].replace('\r\n', '\n')
        if old.split('\n', 1)[1].strip() != LEGACY:
            raise UpdateError('The baseline License body differs from the reviewed legacy declaration.')
        updated = text[:begin] + replacement + (newline if stop < len(text) else '') + text[stop:]
    else:
        if hs:
            raise UpdateError('The overlay already contains a licensing heading; merge manually.')
        separator = '' if not text else (newline if text.endswith(newline) else newline * 2)
        updated = text + separator + replacement
    return readme, raw, updated.encode('utf-8')


def apply_update(path: Path, old: bytes, new: bytes) -> Path:
    """Back up exact original bytes, then atomically replace README only."""
    if read_regular(path) != old:
        raise UpdateError('README changed after validation; no edit applied.')
    backup_dir = path.parent / '.licensing-backups'
    if backup_dir.is_symlink():
        raise UpdateError('Refusing a symlink backup directory.')
    backup_dir.mkdir(mode=0o700, exist_ok=True)
    if not backup_dir.is_dir():
        raise UpdateError('The backup path is not a directory.')
    backup = backup_dir / f'README.md.{blob_hash(old)}.bak'
    if backup.is_symlink():
        raise UpdateError('Refusing a symlink backup file.')
    if backup.exists():
        if read_regular(backup) != old:
            raise UpdateError('An existing backup with this name has different contents.')
    else:
        try:
            with backup.open('xb') as out:
                out.write(old)
                out.flush()
                os.fsync(out.fileno())
        except FileExistsError as exc:
            raise UpdateError('Backup was created concurrently; review and retry.') from exc
        os.chmod(backup, 0o600)
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(prefix='.README.licensing-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(new)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, mode)
        if read_regular(path) != old:
            raise UpdateError('README changed during preparation; backup retained, no edit applied.')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--apply', action='store_true', help='back up and edit the local README only')
    action.add_argument('--check', action='store_true', help='exit zero only when the managed block matches')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent.parent,
                        help='local checkout root; defaults to the parent of licensing/')
    args = parser.parse_args(argv)
    try:
        path, old, new = plan(args.repo_root.resolve())
        if old == new:
            print('README already contains the exact prepared licensing section. No changes.')
            return 0
        if args.check:
            print('README licensing section is not yet applied.', file=sys.stderr)
            return 1
        if args.apply:
            backup = apply_update(path, old, new)
            print(f'Updated {path}. Original backup: {backup}')
            print('Review the diff. No Git commit, push, runtime change, or legal certification was performed.')
        else:
            sys.stdout.writelines(difflib.unified_diff(old.decode().splitlines(keepends=True),
                                                      new.decode().splitlines(keepends=True),
                                                      fromfile='README.md (current)',
                                                      tofile='README.md (proposed)'))
            print('\nPreview only. No files were written. Use --apply after reviewing the proposal.')
        return 0
    except (UpdateError, OSError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
