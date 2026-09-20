#!/usr/bin/env python3
"""Fixture tests for update_readme.py; no network, live checkout or license adoption."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

MODULE = Path(__file__).with_name('update_readme.py')
spec = importlib.util.spec_from_file_location('update_readme', MODULE)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

SNIPPET = ('## License\n\n' + u.START + '\nPrepared test notice.\n' + u.END + '\n')


class ReadmeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'licensing').mkdir()
        (self.root / 'licensing/README-LICENSE-SECTION.md').write_text(SNIPPET)

    def setup_readme(self, text=None, overlay=False):
        if text is None:
            text = '# Project\n\nDescription.\n\n## License\n\n' + u.LEGACY + '\n'
        self.raw = text.encode()
        (self.root / 'README.md').write_bytes(self.raw)
        meta = {'repository': 'igorbarshteyn/athena-consciousness' if overlay else 'igorbarshteyn/athena',
                'reviewed_readme_blob': u.blob_hash(self.raw)}
        (self.root / 'licensing/PACKAGE-METADATA.json').write_text(json.dumps(meta))

    def call(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = u.main(['--repo-root', str(self.root), *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_git_blob_hash(self):
        self.assertEqual(u.blob_hash(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')

    def test_dry_run_does_not_write(self):
        self.setup_readme()
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        code, out, err = self.call()
        self.assertEqual(code, 0, err)
        self.assertIn('Preview only', out)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_apply_backups_and_check(self):
        self.setup_readme()
        self.assertEqual(self.call('--apply')[0], 0)
        backup = self.root / f'.licensing-backups/README.md.{u.blob_hash(self.raw)}.bak'
        self.assertEqual(backup.read_bytes(), self.raw)
        self.assertEqual(self.call('--check')[0], 0)
        self.assertEqual((self.root/'README.md').read_text(), '# Project\n\nDescription.\n\n'+SNIPPET)

    def test_idempotence(self):
        self.setup_readme()
        self.call('--apply')
        before = (self.root/'README.md').read_bytes()
        self.assertEqual(self.call('--apply')[0], 0)
        self.assertEqual((self.root/'README.md').read_bytes(), before)
        self.assertEqual(len(list((self.root/'.licensing-backups').iterdir())), 1)

    def test_overlay_append(self):
        self.setup_readme('# Overlay\n\nKeep all this.\n', overlay=True)
        self.assertEqual(self.call('--apply')[0], 0)
        self.assertEqual((self.root/'README.md').read_bytes(), self.raw+b'\n'+SNIPPET.encode())

    def test_overlay_without_final_newline(self):
        self.setup_readme('# Overlay\nKeep', overlay=True)
        self.call('--apply')
        self.assertEqual((self.root/'README.md').read_bytes(), self.raw+b'\n\n'+SNIPPET.encode())

    def test_changed_readme_refused(self):
        self.setup_readme()
        (self.root/'README.md').write_bytes(self.raw+b'Later edit\n')
        self.assertEqual(self.call('--apply')[0], 2)
        self.assertFalse((self.root/'.licensing-backups').exists())

    def test_check_not_adopted(self):
        self.setup_readme()
        self.assertEqual(self.call('--check')[0], 1)

    def test_partial_marker_refused(self):
        self.setup_readme('# Project\n'+u.START+'\n', overlay=True)
        self.assertEqual(self.call('--apply')[0], 2)

    def test_duplicate_markers_refused(self):
        self.setup_readme('# Project\n'+SNIPPET+SNIPPET, overlay=True)
        self.assertEqual(self.call('--apply')[0], 2)

    def test_reversed_markers_refused(self):
        self.setup_readme('# Project\n'+u.END+'\n'+u.START+'\n', overlay=True)
        self.assertEqual(self.call('--apply')[0], 2)

    def test_modified_managed_block_refused(self):
        self.setup_readme('# Project\n'+SNIPPET.replace('Prepared', 'Altered'), overlay=True)
        self.assertEqual(self.call('--apply')[0], 2)

    def test_duplicate_license_heading_refused(self):
        self.setup_readme('# Project\n## License\n'+u.LEGACY+'\n## License\nOther\n')
        self.assertEqual(self.call('--apply')[0], 2)

    def test_code_fence_heading_ignored(self):
        prefix = '# Project\n```markdown\n## License\nexample\n```\n\n'
        self.setup_readme(prefix+'## License\n\n'+u.LEGACY+'\n')
        self.assertEqual(self.call('--apply')[0], 0)
        self.assertTrue((self.root/'README.md').read_text().startswith(prefix))

    def test_trailing_sections_preserved(self):
        tail = '## Support\n\nDo not edit this.\n'
        self.setup_readme('# Project\n\n## License\n\n'+u.LEGACY+'\n\n'+tail)
        self.assertEqual(self.call('--apply')[0], 0)
        self.assertTrue((self.root/'README.md').read_text().endswith(tail))

    def test_changed_legacy_body_refused(self):
        self.setup_readme('# Project\n\n## License\n\nA different agreement.\n')
        self.assertEqual(self.call('--apply')[0], 2)

    def test_crlf_preserved(self):
        self.setup_readme(('# Project\n\n## License\n\n'+u.LEGACY+'\n').replace('\n','\r\n'))
        self.assertEqual(self.call('--apply')[0], 0)
        data = (self.root/'README.md').read_bytes()
        self.assertNotIn(b'\n', data.replace(b'\r\n', b''))
        self.assertEqual(self.call('--check')[0], 0)

    def test_mixed_newlines_refused(self):
        self.setup_readme('# Project\r\n\nNo license\n', overlay=True)
        self.assertEqual(self.call('--apply')[0], 2)

    def test_file_mode_preserved(self):
        self.setup_readme()
        os.chmod(self.root/'README.md', 0o755)
        self.call('--apply')
        self.assertEqual(stat.S_IMODE((self.root/'README.md').stat().st_mode), 0o755)

    def test_symlink_readme_refused(self):
        self.setup_readme()
        (self.root/'README.md').rename(self.root/'original.md')
        (self.root/'README.md').symlink_to(self.root/'original.md')
        self.assertEqual(self.call('--apply')[0], 2)

    def test_symlink_backup_dir_refused(self):
        self.setup_readme()
        other = self.root/'elsewhere'
        other.mkdir()
        (self.root/'.licensing-backups').symlink_to(other, target_is_directory=True)
        self.assertEqual(self.call('--apply')[0], 2)
        self.assertEqual((self.root/'README.md').read_bytes(), self.raw)

    def test_conflicting_backup_refused(self):
        self.setup_readme()
        (self.root/'.licensing-backups').mkdir()
        (self.root/f'.licensing-backups/README.md.{u.blob_hash(self.raw)}.bak').write_bytes(b'wrong')
        self.assertEqual(self.call('--apply')[0], 2)
        self.assertEqual((self.root/'README.md').read_bytes(), self.raw)

    def test_unrelated_post_adoption_edit_allowed(self):
        self.setup_readme()
        self.call('--apply')
        data = (self.root/'README.md').read_bytes()
        (self.root/'README.md').write_bytes(b'Additional intro\n'+data)
        self.assertEqual(self.call('--check')[0], 0)

    def test_wrong_repository_refused(self):
        self.setup_readme()
        p=self.root/'licensing/PACKAGE-METADATA.json'
        m=json.loads(p.read_text()); m['repository']='someone/else'; p.write_text(json.dumps(m))
        self.assertEqual(self.call('--apply')[0], 2)

    def test_metadata_invalid_json_refused(self):
        self.setup_readme()
        (self.root/'licensing/PACKAGE-METADATA.json').write_text('{bad')
        self.assertEqual(self.call('--apply')[0], 2)

    def test_symlink_snippet_refused(self):
        self.setup_readme()
        s=self.root/'licensing/README-LICENSE-SECTION.md'
        s.rename(s.with_name('other.md')); s.symlink_to(s.with_name('other.md'))
        self.assertEqual(self.call('--apply')[0], 2)

    def test_already_licensed_overlay_refused(self):
        self.setup_readme('# Overlay\n## Licensing\nOther license.\n', overlay=True)
        self.assertEqual(self.call('--apply')[0], 2)

    def test_no_later_files_touched(self):
        self.setup_readme()
        (self.root/'application.cpp').write_bytes(b'unchanged application')
        self.call('--apply')
        self.assertEqual((self.root/'application.cpp').read_bytes(), b'unchanged application')


if __name__ == '__main__':
    unittest.main(verbosity=2)
