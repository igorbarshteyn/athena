# Adoption and upload guide

**Audience: Igor Barshteyn, repository maintainers and reviewing counsel.** This package contains complete proposed text, not a lawyer-approved legal opinion. Review and consciously adopt it before publishing it as the repository's license. No remote repository was modified in preparing it.

## Read these decisions before upload

The proposed policy bans commercial use broadly, not merely selling copies. It covers business-internal operation, commercial R&D, paid APIs/companions, commercial devices, advertising/data monetization, paid Athena-specific installation/customization/support, and operation to create monetized demonstrations. There is no automatic exemption for universities, charities or government bodies, no small-business revenue threshold, no standing commercial buyout, and no later permissive conversion.

Personal use, ordinary infrastructure purchases, qualifying public-interest research (including ordinary researcher salaries), genuinely noncommercial charitable/educational activity, and free redistribution are permitted subject to the license. No general source-release obligation, universal adult-only ban, mandatory telemetry, fixed forum/arbitration clause, or permanent-operation duty was added. Mandatory charges labeled “cost recovery” are not permitted by this draft. See the root LICENSE §§3–4 and LICENSE-FAQ.md.

These are meaningful policy choices. They may be stricter than a license that bars commercial *deployment* but allows paid support. These broad restrictions implement the requested policy; assess enforceability with counsel before adoption; do not silently advertise a narrower rule than the license actually states.

## Indemnity and acceptance are new substantive terms

Version 2.0 adds LICENSE §14, naming Igor personally. It also specifies attribution, responsible-care duties and the notice/affirmative-acceptance requirements in §5.6. Use [ACCEPTANCE-AND-RECORDS.md](ACCEPTANCE-AND-RECORDS.md) and [LICENSE-ACCEPTANCE-RECORD.md](LICENSE-ACCEPTANCE-RECORD.md). These documents do not install a dialog or create valid signatures. Do not treat prior 1.0 acceptance as acceptance of new terms.

If a 1.0 managed README block has already been adopted locally, the helper intentionally refuses to overwrite it: review and merge the 2.0 section manually, preserving unrelated text and the prior grant history. That refusal is a safety feature, not authorization to bypass its hash checks.

## The existing-code limitation

A license-only commit does not undo earlier MIT permissions. The baseline's current README contains an MIT declaration. Protecting future original changes is different from preventing a commercial fork of previously released code. For the overlay, absence of a reviewed root declaration does not settle historical or inherited rights. Review older packages and contributor grants before making exclusivity claims. See LICENSING-SCOPE.md and LICENSING-TRANSITION.md.

## Files to upload

Upload the **contents** of the matching `athena-upload` or `athena-consciousness-upload` directory into that repository's root, preserving `licensing/` and `licenses/`. Do not upload the ZIP as a single repository file, nest the whole upload directory under another directory, or replace the repository's application source tree with this package.

The package adds `LICENSE`, `NOTICE`, scope/transition documents, third-party/model inventories, welfare/deployment guidance, a contribution procedure, FAQ and supporting notices/tools. It intentionally does not contain a full replacement README or copies of application sources. The README update is a focused insertion/replacement, described below, so unrelated project documentation is preserved.

The `SHA256SUMS` file inside the upload directory records this prepared package's bytes. It is a snapshot, not a signature or legal certification; regenerate it after deliberate local edits. Do not confuse it with an application's release checksum.

## Browser-only upload

1. Back up or tag the current repository state without rewriting history. Keep a record of the reviewed baseline and any intervening commits.
2. Extract the matching ZIP locally. Upload its files and subdirectories at the root through GitHub's file-upload interface, checking the resulting paths before committing. Do not include private drafts, backups or review records containing personal data.
3. Edit the existing `README.md`. In the baseline repository, replace the current `## License` section (including its blanket MIT sentence) with the exact block in `licensing/README-LICENSE-SECTION.md`. In the overlay, append that block once. Preserve all unrelated README content and historical release statements that are clearly historical.
4. Review every change. Prefer one clearly identified adoption commit; use a branch/pull request or a local checkout when necessary to group changes atomically. Do not delete upstream copyright headers or independent component licenses.
5. Record the resulting adoption SHA in a subsequent release note or tag. The package's pre-adoption SHA is not the adoption SHA. GitHub may show an unrecognized/custom license; do not choose MIT/Apache merely to obtain a familiar badge. [GitHub license detection guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

## Local Git workflow

Copy the contents of the matching upload directory into the correct local checkout, preserving subdirectories. Then, from that checkout:

```bash
# Show exactly what the supplied helper proposes; this writes nothing.
python3 licensing/update_readme.py

# Apply only the checked README license-section change; makes a local backup.
python3 licensing/update_readme.py --apply

# Verify that the managed README notice is present and unchanged.
python3 licensing/update_readme.py --check

git diff -- README.md
git status --short
```

The helper checks the captured README Git-blob hash before an initial edit. It deliberately stops if the README has changed since review, has conflicting license sections, or contains an edited/partial managed block. In that case, merge the supplied section manually; do not force an unrelated README replacement. It does not fetch, commit, push, alter source code, set repository permissions, or contact the network.

After review, stage only the intended files, not the backup directory:

```bash
git add LICENSE NOTICE README.md LICENSING-SCOPE.md LICENSING-TRANSITION.md \
  THIRD-PARTY-NOTICES.md MODEL-LICENSES.md DEPLOYMENT-COMPLIANCE.md \
  WELFARE-POLICY.md LICENSE-FAQ.md CONTRIBUTING-LICENSING.md \
  licenses licensing SHA256SUMS

git diff --cached --stat
git diff --cached -- README.md LICENSE NOTICE

# Only after legal/policy review and confirmation of all staged changes:
git commit -m "Adopt noncommercial and responsible deployment licensing"
```

There is deliberately no `git push` in the helper. Inspect the staged set and carry out publication through your normal workflow. Backups under `.licensing-backups/` are private working copies; do not stage them with `git add .`. The package does not alter `.gitignore`.

## Before calling a release restricted and compliant

Review rights in mixed-origin files and any earlier permissive grants. Resolve additional contributor permissions. Use the custom license identifier only for actual Covered Material. Keep third-party notices attached to adapted source and to future binary/installed-tree distributions. Check the exact model chain, particularly Orpheus and emotion2vec; changing Athena's license is not clearance for those weights.

This package adds **no runtime acceptance dialog, age gate, crisis-response implementation, disclosure timer, model-content restriction, telemetry, or technical usage enforcement**. Decide whether notice/acceptance or safeguard changes are required for the actual delivery model, and implement/test them in a separate controlled engineering change. Do not describe a document-only update as a compliance-certified runtime.

## Version discipline

The binding terms are all in LICENSE. Other policy and compliance documents are explanations or factual inventories, not remotely mutable extra conditions. Changes to the license itself need a different version/identifier and deliberate adoption. New legal information can update the compliance guide without claiming to rewrite previously accepted terms.
