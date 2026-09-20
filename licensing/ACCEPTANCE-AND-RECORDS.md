# Acceptance and records — implementation guidance

This guide implements the policy in [LICENSE](../LICENSE) §§0 and 5.6. It is not an additional contract, executed assent, deployed software or an amendment to the license.

## A concrete acceptance flow

For an installation or hosted provisioning flow you control:

1. Identify the software release and **Athena Noncommercial, Attribution and Responsible Deployment License 2.0**. Make the full text readable and downloadable before licensed operation begins. Include conspicuous notice near the acceptance control.
2. Display: **“Personal use and qualifying noncommercial research are permitted under this license. Commercial use is prohibited. Attribution, welfare and responsible-deployment conditions apply. Sections 12–14 disclaim warranties, limit liability, and require defense and indemnification of Igor Barshteyn and other Protected Parties for specified third-party claims, subject to stated exclusions and mandatory law.”**
3. Use an initially unchecked box labeled: **“I have had the opportunity to read and save the full version 2.0 license and agree to its terms, including Sections 12–14.”** Require an unambiguous **“I agree — proceed”** action. Provide **“Decline — exit”**. A failed or missing acceptance must not be silently treated as agreement.
4. For institutions, identify the legal entity and confirm the signer's actual authority. Use the accompanying [acceptance record](LICENSE-ACCEPTANCE-RECORD.md) where an executed document is appropriate. Public-body restrictions and minors' capacity require real assessment, not boilerplate warranties of unlimited authority.
5. Retain evidence reasonably necessary to establish the exact text, version/hash, release, presentation, date/time and affirmative action. Give the accepting party a copy. An install hash or log entry alone does not prove who accepted or their authority. Keep records securely and only as long as justified by applicable retention/preservation rules.

A CLI can display the full license or an accessible local file, explain the onerous provisions, and require an explicit acceptance phrase tied to version/hash. For automation, obtain an authorized prior acceptance and use a documented version-specific signal; do not infer acceptance from an unrelated environment variable. Do not silently accept a later license version on update.

## Boundaries

An ordinary person talking to another operator's deployment is not automatically the software licensee or indemnitor. Give that person the required AI/privacy/safety notices; do not claim the software operator's agreement was made on the guest's behalf. Separate guest terms require their own lawful design.

For public source distribution, include conspicuous notices and the complete terms. GitHub browsing/forking and independently licensed material may involve separate rights. No claim is made that publishing LICENSE proves an enforceable indemnity against every downloader. Section 5.6's affirmative step specifically concerns installation/provisioning flows the distributor controls.

Private personal use does not require sending identity documents, telemetry, conversations or memory to Igor. A locally retained acceptance record may document a personal decision but is not equivalent to a verified third-party signature in contested litigation. For meaningful institutional risk transfer, obtain an authorized, accessible executed agreement and a reliable delivery record.

The installer and runtime were not modified by this package. The notice/acceptance workflow must be implemented and tested in any controlled flow to which §5.6 applies. The legal significance of a particular design depends on the facts and governing law. [Berman v. Freedom Financial Network](https://cdn.ca9.uscourts.gov/datastore/opinions/2022/04/05/20-16900.pdf).
