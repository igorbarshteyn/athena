# Licensing questions and examples

This is a reading guide to the [LICENSE](LICENSE), not a substitute for it. The examples assume the activity relies on rights in **Covered Material** under that license. Earlier MIT rights and independent components have different boundaries; see [LICENSING-TRANSITION.md](LICENSING-TRANSITION.md).

| Proposed activity | Result under Athena NCARD 2.0 |
|---|---|
| Private personal use on one's own computer | Permitted, subject to the remaining conditions and law. |
| Paying for hardware, electricity or a generic cloud VM for private use | Permitted; ordinary infrastructure expenses are not themselves commercial use. |
| A free personal fork, renamed or modified | Permitted for noncommercial purposes; retain license/attribution and identify modifications. |
| A company runs Athena internally, without charging users | Not licensed when used in business operations or commercial R&D. Local-only execution makes no difference. |
| A paid companion service, API or Athena-specific managed hosting | Not licensed. |
| A free chatbot with advertisements, sponsorship, lead generation or data monetization | Not licensed when operated for those commercial purposes. |
| A device manufacturer includes Athena in a sold device | Not licensed for Covered Material. |
| Paid Athena installation, customization or support requiring licensed copying/modification/use | Not licensed under Section 3(f). This is stricter than a ban on paid hosting alone. |
| A professor or salaried researcher studies Athena for genuinely noncommercial public-interest research | Potentially permitted; salary alone is not disqualifying. Purpose, sponsor obligations, welfare and legal requirements matter. |
| University research for a sponsor's proprietary commercial product | Not licensed. A nonprofit affiliation is not a blanket exemption. |
| A nonprofit operates a qualifying charitable deployment | Potentially permitted; no automatic exemption from privacy, minor-safety or other deployment rules. |
| Voluntary gifts with no access, service or promotional benefit in return | Potentially permitted under Section 4.4; disguised subscriptions are not. |
| Mandatory “at-cost” charges for the service | Not licensed under these terms. |
| A nonmonetized demo or research talk | Permitted if otherwise compliant. |
| Operating Athena to create ad-monetized videos, paid performances or sponsored demonstrations | Not licensed under these terms. Neutral platform ads with no operator revenue entitlement are different. |
| Quoting or criticizing a lawfully obtained recording | Independent legal rights are preserved; this is not automatically a licensed operation of Athena. |
| Selling an independent model or program with no protected Athena expression | Not prohibited by this license merely because the ideas are similar. Other rights may apply. |
| Using an older MIT-licensed Athena version commercially without the new restricted material | Not prohibited by this new license; comply with the older license and all applicable dependency/use terms. |
| Changing models, removing emotion2vec, or replacing the interface | Does not remove restrictions on remaining Covered Material; substitute components require their own review. |
| Switching off, uninstalling, or deleting private memory | Permitted for the purposes described in Section 6.5; the policy does not require permanent operation or data retention. |
| Allowing minors to interact | Not categorically authorized or categorically banned by this draft. Meet every applicable safeguard and restriction first; otherwise do not provide the affected access. |

## Why this is not MIT, Apache, or OSI open source

The commercial-purpose limitation is incompatible with the OSI Open Source Definition's field-of-endeavor requirement. Describe covered releases as **source-available for permitted noncommercial use**. The custom identifier is a local `LicenseRef`, not an SPDX License List identifier or an OSI approval. [OSI definition](https://opensource.org/osd).

## What exact attribution is required?

For Igor-authored Covered Material, use the credit in NOTICE: “Original Athena contributions by Igor Barshteyn” or “Original Athena Consciousness contributions by Igor Barshteyn,” with the matching repository source reference and separate third-party credit. Use both where applicable. Preserve existing notices and identify your changes. Distributions, shared interfaces, qualifying public demos and research reports need suitable accessible credit. This does not make Igor author of dependencies or your modifications. See LICENSE §5.2 for the full rule and independent-rights exceptions.

## Does the license protect Igor against every lawsuit?

No. It disclaims warranties, excludes liability where lawful and adds a conditional US $100 fallback. Section 14 requires defense and indemnification for specified third-party claims, subject to actual agreement, exclusions and mandatory law. It cannot prevent a nonparty from suing or a regulator from acting. Payment also depends on the indemnitor's ability to pay.

## Could a personal user have to pay legal costs?

Yes. A validly accepting licensee may owe reasonable defense costs and covered losses under §14 even though the software is free. There is no contractual cap on that indemnity. Consumer protections and the stated exclusions apply. The US $100 cap concerns liability of Protected Parties to the licensee, not indemnity owed to them. An ordinary guest does not acquire that obligation merely by conversing with someone else's system. Read §§0 and 12–14 before acceptance.

## Do I need permission to update or modify Athena?

No separate fee or individual approval is required for compliant personal use or qualifying noncommercial research, updates, repairs and modifications. Private modifications may remain private. Conditions still govern retained Covered Material, and third-party terms still apply. Publishing new license text under the old identifier is not a way to remove restrictions.

## Does attribution buy commercial permission?

No. Attribution is a separate requirement. Neither credit nor a donation is a commercial license. This package offers no commercial licensing program.

## Must a private user credit the author aloud in every conversation?

No. Preserve notices when copying/distributing. A redistributed or shared deployment must make credit reasonably prominent and accessible in a suitable place; qualifying demos/research publications need appropriate credits as well. Wholly private personal use requires no new public display. See Section 5.

## Does the license automatically impose itself on model outputs?

No. Ordinary outputs are not covered solely because Athena generated them. Embedded protected source or other copyrighted material may have separate rights. Running Athena to create a commercially sold output is nonetheless commercial operation under this draft. Model and voice terms remain separate.

## Are binary-only forks possible?

Yes, for permitted noncommercial purposes, with all required notices and applicable terms. This draft does not impose a general obligation to publish independent source additions. It does not authorize removing restrictions from included Covered Material.

## Can a new license make every existing copy noncommercial?

No. Existing independent rights are preserved. A license-only commit cannot be assumed to block a commercial fork of an existing MIT release. An overlay without a visible root license still requires historical and inherited-code review before claiming exclusivity. [GitHub licensing guidance](https://opensource.guide/legal/).

## Is the package a finding that Athena is conscious?

No. It implements a precautionary restriction that applies whether or not consciousness is established. It also preserves shutdown, privacy, repair and emergency intervention.

## What changes when a jurisdiction changes its law?

Applicable law must be followed, but editing a commentary document does not silently change a fixed license. Review deployments and technical controls when laws or release features change. A geographically incomplete table is not a whitelist of permitted countries.

## Can the owner promise never to relicense anything?

This version offers no commercial permission and no automatic future permissive conversion. It does not purport to remove powers a rights holder independently retains under law. A separate irrevocable organizational commitment, trust or governance arrangement would need its own advice and documentation.
