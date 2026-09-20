# Licensing scope — Athena

**Project:** `igorbarshteyn/athena`  
**License:** Athena Noncommercial, Attribution and Responsible Deployment License 2.0  
**Reviewed pre-adoption commit:** `e747c0362c4e1c3b95756a57154d09b8df7811ce`  
**Preparation date:** September 19, 2026 (UTC)

## What the new license covers

The accompanying [LICENSE](LICENSE) is offered only for the licensor-controlled, copyrightable original contributions intentionally published with this license, including future such contributions. It may coexist with earlier grants in pre-existing material; it does not extinguish those grants. The project name identifies the work, not a claim of ownership in every dependency.

This scope document accompanies an authorized adoption; preparing it locally is not itself publication. The first authorized commit supplying this scope document and the matching license is the adoption point. A permanent release record should then identify that commit. The pre-adoption hash above is an observed snapshot, **not** a fabricated first-restricted or last-permissive commit.

## Reviewed boundary

MIT declaration found in the current README; preserve earlier permissions.

The review covered current repository metadata, tree/README and relevant installer information, together with selected primary licensing sources. It did not audit all historical releases, privately shared archives, contributors, individual lines of source, or installed binaries. Assertions of copyright and exclusivity must remain limited to rights actually controlled. See [LICENSING-TRANSITION.md](LICENSING-TRANSITION.md).

## Path-level scope guide

| Path/category | Treatment |
|---|---|
| `patches/talk-llama/talk-llama.cpp` and adapted build/helpers | Mixed-origin material. Preserve the whisper.cpp/MIT rights and notices; the new terms cover only authorized original contributions and do not withdraw earlier MIT grants. |
| Athena-specific memory/helpers, orchestration, native TTS, setup and diagnostic scripts | Original licensor-controlled expression may be offered under the new terms. No claim is made that every line was independently authored or lacks earlier permissions. |
| `README.md`, project documentation and original prompts/configuration | New authorized expression can be covered. Preserve quoted, incorporated, historical or separately licensed material and earlier grants. |
| Downloaded `llama.cpp/`, `whisper.cpp/`, ONNX Runtime and system GPU stack | Independently licensed; not converted to Athena NCARD. |
| Models, personal memory, recordings, icon/media and third-party material | Not presumed owned or cleared by the root license; apply the separate rights and factual provenance. |

## Explicit exclusions and independent rights

Third-party works remain subject to their own terms. Earlier MIT/Apache or other valid permissions remain available. User data and ordinary model outputs are not covered merely because Athena handles or creates them. Independent implementations and unprotectable ideas, interfaces and functionality are not swept into this license by similarity.

This package does not claim a complete line-level copyright map. When preparing a distribution, preserve the more specific source notices and establish any unclear rights rather than relying on this guide as an ownership certificate. A later edited scope document cannot retroactively capture independently licensed material.

## Relationship to the other repository

Apply each repository's actual licenses and notices to the contributions it supplies. A newly restricted overlay does not make an old baseline noncommercial by itself. Conversely, the baseline's MIT history does not automatically grant MIT rights in later independently restricted overlay contributions. An installed combined tree must retain both sets of applicable notices and all upstream terms.
