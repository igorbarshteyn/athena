# Third-party software notices and provenance

**Review snapshot: September 19, 2026. This is a targeted inventory, not a complete software bill of materials or a line-by-line authorship audit.**

**Version 2.0 evidence update:** the current repository trees/READMEs and both installation scripts were rechecked on September 19. The three included MIT license blob identities and FunASR model agreement were rechecked; the Qwen license, Orpheus base-model metadata/maintainer clarification, Llama 3.2 agreement and emotion2vec card were also consulted. Remaining inventory entries are retained dependency leads from the supplied 1.0 package, not a new certification of every upstream version or redistribution bundle. See `licensing/SOURCE-REGISTER.md`.

The root [LICENSE](LICENSE) covers only rights the relevant licensor controls in Covered Material. It does not relicense upstream components. A path containing `athena` or a replacement file copied over an upstream example may still contain upstream protected expression. Preserve that expression's license and notices. The [MIT text](https://opensource.org/license/mit) permits broad reuse subject to its notice condition; a new root license does not erase that condition.

## Inventory and distribution boundaries

| Component or category | Observed relationship | License treatment and required action |
|---|---|---|
| whisper.cpp and its talk-llama example | The baseline installer pins commit `afa2ea544fb4b0448916b4a31ecd33c8685bd482`; Athena supplies replacement/adapted example files. | MIT. Preserve inherited notices in adapted files and include the [ggml MIT text](licenses/third-party/ggml-MIT.txt). [Pinned upstream license](https://github.com/ggml-org/whisper.cpp/blob/afa2ea544fb4b0448916b4a31ecd33c8685bd482/LICENSE). |
| llama.cpp / ggml | Baseline clones a moving upstream branch; inference code is independently licensed. The overlay also contains upstream-derived vision source. | MIT for the reviewed upstream project license. Check the actual version and individual embedded components, preserve applicable notices. [Upstream license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE). |
| Bundled vision and shared audio sources in the overlay | `patches/talk-llama/mtmd/` and `patches/whisper-common/` are present in the overlay, according to its README and tree. | Treat upstream-derived expression as an exception to the blanket Athena license. Determine exact source revisions and any nested helper/image-library licenses before making an exhaustive compliance claim. No mass header rewrite is included. |
| ONNX Runtime | Baseline downloads `onnxruntime-linux-x64-gpu_cuda12-1.27.0`; not supplied in this licensing package. | MIT at the reviewed upstream LICENSE. An actual redistributed runtime needs its own release-matched license and [ThirdPartyNotices.txt](https://github.com/microsoft/onnxruntime/blob/main/ThirdPartyNotices.txt), plus any other applicable notices. The included [MIT copy](licenses/third-party/ONNX-Runtime-MIT.txt) is not a substitute for those notices. |
| CUDA toolkit and runtime libraries | Separately installed system prerequisite in the baseline installer. | NVIDIA SDK terms; use the exact installed version's redistribution list and conditions. Do not assume every toolkit file can be bundled. [CUDA 12.9.1 EULA](https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html). |
| cuDNN | Separately installed NVIDIA library. | Applicable NVIDIA/cuDNN terms, not Athena NCARD. Verify redistributable components and conditions for the actual version. [Official EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html). |
| NVIDIA driver, firmware and kernel modules | System software, not included here. | The open kernel modules' dual GPL/MIT licensing does not relicense the entire user-space driver/firmware stack. Inspect actual package terms before redistribution. [NVIDIA explanation](https://developer.nvidia.com/blog/nvidia-releases-open-source-gpu-kernel-modules/). |
| SDL2, libcurl, audio utilities, C/C++ runtime and operating-system packages | Build/runtime dependencies identified by project documentation; not included in this package. | No complete version-by-version review was performed. Preserve the installed packages' copyright/license files and inspect their dependency closure before shipping binaries, containers or appliances. Do not infer the license of these packages from Athena's root file. |
| Python setup/export tooling and dependencies | Baseline's optional emotion2vec export installs packages including FunASR, PyTorch, torchaudio, ModelScope, ONNX and ONNX Runtime. | Setup-only status does not waive license conditions when those packages or copied portions are distributed. This is not a full inventory of their transitive dependencies. |
| Model weights, tokenizers, codecs and conversions | Downloaded separately or generated from upstream weights. | See [MODEL-LICENSES.md](MODEL-LICENSES.md); software and weight licenses can differ. |
| Images, icons, demo recordings and other media | Some are present or referenced by the baseline; ownership/provenance was not audited. | Do not assume a code license establishes voice, likeness, music, image or publicity rights. Resolve provenance separately. |

## What is actually included in this licensing package

The three MIT texts were transcribed from the retrieved upstream LICENSE contents and checked against the Git blob hashes returned by the connector:

| Included file | Retrieved Git blob SHA-1 |
|---|---|
| `licenses/third-party/ggml-MIT.txt` | `e7dca554bcb802f98408383a864404e3aa4eacca` |
| `licenses/third-party/ONNX-Runtime-MIT.txt` | `48bc6bb4996ac924359e8e28b9ae88970e5ed3fc` |
| `licenses/third-party/Whisper-OpenAI-MIT.txt` | `d25552598bb9c5400612159ed4bab92ce12a5ce5` |

The [Apache 2.0 text](licenses/third-party/Apache-2.0.txt) is a standard reference copy for Apache-governed dependencies, not an alternative license for Athena Covered Material. The upstream license notices applicable to an exact model or binary still need to accompany a distribution when required. No CUDA/cuDNN driver binaries, model weights, model AUPs, or complete ONNX third-party notice bundle are supplied here.

## Rules for source releases

Keep all existing third-party notices. Add author/modification attribution only for contributions whose provenance is known. A generic root license cannot replace copyright notices that must travel with copied or adapted source. The current package does not strip headers, relabel complete mixed-origin files, or change build/runtime code.

For the overlay, separately inventory its embedded vision/audio tree; do not rely solely on the licenses fetched by the baseline installer. Also ensure that an **installed or repackaged** mixed tree carries the overlay's license and notices. The reviewed overlay documentation says its repository README is not copied into the base tree; publishing these files does not automatically add notice propagation to that installer. This is a packaging task to verify before distributing installed trees, not a claim that ordinary local installation needs a code rewrite.

## Additional gates for binaries, containers and offline bundles

Before shipping anything beyond the source-only package, record the exact files included, their versions and hashes, all statically and dynamically linked components actually redistributed, and the complete associated license/notice obligations. Check compiler-runtime exceptions, copyleft compatibility where relevant, NVIDIA redistribution limits, and model agreements. Supply all required texts and notices with the delivered artifact, not only with its source repository.

This document does not conclude that all conceivable combinations are compatible. When a component cannot be distributed consistently with all applicable conditions, exclude it, obtain appropriate rights, or change the design. Athena's licensor cannot grant a waiver of someone else's terms.
