# Model licenses and deployment dependencies

**Snapshot: September 19, 2026. No model weights are included in this licensing package.** This inventory follows the baseline installer's model identifiers and the overlay's dependence on that baseline. It is not a representation of what is currently installed on any individual machine.

**Version 2.0 evidence update:** the current repository trees/READMEs and both installation scripts were rechecked on September 19. The three included MIT license blob identities and FunASR model agreement were rechecked; the Qwen license, Orpheus base-model metadata/maintainer clarification, Llama 3.2 agreement and emotion2vec card were also consulted. Remaining inventory entries are retained dependency leads from the supplied 1.0 package, not a new certification of every upstream version or redistribution bundle. See `licensing/SOURCE-REGISTER.md`.

Changing Athena's source license does not change a model license. Quantizing to GGUF, exporting to ONNX, changing precision, or renaming a file does not by itself replace upstream rights or notices. A model can impose conditions on use even when downloaded separately rather than distributed in the GitHub repository.

## Model register

| Component | Identified artifact / source | Licensing finding and action |
|---|---|---|
| Qwen3.5-397B-A17B | `unsloth/Qwen3.5-397B-A17B-GGUF`, `UD-Q3_K_XL`, five shards named `Qwen3.5-397B-A17B-UD-Q3_K_XL-0000N-of-00005.gguf` | The [official Qwen model LICENSE](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/LICENSE) is Apache 2.0. Preserve the applicable license, relevant NOTICE if supplied, attribution and conversion provenance. A live `main` download is not an immutable revision. |
| Whisper small.en | `ggerganov/whisper.cpp`, `ggml-small.en.bin` | OpenAI Whisper uses MIT. Preserve the [OpenAI notice](licenses/third-party/Whisper-OpenAI-MIT.txt) and any applicable conversion notices. [Official license](https://github.com/openai/whisper/blob/main/LICENSE). |
| Orpheus 3B 0.1 fine-tuned | `unsloth/orpheus-3b-0.1-ft-GGUF`, `orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf` | **License-chain clarification needed for the exact checkpoint.** Do not describe weights as Apache-only. See the evidence and release conditions below. |
| emotion2vec+ large | Output `models/emotion2vec_plus_large.onnx`; upstream model card identifies `emotion2vec/emotion2vec_plus_large` and `iic/emotion2vec_plus_large` | **Custom model terms; not simply MIT or Apache.** [Model card](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/main/README.md) links FunASR licensing. Review the exact checkpoint's [model agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE), including attribution, model naming, scope language and revisions. |
| SNAC 24 kHz decoder | `onnx-community/snac_24khz-ONNX`, `onnx/decoder_model_fp16.onnx`, installed as `orpheus/snac24_dynamic_fp16.onnx` | The [conversion repository](https://huggingface.co/onnx-community/snac_24khz-ONNX) identifies MIT and its upstream base model. Record both conversion and original-model provenance; obtain their actual required notices before redistributing weights. |
| Silero VAD | `ggml-org/whisper-vad`, `ggml-silero-v6.2.0.bin` | The [converted-model repository](https://huggingface.co/ggml-org/whisper-vad) identifies MIT. Preserve release-matched upstream and conversion notices. |
| Vision projector and any replacement vision/language model | The overlay contains vision code; the precise deployed projector and model are not established by this package's review. | **Not cleared by inference from Qwen's license.** Inventory the actual projector, tokenizer and model files and their revisions before distributing or enabling a deployment based on them. |
| Any substituted Qwen, GLM, abliterated model, adapter, voice or fine-tune | Not covered by the exact-artifact findings above. | Review independently. A familiar family name, quantizer badge, or previously reviewed version is not a license grant for a new artifact. |

## Orpheus: preserve the unresolved distinction

The published pretrained model metadata names `meta-llama/Llama-3.2-3B-Instruct` as the base while displaying Apache 2.0. An Orpheus maintainer's April 15, 2025 [clarification](https://github.com/canopyai/Orpheus-TTS/issues/29#issuecomment-2806874295) distinguishes Apache repository code from Llama-licensed model weights. The baseline Athena README already identifies Orpheus weights as Llama 3.2 community licensed. These observations are evidence of provenance and inconsistent labeling, not a definitive determination of every term for every quantized checkpoint. [Published model metadata](https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained/blob/main/README.md).

The [Llama 3.2 agreement](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt) includes its own distribution/use, notice, attribution and acceptable-use obligations. For an applicable distribution, its required agreement copy and Meta notice must accompany covered Llama material, and the required “Built with Llama” display must appear in an allowed prominent location. Do not assume noncommercial operation removes these conditions. Review the complete agreement and AUP, not only the attribution sentence.

**Before a packaged deployment:** identify the exact fine-tuned/quantized artifact and its immutable revision; capture the supplied license and base-model chain; resolve conflicting representations with the licensor or counsel; implement the resulting conditions. This package does not manufacture a corrected Orpheus license or grant rights in Meta/Canopy material. `llama.cpp` itself is a different, MIT-licensed software project.

## emotion2vec: the custom agreement needs its own assessment

The model card marks a custom `model-license`; the retrieved FunASR `MODEL_LICENSE` describes model weights and derivatives. Its English text includes a broad use/copy/modify/share clause together with reference-and-learning language, source/author and model-name obligations, conduct-related termination, and an asserted automatic revision mechanism. It also retains unfilled upstream governing-law wording. Do not silently repair or replace those upstream terms. [Model agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).

The noncommercial character of Athena's proposed license does not resolve every ambiguity in that agreement. Obtain a licensor clarification for a contemplated use where the scope is unclear. Do not call the model “unrestricted,” “Apache,” or automatically authorized for every nonprofit, public-facing or research deployment. Re-exporting it to ONNX does not independently clear the underlying rights.

## Capture at release time

For each actual artifact, record the upstream identifier, immutable revision, download location, local filename, SHA-256, original model and conversion author, license text/version, applicable AUP/NOTICE, conversion parameters, and whether it is shipped or separately fetched. Preserve a dated local copy of terms lawfully obtained with the download. Do not hard-code a guessed revision or hash; the installer currently contains moving model references.

This package's notice files cover certain software licenses only. It is not a complete model redistribution bundle. In particular, it does not include all model license/AUP texts, voice rights or training-data permissions. A source release can document dependencies without pretending that it redistributes or owns their weights.
