from __future__ import annotations

import json
import math

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


def _parse_bongard_temperatures(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Bongard temperatures.json must contain a JSON object")
    temps: dict[str, float] = {}
    for key, val in raw.items():
        kind, _, count = str(key).partition(":")
        if kind not in ("noul", "choice", "score") or (count != "" and (not count.isdigit() or int(count) < 2)):
            raise ValueError(f"Invalid Bongard temperature key: {key!r}")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"Invalid Bongard temperature value for {key!r}: {val!r}")
        fval = float(val)
        if not math.isfinite(fval) or fval <= 0.0:
            raise ValueError(f"Bongard temperature must be finite and positive for {key!r}: {val!r}")
        temps[str(key).replace(":", ".")] = fval
    return temps


@ModelBase.register("T5Gemma2ForConditionalGeneration", "T5Gemma2Model", "JudgmentModel")
@ModelBase.example("google/t5gemma-2-270m-270m")
class T5Gemma2Model(TextModel):
    """Convert the text-only portion of a T5Gemma2 encoder-decoder model."""

    model_arch = gguf.MODEL_ARCH.T5GEMMA2

    _BONGARD_HEAD_TENSORS = ("w.weight", "candidate.weight", "global_readout.weight")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not args:
            raise TypeError("T5Gemma2Model requires a model directory")

        dir_model = Path(args[0])
        supplied_hparams = kwargs.pop("hparams", None)
        root_hparams = ModelBase.load_hparams(dir_model, False) if supplied_hparams is None else supplied_hparams

        self.bongard_manifest: dict[str, Any] | None = None
        self.bongard_temperatures: dict[str, float] = {}
        self.bongard_marker_spellings: list[str] = []
        self.dir_tokenizer = dir_model

        is_bongard_bundle = (
            root_hparams.get("model_type") == "bongard"
            or "JudgmentModel" in (root_hparams.get("architectures") or ())
        )
        if is_bongard_bundle:
            self.bongard_manifest = root_hparams
            backbone_dir = dir_model / "backbone"
            if not (backbone_dir / "config.json").is_file():
                raise ValueError(f"Bongard bundle missing backbone/config.json in {dir_model}")
            root_hparams = ModelBase.load_hparams(backbone_dir, False, guess=False)
            if (dir_model / "tokenizer").is_dir():
                self.dir_tokenizer = dir_model / "tokenizer"
            temps_path = dir_model / "temperatures.json"
            if temps_path.is_file():
                self.bongard_temperatures = _parse_bongard_temperatures(temps_path)
            spellings = self.bongard_manifest.get("compiler", {}).get("spellings")
            if isinstance(spellings, dict) and "candidate_end" in spellings and "decision_end" in spellings:
                self.bongard_marker_spellings = [
                    str(spellings["candidate_end"]),
                    str(spellings["decision_end"]),
                ]
            else:
                self.bongard_marker_spellings = ["<unused0>", "<unused1>"]

        try:
            encoder_hparams = root_hparams["encoder"]["text_config"]
            decoder_hparams = root_hparams["decoder"]
        except (KeyError, TypeError) as error:
            raise ValueError("T5Gemma2 config must contain encoder.text_config and decoder") from error

        # TextModel expects a flat config; merge decoder fields at top level.
        flat_hparams = {**root_hparams, **decoder_hparams}
        flat_hparams["architectures"] = root_hparams["architectures"]
        kwargs["hparams"] = flat_hparams
        self.root_hparams = root_hparams
        self.encoder_hparams = encoder_hparams
        self.decoder_hparams = decoder_hparams
        self.shared_token_embeddings_found = False
        super().__init__(*args, **kwargs)
        self._validate_hparams()

        enc_n_layer = self.encoder_hparams["num_hidden_layers"]
        dec_n_layer = self.decoder_hparams["num_hidden_layers"]
        self.block_count = enc_n_layer
        if enc_n_layer != dec_n_layer:
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, max(enc_n_layer, dec_n_layer))

    def index_local_tensors(self, prefix: str) -> dict[str, Callable[[], Tensor]]:
        # Bongard bundles normally provide a root model.safetensors.index.json spanning backbone/ and head.safetensors.
        # Fall back to indexing backbone/ + head.safetensors directly if no root index exists.
        if (
            self.bongard_manifest is not None
            and not (self.dir_model / f"{prefix}.safetensors.index.json").is_file()
            and (self.dir_model / "backbone").is_dir()
        ):
            orig_dir = self.dir_model
            self.dir_model = orig_dir / "backbone"
            try:
                tensors = super().index_local_tensors(prefix)
            finally:
                self.dir_model = orig_dir
            head_path = self.dir_model / "head.safetensors"
            if head_path.is_file():
                tensors.update(super().index_local_tensors("head"))
            return tensors
        return super().index_local_tensors(prefix)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.startswith("decision."):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _validate_hparams(self) -> None:
        matching_keys = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "max_position_embeddings",
            "sliding_window",
            "layer_types",
            "rope_parameters",
            "query_pre_attn_scalar",
            "hidden_activation",
            "vocab_size",
        )
        mismatches = [
            key
            for key in matching_keys
            if self.encoder_hparams.get(key) != self.decoder_hparams.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"T5Gemma2 requires matching encoder and decoder configs; mismatched keys: {mismatches}"
            )

        for candidate in (
            self.decoder_hparams.get("decoder_start_token_id"),
            self.root_hparams.get("decoder_start_token_id"),
            self.decoder_hparams.get("bos_token_id"),
            self.root_hparams.get("bos_token_id"),
            self.encoder_hparams.get("bos_token_id"),
        ):
            if candidate is not None:
                self.decoder_start_token_id = int(candidate)
                break
        else:
            raise ValueError("T5Gemma2 requires decoder_start_token_id or bos_token_id")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith("encoder.text_model."):
            name = "model.encoder." + name.removeprefix("encoder.text_model.")
        elif name.startswith("encoder."):
            name = "model." + name
        elif name.startswith("decoder."):
            name = "model." + name
        if name.endswith(".eoi_embedding") or name == "eoi_embedding":
            logger.info("Skipping text-only unsupported EOI embedding")
            return None
        return super().filter_tensors((name, gen))

    def set_vocab(self) -> None:
        orig_dir = self.dir_model
        self.dir_model = self.dir_tokenizer
        try:
            if (self.dir_model / "tokenizer.model").is_file():
                self._set_vocab_sentencepiece()
                self.gguf_writer.add_add_space_prefix(False)
                return

            vocab = gguf.LlamaHfVocab(self.dir_model)
            tokens: list[bytes] = []
            scores: list[float] = []
            toktypes: list[int] = []
            for text, score, toktype in vocab.all_tokens():
                tokens.append(text)
                scores.append(score)
                toktypes.append(toktype)

            self.gguf_writer.add_tokenizer_model("gemma4")
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_scores(scores)
            self.gguf_writer.add_token_types(toktypes)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
            special_vocab.add_to_gguf(self.gguf_writer)
            self.gguf_writer.add_add_space_prefix(False)
        finally:
            self.dir_model = orig_dir

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.decoder_hparams
        layer_types = hparams["layer_types"]

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_decoder_block_count(hparams["num_hidden_layers"])
        self.gguf_writer.add_decoder_start_token_id(self.decoder_start_token_id)
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([lt == "sliding_attention" for lt in layer_types])

        scales = self.bongard_manifest.get("embedding_scales") if self.bongard_manifest is not None else None
        if scales is None:
            scales = self.root_hparams.get("bongard_embedding_scales")
        if scales is not None:
            if not isinstance(scales, dict) or "encoder" not in scales or "decoder" not in scales:
                raise ValueError("embedding_scales must contain encoder and decoder values")
            enc_scale = float(scales["encoder"])
            dec_scale = float(scales["decoder"])
            if not math.isfinite(enc_scale) or enc_scale <= 0.0 or enc_scale != dec_scale:
                raise ValueError("Text export requires equal positive finite encoder/decoder embedding scales")
            embedding_scale = enc_scale
        else:
            embedding_scale = math.sqrt(hparams["hidden_size"])
        self.gguf_writer.add_embedding_scale(embedding_scale)

        self.gguf_writer.add_attention_scale(1.0 / math.sqrt(hparams["query_pre_attn_scalar"]))
        self.gguf_writer.add_hidden_act(hparams["hidden_activation"])
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])
        self.gguf_writer.add_rope_dimension_count_swa(hparams["head_dim"])

        if (attn_softcap := hparams.get("attn_logit_softcapping")) is not None:
            self.gguf_writer.add_attn_logit_softcapping(attn_softcap)
        if (final_softcap := hparams.get("final_logit_softcapping")) is not None:
            self.gguf_writer.add_final_logit_softcapping(final_softcap)

        head_present = [t in self.model_tensors for t in self._BONGARD_HEAD_TENSORS]
        if any(head_present):
            if not all(head_present):
                missing = [t for t, ok in zip(self._BONGARD_HEAD_TENSORS, head_present) if not ok]
                raise ValueError(f"Incomplete Bongard judgment head; missing tensors: {missing}")
            self.gguf_writer.add_decision_type(gguf.DecisionType.BONGARD)
            labels = self.bongard_marker_spellings or ["<unused0>", "<unused1>"]
            self.gguf_writer.add_decision_labels(labels)
            for key, temp in sorted(self.bongard_temperatures.items()):
                self.gguf_writer.add_decision_temperature(key, temp)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in ("model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "shared.weight"):
            if not self.shared_token_embeddings_found:
                name = "shared.weight"
                self.shared_token_embeddings_found = True
            else:
                return

        # HF T5Gemma2 RMSNorm computes x * (1 + weight).
        if name.endswith("norm.weight"):
            data_torch = data_torch.float() + 1.0

        yield from super().modify_tensors(data_torch, name, bid)
