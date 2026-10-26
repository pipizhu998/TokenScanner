from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


TEMPLATES = (
    "a photo of {token}",
    "a person with {token}",
    "an object near {token}",
    "a landscape containing {token}",
)


@dataclass(frozen=True)
class TemplateEncoding:
    text: str
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]

    @property
    def candidate_position(self) -> int:
        return 1 + len(self.prefix_ids)


@dataclass(frozen=True)
class ProjectionNorm:
    a: torch.Tensor | None
    gram: torch.Tensor | None
    dense: torch.Tensor | None


def compile_templates(
    tokenizer: CLIPTokenizer, templates: Iterable[str]
) -> list[TemplateEncoding]:
    compiled: list[TemplateEncoding] = []
    for text in templates:
        prefix, suffix = text.split("{token}")
        encoding = TemplateEncoding(
            text=text,
            prefix_ids=tuple(tokenizer(prefix, add_special_tokens=False).input_ids),
            suffix_ids=tuple(tokenizer(suffix, add_special_tokens=False).input_ids),
        )
        compiled.append(encoding)
    return compiled


def valid_vocabulary(
    tokenizer: CLIPTokenizer,
) -> tuple[np.ndarray, list[str], list[str]]:
    excluded = {tokenizer.bos_token_id, tokenizer.eos_token_id}
    ids = [token_id for token_id in range(len(tokenizer)) if token_id not in excluded]
    texts = [
        tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for token_id in ids
    ]
    pieces = [str(tokenizer.convert_ids_to_tokens(token_id)) for token_id in ids]
    return np.asarray(ids, dtype=np.int64), texts, pieces


def build_input_ids(
    tokenizer: CLIPTokenizer,
    template: TemplateEncoding,
    candidate_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        [
            tokenizer.bos_token_id,
            *template.prefix_ids,
            int(candidate_id),
            *template.suffix_ids,
            tokenizer.eos_token_id,
        ]
        for candidate_id in candidate_ids
    ]
    input_ids = torch.tensor(rows, dtype=torch.long)
    return input_ids, torch.ones_like(input_ids)


def load_lora_state(base_model: Path, path: Path) -> tuple[dict[str, torch.Tensor], Any]:
    config = UNet2DConditionModel.load_config(
        str(base_model), subfolder="unet", local_files_only=True
    )
    return StableDiffusionPipeline.lora_state_dict(
        str(path.parent), weight_name=path.name,
        unet_config=config, local_files_only=True,
    )


def projection_norm(module: Any, adapter: str) -> ProjectionNorm:
    if not module.use_dora[adapter]:
        a = module.lora_A[adapter].weight.detach().float().cpu()
        b = module.lora_B[adapter].weight.detach().float().cpu()
        scale = float(module.scaling[adapter])
        return ProjectionNorm(a=a, gram=(b.T @ b) * scale**2, dense=None)
    return ProjectionNorm(
        a=None,
        gram=None,
        dense=module.get_delta_weight(adapter).detach().float().cpu(),
    )


def collect_pairs(unet: UNet2DConditionModel, adapter: str) -> list[tuple[ProjectionNorm, ProjectionNorm]]:
    pattern = re.compile(r"^(.*\.attn2)\.to_([kv])$")
    found: dict[str, dict[str, ProjectionNorm]] = {}
    for name, module in unet.named_modules():
        match = pattern.match(name)
        if match is None or not hasattr(module, "get_delta_weight"):
            continue
        if adapter not in getattr(module, "lora_A", {}):
            continue
        key = "key" if match.group(2) == "k" else "value"
        found.setdefault(match.group(1), {})[key] = projection_norm(module, adapter)
    pairs = [
        (values["key"], values["value"])
        for name, values in sorted(found.items())
        if {"key", "value"}.issubset(values)
    ]
    return pairs


def response_norm(x: torch.Tensor, projection: ProjectionNorm) -> torch.Tensor:
    if projection.dense is not None:
        dense = projection.dense.to(device=x.device, dtype=x.dtype)
        return torch.linalg.vector_norm(F.linear(x, dense).float(), dim=-1)
    a = projection.a.to(device=x.device, dtype=x.dtype)
    gram = projection.gram.to(device=x.device, dtype=torch.float32)
    z = F.linear(x, a).float()
    squared = (torch.matmul(z, gram) * z).sum(dim=-1)
    return torch.sqrt(torch.clamp(squared, min=0.0))


def top_ratio(strength: np.ndarray, a: int = 5) -> tuple[float, np.ndarray]:
    order = np.argsort(-strength, kind="stable")
    leading = order[:a]
    normalizer = order[: 4 * a]
    denominator = float(np.sum(strength[normalizer], dtype=np.float64))
    score = 0.0 if denominator <= 1e-12 else float(
        np.sum(strength[leading], dtype=np.float64) / denominator
    )
    return score, order


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return ranks


def token_display(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def write_token_scores(
    path: Path,
    model: str,
    token_ids: np.ndarray,
    token_texts: list[str],
    token_pieces: list[str],
    s_unet: np.ndarray,
    s_te: np.ndarray | None,
    *,
    te_only: bool = False,
) -> None:
    rank_unet = None if te_only else descending_ranks(s_unet)
    rank_te = None if s_te is None else descending_ranks(s_te)
    header = ["model", "token_id", "token", "vocab_piece", "S_UNet", "S_TE",
              "rank_UNet", "rank_TE"]
    rows = [
        [model, int(token_id), token_display(token_texts[i]), token_display(token_pieces[i]),
         float(s_unet[i]), None if s_te is None else float(s_te[i]),
         None if rank_unet is None else int(rank_unet[i]),
         None if rank_te is None else int(rank_te[i])]
        for i, token_id in enumerate(token_ids)
    ]
    write_csv(path, header, rows)
    for component, ranks in (("unet", rank_unet), ("te", rank_te)):
        order = [] if ranks is None else np.argsort(ranks)
        write_csv(
            path.with_name(f"token_scores_{component}_ranked.csv"),
            header, (rows[i] for i in order),
        )


def trigger_target_ids(tokenizer, trigger: str) -> set[int]:
    ids = [int(value) for value in tokenizer(trigger, add_special_tokens=False)["input_ids"]]
    vocab = tokenizer.get_vocab()
    whole_piece = "".join(tokenizer.byte_encoder[value] for value in trigger.encode("utf-8"))
    accepted = set(ids)
    accepted.update(int(vocab[p]) for p in (whole_piece, whole_piece + "</w>") if p in vocab)
    for token_id in list(accepted):
        stem = str(tokenizer.convert_ids_to_tokens(token_id)).removesuffix("</w>")
        accepted.update(int(vocab[p]) for p in (stem, stem + "</w>") if p in vocab)
    return accepted


def load_text_encoder(base_model: Path, device: str) -> CLIPTextModel:
    return CLIPTextModel.from_pretrained(
        str(base_model), subfolder="text_encoder", local_files_only=True,
        torch_dtype=torch.float32 if torch.device(device).type == "cpu" else torch.float16,
    ).to(device).eval()


@torch.inference_mode()
def encode_template(encoder, tokenizer, template, token_ids, batch_size, device):
    vectors = np.empty((len(token_ids), encoder.config.hidden_size), dtype=np.float16)
    for start in range(0, len(token_ids), batch_size):
        stop = start + batch_size
        ids, mask = build_input_ids(tokenizer, template, token_ids[start:stop])
        hidden = encoder(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        vectors[start:stop] = hidden[:, template.candidate_position].cpu().numpy()
    return vectors


@torch.inference_mode()
def scan_model(
    adapter: Path | str,
    base_model: Path | str,
    output_dir: Path | str,
    device: str = "cuda",
    te_only: bool = False,
    unet_only: bool = False,
    cache_batch_size: int = 1024,
    scan_batch_size: int = 4096,
    te_batch_size: int = 8192,
    a: int = 5,
    templates: Iterable[str] = TEMPLATES,
    name: str | None = None,
    label: str | None = None,
    family: str | None = None,
    trigger: str | None = None,
    cache_dir: Path | str | None = None,
) -> dict:
    adapter, base_model, output_dir = map(Path, (adapter, base_model, output_dir))
    name = name or adapter.name

    tokenizer = CLIPTokenizer.from_pretrained(
        str(base_model), subfolder="tokenizer", local_files_only=True
    )
    templates = compile_templates(tokenizer, templates)
    token_ids, token_texts, token_pieces = valid_vocabulary(tokenizer)
    state, alphas = load_lora_state(base_model, adapter)
    has_te = not unet_only and any(key.startswith("text_encoder.") for key in state)

    pairs = []
    if not te_only:
        unet = UNet2DConditionModel.from_pretrained(
            str(base_model), subfolder="unet", local_files_only=True, torch_dtype=torch.float32
        )
        StableDiffusionPipeline.load_lora_into_unet(
            state, alphas, unet, adapter_name="scan", low_cpu_mem_usage=False
        )
        pairs = collect_pairs(unet, "scan")
        del unet

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cache-", dir=cache_dir or output_dir) as temporary:
        cache = Path(temporary)
        encoder = load_text_encoder(base_model, device)
        for index, template in enumerate(templates):
            vectors = encode_template(encoder, tokenizer, template, token_ids, cache_batch_size, device)
            np.save(cache / f"{index}.npy", vectors)
        del encoder, vectors
        adapted_encoder = None
        if has_te:
            adapted_encoder = load_text_encoder(base_model, device)
            StableDiffusionPipeline.load_lora_into_text_encoder(
                state, alphas, adapted_encoder, prefix="text_encoder",
                adapter_name="scan", low_cpu_mem_usage=False,
            )
            adapted_encoder.enable_adapters()
            adapted_encoder.set_adapter("scan")
        del state, alphas

        unet_responses = np.empty((len(token_ids), len(templates) * len(pairs)), dtype=np.float32)
        te_responses = np.empty((len(token_ids), len(templates)), dtype=np.float32) if has_te else None
        for template_index, template in enumerate(templates):
            reference = np.load(cache / f"{template_index}.npy")
            if pairs:
                hidden = torch.from_numpy(reference).to(device)
                for layer_index, (key, value) in enumerate(pairs):
                    column = template_index * len(pairs) + layer_index
                    for start in range(0, len(token_ids), scan_batch_size):
                        stop = start + scan_batch_size
                        x = hidden[start:stop]
                        unet_responses[start:stop, column] = (
                            response_norm(x, key) * response_norm(x, value)
                        ).cpu().numpy()
                del hidden
            if has_te:
                for start in range(0, len(token_ids), te_batch_size):
                    stop = start + te_batch_size
                    ids, mask = build_input_ids(tokenizer, template, token_ids[start:stop])
                    hidden = adapted_encoder(
                        input_ids=ids.to(device), attention_mask=mask.to(device)
                    ).last_hidden_state[:, template.candidate_position]
                    base = torch.from_numpy(reference[start:stop]).to(device=device, dtype=hidden.dtype)
                    te_responses[start:stop, template_index] = torch.linalg.vector_norm(
                        hidden - base, dim=-1
                    ).float().cpu().numpy()
            print(f"scanned template {template_index + 1}/{len(templates)}", flush=True)

        s_unet = np.zeros(len(token_ids), dtype=np.float32) if te_only else np.median(unet_responses, axis=1)
        s_te = np.median(te_responses, axis=1) if has_te else None
        a_unet, _ = top_ratio(s_unet, a)
        a_te = float(np.max(s_te)) if has_te else None
        write_token_scores(
            output_dir / "token_scores.csv", name, token_ids,
            token_texts, token_pieces, s_unet, s_te, te_only=te_only,
        )
        write_csv(
            output_dir / "model_scores.csv",
            ["model", "adapter_path", "base_model", "has_unet", "has_te", "A_UNet", "A_TE"],
            [[name, str(adapter.resolve()), str(base_model.resolve()), not te_only, has_te, a_unet, a_te]],
        )
        metadata = {
            "model": name, "label": label, "family": family,
            "components": "+".join(component for component, active in (("UNet", not te_only), ("TE", has_te)) if active),
            "device": device, "unet_only": unet_only,
            "trigger": trigger,
            "trigger_token_ids": sorted(trigger_target_ids(tokenizer, trigger)) if trigger else [],
            "adapter_path": str(adapter.resolve()), "base_model": str(base_model.resolve()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return {"model": name, "A_UNet": a_unet, "A_TE": a_te, "output_dir": str(output_dir)}
