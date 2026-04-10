from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import RepoClassification, RepoRef

CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt", ".bin", ".pt", ".pth"}
GGUF_EXTENSIONS = {".gguf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
DOC_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".cpp", ".c", ".sh"}
MANIFEST_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "setup.py",
    "setup.cfg",
    "manifest.json",
    "nodes.py",
    "__init__.py",
}
LICENSE_NAMES = {"license", "license.md", "license.txt", "copying", "copying.md"}
README_NAMES = {"readme.md", "readme.txt", "readme.markdown"}
TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
DIFFUSERS_PREFIXES = {
    "scheduler/",
    "tokenizer/",
    "tokenizer_2/",
    "text_encoder/",
    "text_encoder_2/",
    "text_encoder_3/",
    "unet/",
    "transformer/",
    "vae/",
    "vae_decoder/",
    "vae_encoder/",
    "safety_checker/",
    "feature_extractor/",
    "image_encoder/",
}
TRAINING_PATTERNS = (
    "optimizer",
    "trainer_state",
    "training_args",
    "events.out.tfevents",
    "global_step",
    "checkpoint-",
)


def classify_repo(repo: RepoRef, info: Any, file_paths: list[str]) -> RepoClassification:
    lower_paths = [path.lower() for path in file_paths]
    file_names = {Path(path).name.lower() for path in lower_paths}
    tags = {str(tag).lower() for tag in (getattr(info, "tags", None) or [])}
    repo_name = repo.repo_id.lower()
    library_name = str(getattr(info, "library_name", "") or "").lower()
    pipeline_tag = str(getattr(info, "pipeline_tag", "") or "").lower()
    license_name = _extract_license_name(getattr(info, "cardData", None))
    has_readme = any(Path(path).name.lower() in README_NAMES for path in lower_paths)

    diffusers = (
        library_name == "diffusers"
        or "diffusers" in tags
        or "model_index.json" in file_names
        or any(path.startswith(prefix) for path in lower_paths for prefix in DIFFUSERS_PREFIXES)
    )
    gguf = any(Path(path).suffix.lower() in GGUF_EXTENSIONS for path in lower_paths)
    lora = (
        "lora" in repo_name
        or any("lora" in tag for tag in tags)
        or "adapter_config.json" in file_names
        or "adapter_model.safetensors" in file_names
        or "adapter_model.bin" in file_names
    )
    controlnet = "controlnet" in repo_name or any("controlnet" in tag for tag in tags) or any("controlnet" in path for path in lower_paths)
    vae = "vae" in repo_name or any(Path(path).name.lower().startswith("vae") or "/vae" in path for path in lower_paths)
    text_encoder = any("text_encoder" in path for path in lower_paths)
    comfyui = (
        "comfyui" in repo_name
        or any("custom_nodes/" in path for path in lower_paths)
        or ("nodes.py" in file_names and "__init__.py" in file_names)
    )
    workflow = any(Path(path).name.lower() in {"workflow.json", "api.json"} or "workflow" in Path(path).name.lower() for path in lower_paths)
    code_count = sum(Path(path).suffix.lower() in CODE_EXTENSIONS or Path(path).name.lower() in MANIFEST_NAMES for path in lower_paths)
    weight_files = [path for path in lower_paths if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS | GGUF_EXTENSIONS]
    checkpoint_like = any(Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS for path in lower_paths)
    single_checkpoint = checkpoint_like and not diffusers and not lora and not gguf
    source_repo = code_count >= 3 and len(weight_files) <= 1

    image_model = pipeline_tag in {"text-to-image", "image-to-image", "inpainting", "image-classification"} or any(
        marker in tags or marker in pipeline_tag
        for marker in {"text-to-image", "image-to-image", "inpainting", "image-generation", "stable-diffusion"}
    )
    text_model = pipeline_tag in {"text-generation", "text2text-generation", "fill-mask", "feature-extraction"} or any(
        marker in tags or marker in pipeline_tag
        for marker in {"text-generation", "causal-lm", "llm", "gguf", "text2text-generation", "sentence-transformers"}
    )
    audio_model = pipeline_tag.startswith("audio") or any("audio" in tag for tag in tags)

    if repo.repo_type == "dataset":
        return _build_classification(
            primary_label="Dataset Repo",
            package_kind="dataset",
            modality_label="Dataset",
            detected_labels=["Dataset Repo"],
            license_name=license_name,
            has_readme=has_readme,
        )
    if repo.repo_type == "space":
        labels = ["Space App"]
        if comfyui or workflow:
            labels.append("ComfyUI Workflow/Project")
        return _build_classification(
            primary_label="Space App",
            package_kind="space",
            modality_label="App",
            detected_labels=labels,
            license_name=license_name,
            has_readme=has_readme,
        )

    detected_labels: list[str] = []
    modality_label = "General Model"
    if image_model:
        modality_label = "Image Model"
        detected_labels.append("Image Model")
    elif text_model or gguf:
        modality_label = "Text Model"
        detected_labels.append("Text Model")
    elif audio_model:
        modality_label = "Audio Model"
        detected_labels.append("Audio Model")

    primary_label = "Model Repo"
    package_kind = "model_repo"
    if comfyui and workflow:
        primary_label = "ComfyUI Workflow/Project"
        package_kind = "comfyui_project"
        detected_labels.append("ComfyUI Workflow/Project")
    elif comfyui:
        primary_label = "ComfyUI Custom Node Repo"
        package_kind = "comfyui_custom_node"
        detected_labels.append("ComfyUI Custom Node Repo")
    elif lora:
        primary_label = "LoRA"
        package_kind = "lora"
        detected_labels.append("LoRA")
    elif controlnet:
        primary_label = "ControlNet"
        package_kind = "controlnet"
        detected_labels.append("ControlNet")
    elif vae and not diffusers:
        primary_label = "VAE"
        package_kind = "vae"
        detected_labels.append("VAE")
    elif text_encoder and not diffusers:
        primary_label = "Text Encoder"
        package_kind = "text_encoder"
        detected_labels.append("Text Encoder")
    elif diffusers:
        primary_label = "Diffusers Pipeline"
        package_kind = "diffusers"
        detected_labels.append("Diffusers Pipeline")
    elif gguf:
        primary_label = "GGUF Collection"
        package_kind = "gguf"
        detected_labels.append("GGUF")
    elif single_checkpoint:
        primary_label = "Single-File Checkpoint"
        package_kind = "single_checkpoint"
        detected_labels.append("Single-File Checkpoint")
    elif source_repo:
        primary_label = "Full Source Repo"
        package_kind = "source_repo"
        detected_labels.append("Full Source Repo")

    if vae and "VAE" not in detected_labels:
        detected_labels.append("VAE")
    if controlnet and "ControlNet" not in detected_labels:
        detected_labels.append("ControlNet")
    if text_encoder and "Text Encoder" not in detected_labels:
        detected_labels.append("Text Encoder")
    if gguf and "GGUF" not in detected_labels:
        detected_labels.append("GGUF")

    return _build_classification(
        primary_label=primary_label,
        package_kind=package_kind,
        modality_label=modality_label,
        detected_labels=detected_labels,
        license_name=license_name,
        has_readme=has_readme,
    )


def select_paths_for_mode(file_paths: list[str], classification: RepoClassification, mode: str) -> list[str]:
    lower_paths = [path.lower() for path in file_paths]
    mapping = {path.lower(): path for path in file_paths}
    selected: set[str]

    if mode == "full":
        selected = set(lower_paths)
    elif mode == "minimal":
        selected = _minimal_selection(lower_paths, classification)
    else:
        selected = _recommended_selection(lower_paths, classification)

    if not selected:
        selected = set(lower_paths)
    return sorted(mapping[path] for path in selected if path in mapping)


def mode_description(mode: str, classification: RepoClassification) -> str:
    if mode == "minimal":
        return f"Minimal: smallest inference-focused set for {classification.primary_label.lower()}."
    if mode == "full":
        return "Full: every file in the repo."
    return "Recommended: inference files plus configs, excludes obvious previews and training artifacts."


def _minimal_selection(file_paths: list[str], classification: RepoClassification) -> set[str]:
    if classification.package_kind == "diffusers":
        return {
            path
            for path in file_paths
            if _is_inference_file(path)
            and (
                Path(path).name.lower() == "model_index.json"
                or any(path.startswith(prefix) for prefix in DIFFUSERS_PREFIXES)
                or _is_root_config(path)
            )
        }
    if classification.package_kind in {"lora", "controlnet", "vae", "text_encoder"}:
        return {
            path
            for path in file_paths
            if _is_inference_file(path)
            and (
                "adapter_" in Path(path).name.lower()
                or "config" in Path(path).name.lower()
                or Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS
                or _is_tokenizer_related(path)
            )
        }
    if classification.package_kind == "gguf":
        ggufs = [path for path in file_paths if Path(path).suffix.lower() in GGUF_EXTENSIONS]
        if len(ggufs) <= 1:
            return set(ggufs)
        preferred = sorted(ggufs, key=_gguf_priority_key)
        return {preferred[0]}
    if classification.package_kind == "single_checkpoint":
        checkpoints = [path for path in file_paths if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS]
        preferred = [path for path in checkpoints if _is_primary_checkpoint_candidate(path)]
        pick_from = preferred or checkpoints
        selected = set(sorted(pick_from, key=_checkpoint_priority_key)[:1])
        selected.update(path for path in file_paths if _is_root_config(path) or _is_tokenizer_related(path))
        return selected
    if classification.package_kind in {"comfyui_project", "comfyui_custom_node", "source_repo", "space"}:
        return {
            path
            for path in file_paths
            if Path(path).suffix.lower() in CODE_EXTENSIONS
            or Path(path).name.lower() in MANIFEST_NAMES
            or Path(path).name.lower() in LICENSE_NAMES
        }
    return _recommended_selection(file_paths, classification)


def _recommended_selection(file_paths: list[str], classification: RepoClassification) -> set[str]:
    if classification.package_kind in {"dataset"}:
        return set(file_paths)
    selected = {
        path
        for path in file_paths
        if not _is_preview_media(path)
        and not _is_training_artifact(path)
        and not _is_noise_file(path)
    }
    if classification.package_kind == "diffusers":
        return {
            path
            for path in selected
            if _is_inference_file(path)
            or Path(path).name.lower() in README_NAMES | LICENSE_NAMES
        }
    if classification.package_kind == "gguf":
        return {path for path in selected if Path(path).suffix.lower() in GGUF_EXTENSIONS or Path(path).name.lower() in README_NAMES | LICENSE_NAMES}
    if classification.package_kind in {"single_checkpoint", "lora", "controlnet", "vae", "text_encoder"}:
        return {
            path
            for path in selected
            if _is_inference_file(path)
            or Path(path).name.lower() in README_NAMES | LICENSE_NAMES
        }
    return selected


def _build_classification(
    *,
    primary_label: str,
    package_kind: str,
    modality_label: str,
    detected_labels: list[str],
    license_name: str,
    has_readme: bool,
) -> RepoClassification:
    deduped_labels = []
    for label in detected_labels:
        if label not in deduped_labels:
            deduped_labels.append(label)
    if modality_label not in deduped_labels and modality_label not in {"General", "General Model", "App", "Dataset"}:
        deduped_labels.append(modality_label)
    summary = f"Detected as {primary_label}"
    if modality_label and modality_label not in {"General", "General Model"}:
        summary += f" | {modality_label}"
    summary += " | heuristic"
    return RepoClassification(
        primary_label=primary_label,
        package_kind=package_kind,
        modality_label=modality_label,
        detected_labels=deduped_labels,
        summary=summary,
        license_name=license_name,
        has_readme=has_readme,
    )


def _extract_license_name(card_data: Any) -> str:
    if card_data is None:
        return ""
    for key in ("license_name", "license"):
        value = getattr(card_data, key, None)
        if value:
            return str(value)
    if isinstance(card_data, dict):
        return str(card_data.get("license_name") or card_data.get("license") or "")
    return ""


def _is_primary_checkpoint_candidate(path: str) -> bool:
    lower = Path(path).name.lower()
    blocked = ("optimizer", "ema", "lora", "vae", "text_encoder", "controlnet", "adapter")
    return not any(marker in lower for marker in blocked)


def _checkpoint_priority_key(path: str) -> tuple[int, int, str]:
    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    suffix_rank = {".safetensors": 0, ".ckpt": 1, ".bin": 2, ".pt": 3, ".pth": 4}.get(suffix, 9)
    top_level = 0 if "/" not in path else 1
    return (suffix_rank, top_level, name)


def _gguf_priority_key(path: str) -> tuple[int, str]:
    name = Path(path).name.lower()
    preferred_markers = ["q8", "q6", "q5", "q4_k_m", "q4"]
    for rank, marker in enumerate(preferred_markers):
        if marker in name:
            return (rank, name)
    return (len(preferred_markers), name)


def _is_inference_file(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    return (
        suffix in CHECKPOINT_EXTENSIONS | GGUF_EXTENSIONS | CONFIG_EXTENSIONS
        or name in README_NAMES | LICENSE_NAMES | TOKENIZER_NAMES
        or _is_tokenizer_related(path)
        or Path(path).name.lower() == "model_index.json"
    )


def _is_root_config(path: str) -> bool:
    lower = path.lower()
    return "/" not in lower and (Path(lower).suffix in CONFIG_EXTENSIONS or Path(lower).name in README_NAMES | LICENSE_NAMES)


def _is_tokenizer_related(path: str) -> bool:
    lower = path.lower()
    return "tokenizer" in lower or Path(lower).name in TOKENIZER_NAMES


def _is_preview_media(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _is_training_artifact(path: str) -> bool:
    lower = path.lower()
    return any(pattern in lower for pattern in TRAINING_PATTERNS)


def _is_noise_file(path: str) -> bool:
    name = Path(path).name.lower()
    return name in {".gitattributes", ".gitignore"}
