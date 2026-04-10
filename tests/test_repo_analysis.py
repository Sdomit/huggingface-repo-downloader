from __future__ import annotations

from types import SimpleNamespace

from hf_downloader.models import RepoClassification, RepoRef
from hf_downloader.repo_analysis import classify_repo, mode_description, select_paths_for_mode


def make_info(**kwargs):
    defaults = {
        "library_name": None,
        "pipeline_tag": None,
        "tags": [],
        "cardData": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_classify_diffusers_pipeline_image_model() -> None:
    classification = classify_repo(
        RepoRef(repo_type="model", repo_id="stabilityai/sdxl"),
        make_info(library_name="diffusers", pipeline_tag="text-to-image", tags=["diffusers", "text-to-image"]),
        [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "scheduler/scheduler_config.json",
        ],
    )

    assert classification.primary_label == "Diffusers Pipeline"
    assert classification.package_kind == "diffusers"
    assert "Image Model" in classification.detected_labels


def test_classify_single_checkpoint_text_model() -> None:
    classification = classify_repo(
        RepoRef(repo_type="model", repo_id="org/text-model"),
        make_info(pipeline_tag="text-generation", tags=["text-generation"]),
        [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
        ],
    )

    assert classification.primary_label == "Single-File Checkpoint"
    assert classification.package_kind == "single_checkpoint"
    assert classification.modality_label == "Text Model"


def test_select_paths_for_modes_filters_preview_and_training_files() -> None:
    classification = RepoClassification(primary_label="Single-File Checkpoint", package_kind="single_checkpoint", modality_label="Image Model")
    paths = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preview.png",
        "checkpoint-1000/optimizer.bin",
        "README.md",
    ]

    minimal = select_paths_for_mode(paths, classification, "minimal")
    recommended = select_paths_for_mode(paths, classification, "recommended")
    full = select_paths_for_mode(paths, classification, "full")

    assert "model.safetensors" in minimal
    assert "config.json" in minimal
    assert "preview.png" not in recommended
    assert "checkpoint-1000/optimizer.bin" not in recommended
    assert full == sorted(paths)


def test_select_paths_for_diffusers_minimal_focuses_on_runtime_files() -> None:
    classification = RepoClassification(primary_label="Diffusers Pipeline", package_kind="diffusers", modality_label="Image Model")
    paths = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer.json",
        "README.md",
        "example.png",
    ]

    minimal = select_paths_for_mode(paths, classification, "minimal")

    assert "model_index.json" in minimal
    assert "unet/diffusion_pytorch_model.safetensors" in minimal
    assert "example.png" not in minimal


def test_mode_description_mentions_requested_mode() -> None:
    classification = RepoClassification(primary_label="LoRA", package_kind="lora", modality_label="Image Model")
    assert "Minimal" in mode_description("minimal", classification)
    assert "Recommended" in mode_description("recommended", classification)
    assert "Full" in mode_description("full", classification)
