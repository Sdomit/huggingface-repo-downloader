from hf_downloader.models import RepoRef
from hf_downloader.parsing import parse_repo_input


def test_parse_model_tree_url() -> None:
    parsed = parse_repo_input("https://huggingface.co/openai-community/gpt2/tree/main/onnx")
    assert parsed == RepoRef(
        repo_type="model",
        repo_id="openai-community/gpt2",
        requested_revision="main",
        deep_path="onnx",
    )


def test_parse_dataset_blob_url() -> None:
    parsed = parse_repo_input("https://huggingface.co/datasets/user/my-dataset/blob/main/data/train.jsonl")
    assert parsed == RepoRef(
        repo_type="dataset",
        repo_id="user/my-dataset",
        requested_revision="main",
        deep_path="data/train.jsonl",
    )


def test_parse_space_raw_reference() -> None:
    parsed = parse_repo_input("spaces/user/my-space")
    assert parsed == RepoRef(repo_type="space", repo_id="user/my-space")


def test_search_term_is_not_treated_as_repo_id() -> None:
    assert parse_repo_input("gpt2") is None


def test_parse_hf_uri() -> None:
    parsed = parse_repo_input("hf://datasets/user/my-data/folder/file.json")
    assert parsed == RepoRef(repo_type="dataset", repo_id="user/my-data", deep_path="folder/file.json")
