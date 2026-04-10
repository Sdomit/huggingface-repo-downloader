# HF Repo Downloader

HF Repo Downloader is a Windows-first PyQt6 desktop app for browsing Hugging Face repositories and downloading only the files you actually want.

It supports models, datasets, and spaces, lets you search by name or paste a Hugging Face URL directly, and gives you a queue-based download workflow with accurate byte progress, speed, ETA, and resumable downloads.

## Highlights

- Search Hugging Face repos or open them directly from pasted URLs
- Browse full repository trees for models, datasets, and spaces
- Select individual files, folders, or filtered file groups before download
- Detect repo/package type such as Diffusers, checkpoint, LoRA, GGUF, and ComfyUI-related repos
- Use guided download modes like `Minimal`, `Recommended`, and `Full`
- See total selected size and planned download size before queueing
- Resume interrupted downloads and skip files already available locally
- Manage a download queue with start, pause, resume, cancel, retry, reorder, and clear actions
- Use private or gated repositories with a Hugging Face token stored via keyring

## UI Features

- Fast search/open bar for repo IDs and Hugging Face links
- Repository details panel with revision picker, guided modes, and file/folder search
- Multi-select tree for nested folders and individual files
- Queue view with per-job progress, speed, ETA, and per-file task status
- Saved settings for default download location, worker count, retry count, and token source

## Install

```powershell
python -m pip install -e .[dev]
```

## Run

```powershell
python -m hf_downloader
```

Or on Windows:

```powershell
.\run_hf_downloader.bat
```

## Windows EXE

This repository builds a Windows executable automatically with GitHub Actions.

To download it:

1. Open the `Actions` tab on GitHub.
2. Select the latest `Build Windows EXE` workflow run.
3. Download the `HF-Repo-Downloader-Windows` artifact.
4. Extract the zip and run `HF Repo Downloader.exe`.

To build the executable locally on Windows:

```powershell
.\build_exe.bat
```

The local build output is written to:

```text
dist/HF Repo Downloader/HF Repo Downloader.exe
```

## Development

Run the test suite:

```powershell
pytest -q
```

## Project Structure

```text
hf_downloader/
  auth.py
  hf_service.py
  models.py
  progress.py
  queue_manager.py
  repo_analysis.py
  settings.py
  tree_ops.py
  ui.py
tests/
run_hf_downloader.bat
build_exe.bat
tools/build_exe.ps1
pyproject.toml
```

## Notes

- This project is currently optimized for Windows desktop use.
- The app uses the official `huggingface_hub` client for repo discovery, metadata, and downloads.
- GitHub repo description suggestion:
  `Desktop Hugging Face downloader for Windows with search, guided file selection, resumable downloads, queue management, and accurate progress.`
