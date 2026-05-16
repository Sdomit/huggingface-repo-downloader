<div align="center">

# 🤗 HF Repo Downloader

**Desktop Hugging Face downloader for Windows — search, select, and download only what you need.**

[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows)]()
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📖 What It Does

HF Repo Downloader is a PyQt6 desktop app for browsing Hugging Face repositories and downloading only the files you actually want — with a queue, real byte-level progress, speed and ETA, and full resume support.

Supports models, datasets, and spaces. Search by name or paste a Hugging Face URL directly.

---

## ✨ Highlights

- 🔍 Search by repo name or paste any Hugging Face URL directly
- 🌲 Browse full repository trees — models, datasets, and spaces
- ☑️ Select individual files, entire folders, or filtered file groups
- 🏷️ Detect repo type: Diffusers, checkpoint, LoRA, GGUF, ComfyUI-related, and more
- 🚀 Guided download modes: **Minimal**, **Recommended**, and **Full**
- 📊 See total selected size before queueing anything
- ♻️ Resume interrupted downloads; skip files already present locally
- 📋 Queue management with start, pause, resume, cancel, retry, and reorder
- 🔐 Private and gated repos via a Hugging Face token stored securely in keyring

---

## 🚀 Quick Start

**Option A — Windows EXE (no Python needed)**

1. Open the **Actions** tab on GitHub
2. Select the latest `Build Windows EXE` workflow run
3. Download the `HF-Repo-Downloader-Windows` artifact
4. Extract and run `HF Repo Downloader.exe`

**Option B — Run from source**

```powershell
python -m pip install -e .[dev]
python -m hf_downloader
```

Or use the one-click launcher:
```powershell
.\run_hf_downloader.bat
```

**Build the EXE locally:**
```powershell
.\build_exe.bat
# Output: dist/HF Repo Downloader/HF Repo Downloader.exe
```

---

## 📦 Requirements (source only)

- Python 3.11+
- PyQt6
- `huggingface_hub` (official client — used for all repo discovery and downloads)

---

## 🗂️ Project Structure

```
hf_downloader/
  auth.py            Hugging Face token management via keyring
  hf_service.py      Repo search, metadata, and file listing
  models.py          Data models for repos, files, queue items
  progress.py        Byte-level download progress tracking
  queue_manager.py   Download queue — state machine, workers
  repo_analysis.py   Repo type detection and guided mode suggestions
  settings.py        Saved preferences (download path, workers, retries)
  tree_ops.py        File tree selection logic
  ui.py              PyQt6 application
tests/
```

---

## 🧪 Development

```powershell
pytest -q
```

---

## 📄 License

MIT
