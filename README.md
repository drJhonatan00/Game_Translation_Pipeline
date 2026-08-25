# Game Translation Pipeline
A high-performance, offline game translation framework built in Python. Powered by CTranslate2 (INT8 quantization) and Helsinki-NLP neural models, this tool automates multi-language translation for game scripts while strictly protecting low-level syntax, variables, and memory pointers.

# Neural Game Localization Pipeline

A high-performance, offline game translation framework built in Python. Powered by **CTranslate2 (INT8 quantization)** and **Helsinki-NLP neural models**, this tool automates context-aware, multi-language translation for game scripts while strictly protecting low-level engine syntax, variables, and memory pointers.

While optimized to parse raw text dumps extracted from Nintendo 3DS ROMs and `.garc` archives (such as Pokémon Gen 6/7 titles), this pipeline is **fully game-agnostic**. It operates universally on any plain text (`.txt`) script containing custom control codes or embedded engine variables.

---

## Key Features

* **Zero-Breakage Variable Protection:** Utilizes a Regex AST-style parsing engine to mask game variables (`[VAR ...]`), memory index pointers (`[~ 0]`), and escape characters (`\c`, `\n`, `/a`). Placeholders are safely restored post-translation to prevent ROM crashes.
* **Hardware-Accelerated Inference:** Powered by `CTranslate2` with `INT8` quantization for fast, low-RAM offline processing. Automatically detects and leverages CUDA (NVIDIA GPUs) when available.
* **Automated Safety Fallback:** Rigorously verifies tag integrity post-unmasking. If a neural model drops or corrupts a control code, the tool discards the translated line and retains the original to guarantee file stability.
* **Dynamic Multi-Language Support:** Interactive CLI menu providing one-click setup and on-demand model conversion for **9 target languages**:
  * Portuguese
  * Spanish
  * French
  * Italian
  * German
  * Japanese
  * Chinese (Simplified — `cmn_Hans`)
  * Chinese (Traditional — `cmn_Hant`)
  * Korean
* **Stream I/O Alignment:** Preserves exact leading/trailing whitespaces and native OS line endings (`\r\n` / `\n`) to prevent buffer overflows in target game UI boxes.

---

### Prerequisites

* **Python 3.10+**
* **NVIDIA GPU** (Optional, for CUDA acceleration)

---

### Legal & Disclaimer
No Game Assets Included: This repository contains only source code for text processing. It does not distribute ROMs, `.garc` archives, copyrighted game scripts, or Nintendo proprietary assets.

Utility Software: Users are responsible for providing their own legally obtained text dumps.
