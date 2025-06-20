
# GexT/MixTex 模型转换工具 gext_converted

本项目提供了一个命令行工具，用于将 Hugging Face 上的 `MixTex` 和 `GexT` 模型转换为适用于高效推理的格式（ONNX 和 GGUF），并对模型进行量化。项目环境和依赖由 `uv` 进行管理。

## ✨ 功能特性

-   **MixTex 模型转换**:
    -   将 `Swin` 编码器导出为优化的 **ONNX** 格式。
    -   将解码器转换为 **GGUF** 格式并进行量化。
-   **GexT 模型转换**:
    -   将语言模型部分和视觉多模态投影（mmproj）转换为 **GGUF** 格式。
    -   支持对语言模型进行量化。


## ⚙️ 环境准备

1.  **安装 uv**:
    如果你还没有安装 `uv`，请参照 [官方指南](https://github.com/astral-sh/uv) 进行安装。

2.  **准备二进制文件**:
    本项目依赖 `llama.cpp` 的量化工具。请确保已根据你的操作系统，将预编译的二进制文件（如 `llama-quantize` 或 `llama-quantize.exe`）放置在项目根目录下的 `bin` 文件夹中。

    ```
    .
    ├── bin/
    │   ├── llama-quantize       # (适用于 Linux)
    │   └── llama-quantize.exe   # (适用于 Windows)
    ├── main.py
    └── pyproject.toml
    ```

3.  **同步 Python 环境**:
    在项目根目录下运行以下命令，`uv` 将根据 `pyproject.toml` 文件自动创建虚拟环境并安装所有依赖。

    ```bash
    uv sync
    ```

## 🚀 使用方法

使用 `uv run` 命令执行转换脚本。所有模型文件将首先从 Hugging Face 下载并缓存到本地的 `./hf_models` 目录中，以加速后续执行。

### 示例 1: 转换 MixTex 模型

此命令将从 Hugging Face 下载 `MixTex/base_ZhEn`，将其编码器转换为 ONNX，解码器转换为 GGUF (Q4_K_M 量化)，并保存到 `./mixtex_converted` 目录。

```bash
uv run python main.py --hf_model_path MixTex/base_ZhEn --model_type mixtex --output_dir ./mixtex_converted
```

### 示例 2: 转换 GexT 模型

此命令将下载 `MosRat/GexT_V1`，将其解码器和 `mmproj` 转换为 GGUF 格式（解码器进行 Q4_K_M 量化），并保存到 `./gext_converted` 目录。

```bash
uv run python main.py --hf_model_path MosRat/GexT_V1 --model_type gext --output_dir ./gext_converted
```

### 参数说明

-   `--hf_model_path` (必需): 指定 Hugging Face Hub 上的模型路径。
-   `--model_type` (必需): 指定模型类型，可选值为 `mixtex` 或 `gext`。
-   `--output_dir`: 指定转换后模型的输出目录 (默认为 `./converted_models`)。
-   `--quant_type`: 指定 GGUF 的量化类型 (默认为 `Q4_K_M`)。
-   `--debug`: 启用调试模式，会打印更详细的日志信息。