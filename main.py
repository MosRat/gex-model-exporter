import modeling_swin as org
from modeling_swin import SwinSelfAttention
import transformers.models
import transformers.models.swin.modeling_swin
from transformers.models.swin import modeling_swin

# 使用patch 的swin定义
modeling_swin.SwinSelfAttention = org.SwinSelfAttention
modeling_swin.SwinAttention = org.SwinAttention
modeling_swin.SwinModel = org.SwinModel
modeling_swin.SwinBackbone = org.SwinBackbone
modeling_swin.SwinStage = org.SwinStage
modeling_swin.SwinIntermediate = org.SwinIntermediate

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForCausalLM,
)

import subprocess
import torch
import onnx
import onnxruntime
import numpy as np
from onnxsim import simplify
import torch.onnx
import warnings
from pathlib import Path
import argparse
import sys

warnings.filterwarnings("ignore")


def run_command(cmd, debug=False, message="", success_message=""):
    """
    运行 shell 命令并打印进度。
    参数:
        cmd (list): 命令及其参数的列表。
        debug (bool): 如果为 True，则打印子进程的标准输出和标准错误。
        message (str): 运行命令前要打印的消息。
        success_message (str): 成功完成时要打印的消息。
    返回:
        bool: 如果命令成功则为 True，否则为 False。
    """
    if message:
        print(f"✨ {message}...", end="", flush=True)

    # 根据 debug 模式捕获或显示输出
    result = subprocess.run(cmd, capture_output=not debug, text=True)

    if result.returncode == 0:
        if message:
            print("✅")
        if success_message:
            print(f"✅ {success_message}")
        return True
    else:
        if message:
            print("❌")
        print(f"错误：命令执行失败: {' '.join(cmd)}\n 返回 {result.returncode}")
        if not debug:
            if result.stdout:
                print("STDOUT:\n", result.stdout)
            if result.stderr:
                print("STDERR:\n", result.stderr)
        return False


def fix_attn_op(model_path, debug=False):
    """
    修复 ONNX 模型中的 Attention 操作并进行简化。
    """
    print(f"✨ 修复并简化 ONNX 模型: {model_path.name}...", end="", flush=True)
    try:
        model_onnx = onnx.load(model_path)

        # 遍历模型中的所有节点
        for node in model_onnx.graph.node:
            # 检查节点是否属于com.microsoft域
            if node.domain == "com.microsoft" and "Attention" in node.op_type:
                if debug:
                    print(f"\n  - 发现 Attention 节点: {node.name}", end="", flush=True)
                # 检查节点是否有4个输入
                if len(node.input) == 4:
                    # 在第3和第4个输入之间插入两个空输入
                    node.input.insert(3, "")
                    node.input.insert(4, "")
                    if debug:
                        print(" (已插入空输入)", end="", flush=True)

        # 对于 ONNX 简化，假设输入形状为 1, 3, 448, 448
        initial_input_shape = {"pixel_values": (1, 3, 448, 448)}

        model_onnx, check = simplify(
            model_onnx, overwrite_input_shapes=initial_input_shape
        )
        if not check and debug:
            print("❗ ONNX 简化检查失败，但继续处理。", end="", flush=True)

        onnx.save(model_onnx, model_path)
        print("✅")
        return True
    except Exception as e:
        print("❌")
        print(f"ONNX 修复和简化过程中出错: {e}")
        return False


def export_mixtex_encoder(hf_model, output_dir, debug=False):
    """
    将 MixTex 编码器导出为 ONNX 格式。
    """
    model_path = output_dir / "encoder.onnx"
    model = hf_model

    # 确保编码器的 pooler 为 None，以便 ONNX 导出
    if hasattr(model.encoder, "pooler"):
        model.encoder.pooler = None

    fake_input = torch.randn(1, 3, 448, 448, dtype=torch.float32, device=model.device)

    # 转换 SwinSelfAttention 中的 QKV
    for m in model.encoder.modules():
        if isinstance(m, SwinSelfAttention):
            m.convert_qkv()

    # 再次设置 pooler 为 None，以防它被重新赋值
    if hasattr(model.encoder, "pooler"):
        model.encoder.pooler = None

    print("✨ 转换编码器为 ONNX 格式...", end="", flush=True)
    try:
        with torch.inference_mode():
            _ = torch.onnx.export(
                model.encoder,
                (fake_input,),
                model_path,
                input_names=["pixel_values"],
                output_names=["last_hidden_state"],
                opset_version=20,
                # 如果需要动态批处理大小，可以添加 dynamic_axes={"pixel_values": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"}}
            )
        print("✅")
    except Exception as e:
        print("❌")
        print(f"ONNX 导出过程中出错: {e}")
        return False

    if not fix_attn_op(model_path, debug):
        return False

    print("✨ 验证 ONNX 导出结果...", end="", flush=True)
    try:
        o = model.encoder(fake_input)[0].detach().cpu().numpy()

        s = onnxruntime.InferenceSession(str(model_path))  # 将 Path 对象转换为字符串
        o_s = s.run(
            None,
            {
                "pixel_values": fake_input.detach().cpu().numpy(),
            },
        )[0]

        assert np.allclose(o, o_s, rtol=1e-4, atol=1e-4)
        print("✅")
        if debug:
            print(f"PyTorch 和 ONNX 输出的平均绝对差: {np.mean(np.abs(o - o_s))}")
        return True
    except Exception as e:
        print("❌")
        print(f"ONNX 验证过程中出错: {e}")
        return False


def export_mixtex_decoder(
    model_cache_dir, output_dir, quant_type="Q4_K_M", debug=False
):
    """
    将 MixTex 解码器导出并量化为 GGUF 格式。
    """
    decoder_fp32_gguf = output_dir / "mixtex-dec_fp32.gguf"
    decoder_quant_gguf = output_dir / f"mixtex-dec-{quant_type}.gguf"

    # 转换 HF 模型为 GGUF (fp32)
    cmd = [
        "uv",
        "run",
        "convert_hf_to_gguf.py",
        f"{str(model_cache_dir)}",
        "--outfile",
        str(decoder_fp32_gguf),
    ]
    if not run_command(
        cmd, debug, f"转换解码器为 fp32 GGUF ({decoder_fp32_gguf.name})"
    ):
        return False

    # 量化 GGUF
    quantizer_bin = (
        Path("bin/llama-quantize.exe")
        if sys.platform == "win32"
        else Path("bin/llama-quantize")
    )
    # 确保 llama-quantize 工具存在于预期的路径
    if not quantizer_bin.exists():
        print(f"错误：量化器二进制文件未找到，请检查路径: {quantizer_bin.absolute()}")
        return False

    cmd = [
        str(quantizer_bin.absolute()),
        str(decoder_fp32_gguf),
        str(decoder_quant_gguf),
        quant_type,
    ]
    if not run_command(
        cmd, debug, f"量化解码器为 {quant_type} GGUF ({decoder_quant_gguf.name})"
    ):
        return False

    return True


def export_gex(model_cache_dir, output_dir, quant_type="Q4_K_M", debug=False):
    """
    将 GexT 模型 (解码器和 mmproj) 导出并量化为 GGUF 格式。
    """
    gext_fp32_gguf = output_dir / "gext_fp32.gguf"
    gext_quant_gguf = output_dir / f"gext-{quant_type}.gguf"
    mmproj_fp32_gguf = output_dir / "mmproj_fp32.gguf"

    # 转换 HF 模型为 GGUF (解码器 fp32)
    cmd_gext = [
        "uv",
        "run",
        "convert_hf_to_gguf.py",
        f"{str(model_cache_dir)}",
        "--outfile",
        str(gext_fp32_gguf),
    ]
    if not run_command(
        cmd_gext, debug, f"转换 GexT 解码器为 fp32 GGUF ({gext_fp32_gguf.name})"
    ):
        return False

    # 转换 HF 模型为 GGUF (mmproj fp32)
    cmd_mmproj = [
        "uv",
        "run",
        "convert_hf_to_gguf.py",
        f"{str(model_cache_dir)}",
        "--outfile",
        str(mmproj_fp32_gguf),
        "--mmproj",
    ]
    if not run_command(
        cmd_mmproj, debug, f"转换 GexT mmproj 为 fp32 GGUF ({mmproj_fp32_gguf.name})"
    ):
        return False

    # 量化 GexT 解码器
    quantizer_bin = (
        Path("bin/llama-quantize.exe")
        if sys.platform == "win32"
        else Path("bin/llama-quantize")
    )
    if not quantizer_bin.exists():
        print(f"错误：量化器二进制文件未找到，请检查路径: {quantizer_bin.absolute()}")
        return False

    cmd_quant = [
        str(quantizer_bin.absolute()),
        str(gext_fp32_gguf),
        str(gext_quant_gguf),
        quant_type,
    ]
    if not run_command(
        cmd_quant,
        debug,
        f"量化 GexT 解码器为 {quant_type} GGUF ({gext_quant_gguf.name})",
    ):
        return False

    # 注意: 原始脚本没有量化 mmproj，只转换为 fp32。如果需要量化 mmproj，请在此处添加额外的量化步骤。

    return True


def main():
    parser = argparse.ArgumentParser(
        description="将 Hugging Face 模型转换为 ONNX 和 GGUF 格式。"
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Hugging Face 模型路径 (例如: 'MosRat/GexT_V1' 或 'MixTex/base_ZhEn')。",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mixtex", "gext"],
        required=True,
        help="模型类型: 'mixtex' (编码器-解码器) 或 'gext' (AutoModelForCausalLM)。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./converted_models",
        help="保存转换后的 ONNX 和 GGUF 模型的目录。",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="Q4_K_M",
        help="GGUF 模型的量化类型 (例如: Q4_K_M, Q5_K_M)。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式以显示完整的子进程输出和更详细的消息。",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    print(
        f"\n🚀 开始模型转换: {args.hf_model_path} (类型: {args.model_type.upper()})..."
    )
    print(f"📦 输出将保存到: {output_dir.absolute()}")
    print(f"⚙️ 量化类型: {args.quant_type}")
    if args.debug:
        print("🐞 调试模式已启用。")
    print("-" * 50)

    # --- 加载模型和分词器 ---
    # 为模型在本地的缓存创建一个专用目录
    model_name_for_dir = args.hf_model_path.replace("/", "_")
    model_cache_dir = Path("./hf_models") / model_name_for_dir
    model_cache_dir.mkdir(exist_ok=True, parents=True)  # 确保基本缓存目录存在

    tokenizer = None
    hf_model = None

    try:
        print("✨ 加载分词器...", end="", flush=True)
        # 将 cache_dir 设置为模型专属的本地缓存目录
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model_path, trust_remote_code=True, cache_dir=model_cache_dir
        )
        print("✅")

        print("✨ 加载 Hugging Face 模型...", end="", flush=True)
        if args.model_type == "mixtex":
            hf_model = VisionEncoderDecoderModel.from_pretrained(
                args.hf_model_path, cache_dir=model_cache_dir
            ).eval()
        elif args.model_type == "gext":
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.hf_model_path, trust_remote_code=True, cache_dir=model_cache_dir
            ).eval()
        print("✅")

        # 将模型和分词器保存到缓存目录，以便 `convert_hf_to_gguf.py` 可以访问它们
        print(f"✨ 缓存模型和分词器到 {model_cache_dir}...", end="", flush=True)
        hf_model.save_pretrained(model_cache_dir)
        tokenizer.save_pretrained(model_cache_dir)
        # 对于 MixTex，如果存在，也保存 feature_extractor
        if args.model_type == "mixtex":
            try:
                feature_extractor = AutoImageProcessor.from_pretrained(
                    args.hf_model_path, cache_dir=model_cache_dir
                )
                feature_extractor.save_pretrained(model_cache_dir)
            except Exception as fe_e:
                if args.debug:
                    print(
                        f"\n警告: 无法加载/保存 feature extractor (可能不存在): {fe_e}"
                    )
        print("✅")

    except Exception as e:
        print("❌")
        print(f"从 Hugging Face 加载模型或分词器失败: {e}")
        sys.exit(1)  # 加载失败则退出

    # --- 根据模型类型执行转换 ---
    success = False
    if args.model_type == "mixtex":
        print("\n--- 导出 MixTex 编码器 (ONNX) ---")
        if export_mixtex_encoder(hf_model, output_dir, args.debug):
            print("\n--- 导出 MixTex 解码器 (GGUF) ---")
            success = export_mixtex_decoder(
                model_cache_dir, output_dir, args.quant_type, args.debug
            )
    elif args.model_type == "gext":
        # GexT 模型直接通过 convert_hf_to_gguf.py 处理，包括其视觉部分 (mmproj)
        print("\n--- 导出 GexT (GGUF) ---")
        success = export_gex(model_cache_dir, output_dir, args.quant_type, args.debug)

    print("-" * 50)
    if success:
        print("🎉 转换过程完成成功！")
    else:
        print("💔 转换过程失败。")
        sys.exit(1)  # 转换失败则退出


if __name__ == "__main__":
    main()
