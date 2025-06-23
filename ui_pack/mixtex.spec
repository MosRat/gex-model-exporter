# -*- mode: python ; coding: utf-8 -*-

# modified from https://github.com/RQLuo/MixTeX-Latex-OCR/blob/main/mixtexgui/mixtex_ui.spec
import os

# 创建DPI感知的清单文件 (保持不变)
manifest = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity type="win32" name="MixTeX" version="3.2.4.0" processorArchitecture="*"/>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" 
                        processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
    </dependentAssembly>
  </dependency>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, PerMonitor</dpiAwareness>
    </windowsSettings>
  </application>
</assembly>'''

manifest_path = 'app.manifest'
with open(manifest_path, 'w', encoding='utf-8') as f:
    f.write(manifest)

# --- 修改 Analysis 配置部分 ---
a = Analysis(
    ['main.py'],  # <-- 替换为你的Python脚本文件名, e.g., 'mixtex_gex.py'
    pathex=[],
    binaries=[
        # ADDED: 包含核心库 libgex.dll，并放在输出目录的根下
        ('libgex.dll', '.')
    ],
    datas=[
        # UI 资源文件
        ('donate.png', '.'), 
        ('icon.png', '.'), 
        ('icon_gray.png', '.'),
        # ADDED: 包含整个 gex_model 文件夹及其中的模型文件
        ('gex_model', 'gex_model')
    ],
    hiddenimports=[
        # REMOVED: 不再需要 transformers 和 onnxruntime
        # REMOVED: 不再需要 numpy
        # Kept: 以下是UI和系统托盘所需的核心库
        'PIL',
        'pystray',
        'pystray._win32' # 显式包含pystray的Windows后端
    ],
    excludes=[
        # 保持排除以减小体积，防止意外包含
        'torch', 
        'torchvision', 
        'torchaudio',
        'tensorflow',
        'keras',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'transformers',
        'onnx',
        'onnxruntime'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MixTeX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',         # <-- 确保 icon.ico 文件存在
    manifest=manifest_path,
    uac_admin=False,
)

# 清理临时清单文件
if os.path.exists(manifest_path):
    try:
        os.remove(manifest_path)
    except:
        pass