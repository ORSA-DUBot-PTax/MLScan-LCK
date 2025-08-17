# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

# Path to your Conda environment
env_path = r"C:\Users\UseR\.conda\envs\mldsxi_env"

# Path to xgboost.dll
xgboost_dll_path = os.path.join(env_path, "Lib", "site-packages", "xgboost", "lib", "xgboost.dll")

# Model files to include (no scaler.pkl)
model_files = [
    ("RF.pkl", "."),
    ("SVM.pkl", "."),
    ("KNN.pkl", "."),
    ("XGBoost.pkl", "."),
    ("LightGBM.pkl", "."),
    ("ET.pkl", "."),
    ("selected_feature_indices.npy", "."),
]

# Collect all data files from xgboost
xgboost_data = collect_data_files('xgboost')

# Prepare datas list
datas = []
for src, dest in model_files:
    if os.path.exists(src):
        datas.append((src, dest))

# Add xgboost data files (like VERSION)
for src, dest in xgboost_data:
    datas.append((src, dest))

# Add xgboost.dll explicitly
binaries = []
if os.path.exists(xgboost_dll_path):
    binaries.append((xgboost_dll_path, "xgboost/lib"))

a = Analysis(
    ['mldsxi.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'xgboost',
        'lightgbm',
        'imblearn',
        'sklearn',
        'joblib',
        'scipy',
        'numpy'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='MLDSXI',
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
)
