# -*- mode: python -*-
import shutil
import subprocess
import os
from cdprep import __version__
from cdprep.utils.ospath import delete_folder_recursively

block_cipher = None

added_files = [
    ('../cdprep/ressources/icons/*.png', 'ressources/icons'),
    ('../cdprep/ressources/icons/*.svg', 'ressources/icons'),
    ('../cdprep/ressources/Station Inventory EN.csv', 'ressources/Station Inventory EN.csv')
    ]
a = Analysis(['../cdprep/app/mainwindow.py'],
             pathex=['C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\ucrt\\DLLs\\x64'],
             binaries=[('C:\\Windows\\System32\\vcruntime140_1.dll', '.')],
             datas=added_files ,
             hiddenimports=[
                 'win32timezone', 'pkg_resources.py2_warn', 'gdown'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PySide', 'PyQt4'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='cdprep',
          debug=False,
          strip=False,
          upx=True,
          console=True,
          icon='cdprep.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Climate Data Preprocessing Tool')

# Prepare the binary folder.
shutil.copyfile("../LICENSE", "dist/LICENSE")
if os.environ.get('AZURE'):
    output_dirname = os.environ.get('CDPREP_OUTPUT_DIRNAME')
else:
    output_dirname = 'cdprep_'+__version__+'_win_amd64'
delete_folder_recursively(output_dirname, delroot=True)
os.rename('dist', output_dirname)
