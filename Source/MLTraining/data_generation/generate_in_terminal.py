#!/usr/bin/env python3
"""
在新Terminal窗口中生成sanity check数据集 (跨平台)
显示实时进度

Mac: 使用osascript唤起Terminal
Windows: 使用start命令唤起cmd
Linux: 使用gnome-terminal或xterm
"""

import subprocess
import sys
import platform
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    shell_script = script_dir / "run_in_terminal.sh"
    bat_script = script_dir / "run_in_terminal.bat"
    
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        print("Launching Terminal for data generation...")
        print("The Terminal window will show real-time progress.")
        print("")
        
        # 确保脚本是可执行的
        shell_script.chmod(0o755)
        
        # 使用osascript唤起Terminal
        applescript = f'''
        tell application "Terminal"
            activate
            do script "bash '{shell_script}'"
        end tell
        '''
        
        subprocess.run(["osascript", "-e", applescript])
        print("Terminal launched! Check the new window for progress.")
        
    elif system == 'Windows':
        print("Launching Command Prompt for data generation...")
        print("The new window will show real-time progress.")
        print("")
        
        # Windows使用start命令
        # 创建批处理脚本
        bat_content = f'''@echo off
cd /d "{script_dir}"
call conda activate spm_ml
echo ==========================================
echo   Generating Sanity Check Dataset
echo ==========================================
echo.
echo Configuration:
echo   - Total samples: 20480 (2048 bins x 10)
echo   - Timbre: Sine only
echo   - Note type: Type 4 (full duration)
echo   - Output: ..\\..\\..\\TrainingData\\sanity_check_20k.hdf5
echo.
echo Starting generation...
echo.
python hdf5_writer.py --output ..\\..\\..\\TrainingData\\sanity_check_20k.hdf5 --sanity
echo.
echo ==========================================
if errorlevel 1 (
    echo   Generation FAILED!
) else (
    echo   Generation COMPLETED!
)
echo ==========================================
pause
'''
        bat_script.write_text(bat_content)
        
        # 启动新cmd窗口
        subprocess.Popen(['start', 'cmd', '/k', str(bat_script)], shell=True)
        print("Command Prompt launched! Check the new window for progress.")
        
    elif system == 'Linux':
        print("Launching terminal for data generation...")
        
        # 确保脚本是可执行的
        shell_script.chmod(0o755)
        
        # 尝试不同的终端模拟器
        terminals = [
            ['gnome-terminal', '--', 'bash', str(shell_script)],
            ['konsole', '-e', 'bash', str(shell_script)],
            ['xterm', '-e', 'bash', str(shell_script)],
        ]
        
        launched = False
        for term in terminals:
            try:
                subprocess.Popen(term)
                launched = True
                print(f"Launched with {term[0]}")
                break
            except FileNotFoundError:
                continue
        
        if not launched:
            print("Could not find a terminal emulator.")
            print("Please run the script manually:")
            print(f"  bash {shell_script}")
    else:
        print(f"Unsupported platform: {system}")
        print("Please run the generation script manually:")
        print(f"  python hdf5_writer.py --output ../../../TrainingData/sanity_check_20k.hdf5 --sanity")


if __name__ == "__main__":
    main()
