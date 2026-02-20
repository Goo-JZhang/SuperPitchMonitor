-- 唤起Terminal并运行数据生成脚本

tell application "Terminal"
    activate
    set scriptPath to POSIX path of ((path to me as text) & "::run_in_terminal.sh")
    set cmd to "bash '" & scriptPath & "'"
    
    do script cmd
end tell
