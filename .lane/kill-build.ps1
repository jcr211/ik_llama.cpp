# Kill only this lane's build processes (command line mentions build-sl1 or the lane worktree),
# with their children.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('ninja.exe', 'cmake.exe', 'cmd.exe') -and ([string]$_.CommandLine) -match 'build-sl1|sl1-spec-ckpt'
}
foreach ($p in $procs) {
    Write-Output ("kill-build: taskkill /T {0} {1}" -f $p.ProcessId, $p.Name)
    taskkill /F /T /PID $p.ProcessId | Out-Null
}
