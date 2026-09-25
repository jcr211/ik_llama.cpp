$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sha = (git -C $source rev-parse --short HEAD).Trim()
$build = Join-Path $source "build-avx512-prod-$sha"
$snapshot = Join-Path $PSScriptRoot 'served-CMakeCache.txt'
$configure = @('-S', $source, '-B', $build, '-G', 'Ninja')
$names = '^(GGML_|LLAMA_|CMAKE_BUILD_TYPE$|CMAKE_CUDA_ARCHITECTURES$|CMAKE_C_COMPILER$|' +
    'CMAKE_CXX_COMPILER$|CMAKE_CUDA_COMPILER$|CMAKE_CUDA_HOST_COMPILER$|CMAKE_CXX_FLAGS|' +
    'CMAKE_C_FLAGS|CMAKE_CUDA_FLAGS|CUDAToolkit_|CUDA_)'

function Get-SelectedCache([string] $path) {
    $entries = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        if ($line -match '^([^/#][^:=]*):([^=]+)=(.*)$') {
            $name = $Matches[1]
            $type = $Matches[2]
            $value = $Matches[3]
            if ($name -cmatch $names) {
                $entries[$name] = @{ Type = $type; Value = $value }
            }
        }
    }
    return $entries
}

if (Test-Path -LiteralPath $snapshot) {
    $served = Get-SelectedCache $snapshot
    if (-not $served.ContainsKey('CMAKE_BUILD_TYPE') -or
        -not $served.ContainsKey('CMAKE_CUDA_ARCHITECTURES') -or
        -not $served.ContainsKey('GGML_AVX512')) {
        Write-Error 'The served cache snapshot lacks required build variables.'
        exit 10
    }
    $initialCache = Join-Path $PSScriptRoot 'tmp/served-initial-cache.cmake'
    $cacheLines = foreach ($name in ($served.Keys | Sort-Object)) {
        $value = $served[$name].Value
        $equals = '='
        while ($value.Contains("]$equals]")) { $equals += '=' }
        $type = $served[$name].Type
        if ($type -eq 'UNINITIALIZED') { $type = 'STRING' }
        "set($name [$equals[$value]$equals] CACHE $type `"Served production cache`" FORCE)"
    }
    $cacheLines | Set-Content -LiteralPath $initialCache -Encoding Ascii
    $configure += @('-C', $initialCache)
    $mode = 'served cache snapshot'
} else {
    if ($env:ALLOW_PROVISIONAL -ne '1') {
        Write-Error 'Missing .lane/served-CMakeCache.txt; exact served-build configuration is unavailable.'
        exit 10
    }
    $configure += @('-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON', '-DGGML_AVX512=ON',
        '-DCMAKE_CUDA_ARCHITECTURES=120', '-DLLAMA_BUILD_TESTS=ON')
    $mode = 'F11 lane flags; served cache equivalence unverified'
}

Write-Output "Build directory: $build"
Write-Output "Configuration: $mode"
$buildInfo = Join-Path $source 'common/build-info.cpp'
if (Test-Path -LiteralPath $buildInfo) { Remove-Item -LiteralPath $buildInfo -Force }
& cmake @configure
if ($LASTEXITCODE -ne 0) { exit 12 }
if ($served) {
    $actual = Get-SelectedCache (Join-Path $build 'CMakeCache.txt')
    $diff = foreach ($name in $served.Keys) {
        $compilerSpelling = $name -in @('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER') -and
            $served[$name].Value -eq 'cl' -and
            $actual.ContainsKey($name) -and
            (Split-Path $actual[$name].Value -Leaf) -eq 'cl.exe'
        if (-not $actual.ContainsKey($name) -or
            ($actual[$name].Value -ne $served[$name].Value -and -not $compilerSpelling)) {
            "$name served=$($served[$name].Value) built=$($actual[$name].Value)"
        }
    }
    $diffPath = Join-Path $PSScriptRoot 'cache-diff.out'
    if ($diff) {
        $diff | Set-Content -LiteralPath $diffPath
        Write-Error "Selected cache variables differ; see $diffPath"
        exit 15
    }
    'No differences in selected served cache variables.' | Set-Content -LiteralPath $diffPath
}
$targets = @('llama-server', 'test-stateos-header', 'test-speculative-params', 'test-ple-hist',
    'test-stateos-layout')
& cmake --build $build --target @targets -j 12
if ($LASTEXITCODE -ne 0) { exit 13 }

$exe = Join-Path $build 'bin/llama-server.exe'
if (-not (Test-Path -LiteralPath $exe)) { exit 14 }
Write-Output "Built: $exe"
