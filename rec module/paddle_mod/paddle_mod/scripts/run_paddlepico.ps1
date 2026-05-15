param(
    [string]$ConfigPath = "configs/layout_config.json",
    [string]$InputPath = "",
    [string]$OutputPath = ""
)

$argsList = @("src/run_paddlepico.py", "--config", $ConfigPath)

if ($InputPath -ne "") {
    $argsList += @("--input", $InputPath)
}

if ($OutputPath -ne "") {
    $argsList += @("--output", $OutputPath)
}

python @argsList
