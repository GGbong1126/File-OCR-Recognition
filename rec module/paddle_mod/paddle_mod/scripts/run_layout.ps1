param(
    [string]$InputPath = "data/raw",
    [string]$ConfigPath = "configs/layout_config.json",
    [string]$OutputPath = "outputs"
)

python src/run_layout.py --input $InputPath --config $ConfigPath --output $OutputPath
