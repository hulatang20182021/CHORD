$ErrorActionPreference = "Stop"

$Base = "component_relation_sid/rqvae_supervision"
$InputEmb = "$Base/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy"
$TextOrder = "$Base/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json"
$CfEmb = "$Base/results/cf_embeddings/Beauty_cf_svd_item_emb.npy"
$CfOrder = "$Base/results/cf_embeddings/Beauty_cf_svd_item_id_order.json"
$IndexDir = "$Base/results/indices"

$Runs = @(
  @{ Label = "lam001"; Lambda = "0.01"; Out = "$Base/checkpoints/Beauty/cf_only_st5_rqvae_lam001_seed2024"; Prefix = "Beauty_cf_only_lam001_st5_rqvae" },
  @{ Label = "lam005"; Lambda = "0.05"; Out = "$Base/checkpoints/Beauty/cf_only_st5_rqvae_lam005_seed2024"; Prefix = "Beauty_cf_only_lam005_st5_rqvae" },
  @{ Label = "lam010"; Lambda = "0.10"; Out = "$Base/checkpoints/Beauty/cf_only_st5_rqvae_lam010_seed2024"; Prefix = "Beauty_cf_only_lam010_st5_rqvae" }
)

foreach ($Run in $Runs) {
  Write-Host "=== Training CF-only $($Run.Label) lambda=$($Run.Lambda) ==="
  conda run -n cr_letter python "$Base/scripts/train_cf_only_st5_rqvae.py" `
    --input $InputEmb `
    --cf_input $CfEmb `
    --text_item_order $TextOrder `
    --cf_item_order $CfOrder `
    --output_dir $Run.Out `
    --seed 2024 `
    --epochs 50 `
    --batch_size 256 `
    --lambda_cf_global $Run.Lambda `
    --temperature 0.1 `
    --device cuda:0

  Write-Host "=== Generating index $($Run.Label) ==="
  conda run -n cr_letter python "$Base/scripts/generate_cf_only_st5_rqvae_index.py" `
    --checkpoint_dir $Run.Out `
    --input $InputEmb `
    --item_order $TextOrder `
    --output_dir $IndexDir `
    --output_prefix $Run.Prefix `
    --device cuda:0

  Write-Host "=== Auditing $($Run.Label) ==="
  conda run -n cr_letter python "$Base/scripts/audit_cf_only_st5_rqvae_index.py" `
    --index "$IndexDir/$($Run.Prefix).index.json" `
    --method_name $Run.Prefix `
    --output_prefix $Run.Prefix
}
