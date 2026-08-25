<#
  Windows-native orchestrator (PowerShell) — same flow as run_on_gce.sh but
  without Git Bash's MSYS path mangling, which breaks gcloud's bundled pscp/plink.

  Provisions a GCE VM on ajicore, ships the bundle + a .env (read from Secret
  Manager) over the IAP tunnel with `gcloud compute scp`, runs the benchmark,
  brings results back, and deletes the VM (a --max-run-duration safety net also
  auto-deletes it if this script dies).

  Usage (from the repo root or anywhere):
    pwsh deploy/gcp/run_on_gce.ps1 -Limit 2 -Docs 3
#>
param(
  [string]$Project  = "project-03250746-ec5b-4198-990",
  [string]$Zone     = "us-central1-a",
  [string]$Machine  = "e2-standard-4",
  [string]$Name     = "vlbench-$(Get-Date -Format yyyyMMdd-HHmmss)",
  [string]$Config   = "configs/financebench_gemini.yaml",
  [string]$Secret   = "server-config",
  [string]$BaseUrl  = "https://vectorless-server-2rzh3kctga-uc.a.run.app",
  [string]$Tags     = "dokploy",          # IAP firewall targets this tag
  [int]$Limit       = 0,                    # 0 = use config's limit
  [int]$Docs        = 0,                    # 0 = download all FinanceBench docs
  [int]$DiskGb      = 50,
  [switch]$KeepVm                           # skip deletion (debug)
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repo

function GcloudSsh([string]$cmd) {
  "y" | gcloud compute ssh $Name --project=$Project --zone=$Zone --tunnel-through-iap --command="$cmd"
}

Write-Host ">> vendoring the vectorless SDK into the build context"
$sdkSrc = (Resolve-Path "$repo\..\vectorless-sdk\python").Path
$dest = "$repo\vendor\vectorless-sdk"
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
Copy-Item -Recurse $sdkSrc $dest
Remove-Item -Recurse -Force "$dest\.venv","$dest\dist","$dest\build" -ErrorAction SilentlyContinue

Write-Host ">> reading secrets from Secret Manager ($Secret) into .env"
python deploy\load_secrets.py --project $Project --secret $Secret --base-url $BaseUrl --out .env
if ($LASTEXITCODE -ne 0) { throw "load_secrets failed" }

Write-Host ">> creating VM $Name ($Machine, $Zone)"
gcloud compute instances create $Name --project=$Project --zone=$Zone --machine-type=$Machine `
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size="${DiskGb}GB" `
  --tags=$Tags --max-run-duration=4h --instance-termination-action=DELETE `
  --metadata-from-file=startup-script="deploy\gcp\startup-script.sh"
if ($LASTEXITCODE -ne 0) { throw "VM create failed" }

try {
  Write-Host ">> waiting for Docker on the VM (caching host key)"
  $ready = $false
  for ($i = 0; $i -lt 40; $i++) {
    GcloudSsh "sudo docker ps >/dev/null 2>&1 && test -f /var/run/vlbench-ready" 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; Write-Host "   ready"; break }
    Start-Sleep -Seconds 15
  }
  if (-not $ready) { throw "Docker never came up on the VM" }

  Write-Host ">> shipping bundle + .env over IAP"
  # gcloud/pscp misreads a Windows drive path (C:\...) as a remote host:path
  # because of the drive colon. Use RELATIVE local paths (cwd is $repo) and a
  # home-relative remote path (no ~) to dodge it entirely.
  $tgzName = "vlbench-$Name.tgz"
  tar --exclude=.git --exclude=runs --exclude="data/financebench/docs" `
      --exclude=.venv --exclude=__pycache__ --exclude="*.pyc" --exclude="*.tgz" `
      -czf $tgzName -C $repo .
  gcloud compute scp $tgzName "${Name}:vlbench.tgz" --project=$Project --zone=$Zone --tunnel-through-iap
  if ($LASTEXITCODE -ne 0) { throw "scp bundle failed" }
  gcloud compute scp ".env" "${Name}:vlbench.env" --project=$Project --zone=$Zone --tunnel-through-iap
  if ($LASTEXITCODE -ne 0) { throw "scp .env failed" }

  $limArg = if ($Limit -gt 0) { "--limit $Limit" } else { "" }
  $docArg = if ($Docs  -gt 0) { "--limit $Docs"  } else { "" }
  $remote = @"
set -euxo pipefail
rm -rf ~/vlbench && mkdir -p ~/vlbench && tar -xzf ~/vlbench.tgz -C ~/vlbench
cd ~/vlbench && cp ~/vlbench.env .env
sudo docker compose build
sudo docker compose run --rm --entrypoint python bench scripts/download_financebench.py $docArg
sudo docker compose run --rm bench run --config $Config $limArg --out /results
"@
  Write-Host ">> building + running on the VM (long part: PDF ingest of 10-Ks)"
  GcloudSsh $remote
  if ($LASTEXITCODE -ne 0) { throw "remote benchmark failed" }

  Write-Host ">> bringing results back to runs/vm-$Name"
  New-Item -ItemType Directory -Force "runs" | Out-Null
  gcloud compute scp --recurse "${Name}:vlbench/results" "runs/vm-$Name" `
    --project=$Project --zone=$Zone --tunnel-through-iap
  if ($LASTEXITCODE -ne 0) { throw "scp results back failed" }
  Remove-Item -Force $tgzName -ErrorAction SilentlyContinue
  Get-ChildItem -Recurse "runs/vm-$Name" -Filter report.html | Select-Object -First 1 -ExpandProperty FullName
}
finally {
  if (-not $KeepVm) {
    Write-Host ">> deleting VM $Name"
    gcloud compute instances delete $Name --project=$Project --zone=$Zone --quiet
  }
}
