<#
  Autonomous (fire-and-forget) GCE runner for the full FinanceBench run.

  Unlike run_on_gce.ps1 (which runs the benchmark attached over SSH and pulls
  results back to the local machine), this:
    - provisions the VM with cloud-platform scope and a long max-run-duration
      (so a ~14h GLM ingest of 150 10-Ks can finish), termination-action=DELETE
      as the backstop,
    - ships the bundle + .env (secrets from Secret Manager),
    - launches the build+download+run DETACHED on the VM (setsid/nohup), so this
      script returns immediately and the run survives the local session ending,
    - the remote job uploads results to gs://BUCKET/NAME/ on completion and
      best-effort self-deletes the VM.

  Watch progress later with:
    gcloud compute ssh NAME --tunnel-through-iap --command "tail -f ~/vlbench-run.log"
  Results land in: gs://BUCKET/NAME/   (fetch_results.ps1 / .sh)
#>
param(
  [string]$Project  = "project-03250746-ec5b-4198-990",
  [string]$Zone     = "us-central1-a",
  [string]$Machine  = "e2-standard-4",
  [string]$Name     = "vlbench-full-$(Get-Date -Format yyyyMMdd-HHmmss)",
  [string]$Config   = "configs/financebench_threeway.yaml",
  [string]$Secret   = "server-config",
  [string]$BaseUrl  = "https://vectorless-server-2rzh3kctga-uc.a.run.app",
  [string]$Bucket   = "gs://vectorless-engine-us-central1",
  [string]$Tags     = "dokploy",
  [int]$DiskGb      = 100,
  [string]$MaxRun   = "20h"
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

# Note: the project's default compute service account already holds
# roles/storage.objectAdmin on the results bucket, so the VM (cloud-platform
# scope) can upload results — no IAM change needed here.

Write-Host ">> creating VM $Name ($Machine, $Zone, disk ${DiskGb}GB, max-run $MaxRun)"
gcloud compute instances create $Name --project=$Project --zone=$Zone --machine-type=$Machine `
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size="${DiskGb}GB" `
  --scopes=cloud-platform `
  --tags=$Tags --max-run-duration=$MaxRun --instance-termination-action=DELETE `
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
  $tgzName = "vlbench-$Name.tgz"
  tar --exclude=.git --exclude=runs --exclude="data/financebench/docs" `
      --exclude=.venv --exclude=__pycache__ --exclude="*.pyc" --exclude="*.tgz" `
      -czf $tgzName -C $repo .
  gcloud compute scp $tgzName "${Name}:vlbench.tgz" --project=$Project --zone=$Zone --tunnel-through-iap
  if ($LASTEXITCODE -ne 0) { throw "scp bundle failed" }
  gcloud compute scp ".env" "${Name}:vlbench.env" --project=$Project --zone=$Zone --tunnel-through-iap
  if ($LASTEXITCODE -ne 0) { throw "scp .env failed" }

  # Remote runner: build -> download docs -> run -> upload to GCS -> self-delete.
  # Single-quoted here-string so PowerShell does NOT evaluate bash $(...) — the
  # PS variables are injected via __PLACEHOLDER__ replacement afterwards.
  $runnerTemplate = @'
#!/usr/bin/env bash
set -uxo pipefail
exec > ~/vlbench-run.log 2>&1
# Always shut the VM down when this script ends (success OR failure) so a broken
# build can't leave it running until the max-run-duration cap.
trap 'rc=$?; echo "EXIT rc=$rc $(date -u)"; sudo gcloud compute instances delete __NAME__ --zone=__ZONE__ --quiet || sudo poweroff' EXIT
echo "START $(date -u)"
set -e   # fail LOUD: abort on first error rather than producing an empty bucket
rm -rf ~/vlbench && mkdir -p ~/vlbench && tar -xzf ~/vlbench.tgz -C ~/vlbench
cd ~/vlbench && cp ~/vlbench.env .env
sudo docker compose build
sudo docker compose run --rm --entrypoint python bench scripts/download_financebench.py
sudo docker compose run --rm bench run --config __CONFIG__ --out /results
echo "RUN DONE $(date -u); uploading to __BUCKET__/__NAME__/"
sudo gcloud storage cp -r results/* "__BUCKET__/__NAME__/"
echo "UPLOADED $(date -u)"
'@
  $runner = $runnerTemplate.Replace('__CONFIG__', $Config).Replace('__BUCKET__', $Bucket).Replace('__NAME__', $Name).Replace('__ZONE__', $Zone)
  # write the runner on the VM (base64 to avoid quoting hell), then launch detached
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($runner))
  # Run the runner as the SSH user (NOT sudo) so ~ resolves to the user's home
  # where the bundle was scp'd; the runner's own docker/gcloud calls use sudo.
  GcloudSsh "echo $b64 | base64 -d > ~/vlbench-runner.sh && chmod +x ~/vlbench-runner.sh && setsid bash ~/vlbench-runner.sh </dev/null >/dev/null 2>&1 & echo LAUNCHED"
  Write-Host ""
  Write-Host ">> LAUNCHED autonomous run on $Name."
  Write-Host "   results  -> $Bucket/$Name/"
  Write-Host "   progress -> gcloud compute ssh $Name --zone=$Zone --tunnel-through-iap --command 'tail -f ~/vlbench-run.log'"
  Remove-Item -Force $tgzName -ErrorAction SilentlyContinue
}
catch {
  Write-Host "ERROR: $_"
  Write-Host ">> deleting VM $Name (launch failed)"
  gcloud compute instances delete $Name --project=$Project --zone=$Zone --quiet
  throw
}
