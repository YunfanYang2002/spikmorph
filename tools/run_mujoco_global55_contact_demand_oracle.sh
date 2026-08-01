#!/usr/bin/env bash

repo="/home/yyf/Workspace/Code/spikmorph"
python_bin="/home/yyf/miniconda3/envs/spikmorph/bin/python"
batch="$repo/output/diagnostics/mujoco_control_51k_20260727_091638"
job="$batch/jobs/job_000_seed1409_lr0p00015"
checkpoint="$job/Unimal-v0.pt"
existing_oracle="$repo/output/diagnostics/mujoco_physical_contact_projection_20260729_195100"
morphology="floor-1409-0-3-01-15-56-55"

cd "$repo"

walker_dir="$($python_bin -c "import json,pathlib; p=pathlib.Path('$batch/manifest.json'); print(json.loads(p.read_text(encoding='utf-8'))['source_audit']['walker_dir'])" 2>/dev/null)"
stamp="$(date +%Y%m%d_%H%M%S)"
out="$repo/output/diagnostics/mujoco_global55_contact_demand_oracle_${stamp}"
zip_path="$repo/tmp/mujoco_global55_contact_demand_oracle_${stamp}.zip"
staging_log="$repo/tmp/mujoco_global55_contact_demand_oracle_${stamp}.run.log"
identity_before="$repo/tmp/mujoco_global55_contact_demand_oracle_${stamp}.identity_before.txt"
status_before="$repo/tmp/mujoco_global55_contact_demand_oracle_${stamp}.status_before.txt"
hashes_before="$repo/tmp/mujoco_global55_contact_demand_oracle_${stamp}.hashes_before.sha256"
xml="$walker_dir/xml/${morphology}.xml"
metadata="$walker_dir/metadata/${morphology}.json"

mkdir -p "$repo/tmp"

{
  echo "TOPLEVEL=$(git rev-parse --show-toplevel)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "BRANCH=$(git branch --show-current)"
  echo "WALKER_DIR_FROM_FROZEN_TRAINING_MANIFEST=$walker_dir"
} > "$identity_before"
git status --short > "$status_before"
sha256sum "$xml" "$metadata" "$checkpoint" \
  tools/analyze_mujoco_global55_contact_demand.py \
  tools/evaluate_mujoco_checkpoint.py > "$hashes_before" 2>&1

"$python_bin" -B tools/analyze_mujoco_global55_contact_demand.py \
  --checkpoint "$checkpoint" \
  --walker-dir "$walker_dir" \
  --morphology-id "$morphology" \
  --existing-oracle "$existing_oracle" \
  --output-dir "$out" \
  --device cpu 2>&1 | tee "$staging_log"
probe_rc=${PIPESTATUS[0]}

mkdir -p "$out"
cp "$staging_log" "$out/run.log"
cp "$identity_before" "$out/git_head.txt"
cp "$status_before" "$out/git_status_short.txt"
cp "$hashes_before" "$out/hashes_before.sha256"

sha256sum "$xml" "$metadata" "$checkpoint" \
  tools/analyze_mujoco_global55_contact_demand.py \
  tools/evaluate_mujoco_checkpoint.py > "$out/hashes_after.sha256" 2>&1

"$python_bin" -c "import json,pathlib; p=pathlib.Path('$out/validation.json'); v=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}; v['formal_probe_return_code']=int('$probe_rc'); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8')"

"$python_bin" -c "from pathlib import Path; import zipfile; out=Path('$out').resolve(); z=Path('$zip_path').resolve(); z.parent.mkdir(parents=True,exist_ok=True); a=zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED); [a.write(p,p.relative_to(out.parent)) for p in sorted(out.rglob('*')) if p.is_file()]; a.close()"

"$python_bin" -c "from pathlib import Path; import hashlib,zipfile; z=Path('$zip_path').resolve(); a=zipfile.ZipFile(z); bad=a.testzip(); names=a.namelist(); a.close(); print('ZIP_VERIFY=' + ('PASS' if bad is None else 'FAIL:' + str(bad))); print('ZIP_FILE_COUNT=' + str(len(names))); print('ZIP_SHA256=' + hashlib.sha256(z.read_bytes()).hexdigest())"

printf 'UPLOAD_THIS_ZIP=%s\n' "$zip_path"
