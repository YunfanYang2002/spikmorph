#!/usr/bin/env bash

repo="/home/yyf/Workspace/Code/spikmorph"
source_xml="/home/yyf/Workspace/Code/metamorph-isaac/output/unimals_100/train/xml/floor-1409-0-3-01-15-56-55.xml"
metadata="/home/yyf/Workspace/Code/metamorph-isaac/output/unimals_100/train/metadata/floor-1409-0-3-01-15-56-55.json"
checkpoint="$repo/output/diagnostics/mujoco_control_51k_20260727_091638/jobs/job_000_seed1409_lr0p00015/Unimal-v0.pt"
existing_oracle="$repo/output/diagnostics/mujoco_physical_contact_projection_20260729_195100"

cd "$repo"

stamp="$(date +%Y%m%d_%H%M%S)"
out="$repo/output/diagnostics/mujoco_distal_friction_mu_bracket_${stamp}"
temporary_root="$repo/tmp/mujoco_distal_friction_mu_bracket_${stamp}"
zip_path="$repo/tmp/mujoco_distal_friction_mu_bracket_${stamp}.zip"
staging_log="$repo/tmp/mujoco_distal_friction_mu_bracket_${stamp}.run.log"
identity_before="$repo/tmp/mujoco_distal_friction_mu_bracket_${stamp}.identity_before.txt"

mkdir -p "$repo/tmp"

{
  echo "TOPLEVEL=$(git rev-parse --show-toplevel)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "BRANCH=$(git branch --show-current)"
  git status --short
} > "$identity_before"

python tools/run_mujoco_distal_friction_bracket.py \
  --source-xml "$source_xml" \
  --metadata "$metadata" \
  --checkpoint "$checkpoint" \
  --existing-oracle "$existing_oracle" \
  --output-dir "$out" \
  --temporary-root "$temporary_root" \
  --device cpu 2>&1 | tee "$staging_log"
probe_rc=${PIPESTATUS[0]}

mkdir -p "$out"
cp "$staging_log" "$out/run.log"
cp "$identity_before" "$out/source_identity_before.txt"

{
  echo "TOPLEVEL=$(git rev-parse --show-toplevel)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "BRANCH=$(git branch --show-current)"
  echo "PROBE_RC=$probe_rc"
  git status --short
} > "$out/source_identity_after.txt"

sha256sum "$source_xml" "$checkpoint" > "$out/source_hashes_after.sha256"

python -c "import json,pathlib; p=pathlib.Path('$out/validation.json'); v=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}; v['formal_probe_return_code']=int('$probe_rc'); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8')"

python -c "from pathlib import Path; import zipfile; out=Path('$out').resolve(); z=Path('$zip_path').resolve(); z.parent.mkdir(parents=True,exist_ok=True); a=zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED); [a.write(p,p.relative_to(out.parent)) for p in sorted(out.rglob('*')) if p.is_file()]; a.close()"

unzip -l "$zip_path"
unzip -t "$zip_path"
sha256sum "$zip_path"

printf 'UPLOAD_THIS_ZIP=%s\n' "$zip_path"
