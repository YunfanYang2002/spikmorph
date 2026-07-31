#!/usr/bin/env bash

repo="/home/yyf/Workspace/Code/spikmorph"
failed_bracket="$repo/output/diagnostics/mujoco_distal_friction_mu_bracket_20260731_091808"
existing_temp="$repo/tmp/mujoco_distal_friction_mu_bracket_20260731_091808"
source_xml="/home/yyf/Workspace/Code/metamorph-isaac/output/unimals_100/train/xml/floor-1409-0-3-01-15-56-55.xml"
metadata="/home/yyf/Workspace/Code/metamorph-isaac/output/unimals_100/train/metadata/floor-1409-0-3-01-15-56-55.json"
checkpoint="$repo/output/diagnostics/mujoco_control_51k_20260727_091638/jobs/job_000_seed1409_lr0p00015/Unimal-v0.pt"

cd "$repo"

stamp="$(date +%Y%m%d_%H%M%S)"
out="$repo/output/diagnostics/mujoco_explicit_pair_compile_audit_${stamp}"
regenerated_temp="$repo/tmp/mujoco_explicit_pair_compile_audit_${stamp}"
zip_path="$repo/tmp/mujoco_explicit_pair_compile_audit_${stamp}.zip"
staging_log="$repo/tmp/mujoco_explicit_pair_compile_audit_${stamp}.run.log"
identity_before="$repo/tmp/mujoco_explicit_pair_compile_audit_${stamp}.identity_before.txt"

mkdir -p "$repo/tmp"

{
  echo "TOPLEVEL=$(git rev-parse --show-toplevel)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "BRANCH=$(git branch --show-current)"
  git status --short
  sha256sum "$source_xml" "$checkpoint"
} > "$identity_before"

python tools/audit_mujoco_explicit_pair_compile.py \
  --failed-bracket-root "$failed_bracket" \
  --existing-temporary-root "$existing_temp" \
  --regenerated-temporary-root "$regenerated_temp" \
  --source-xml "$source_xml" \
  --metadata "$metadata" \
  --checkpoint "$checkpoint" \
  --output-dir "$out" \
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
  sha256sum "$source_xml" "$checkpoint"
} > "$out/source_identity_after.txt"

python -c "import json,pathlib; p=pathlib.Path('$out/validation.json'); v=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}; v['formal_probe_return_code']=int('$probe_rc'); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8')"

python -c "from pathlib import Path; import zipfile; out=Path('$out').resolve(); z=Path('$zip_path').resolve(); z.parent.mkdir(parents=True,exist_ok=True); a=zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED); [a.write(p,p.relative_to(out.parent)) for p in sorted(out.rglob('*')) if p.is_file()]; a.close()"

unzip -l "$zip_path"
unzip -t "$zip_path"
sha256sum "$zip_path"

printf 'UPLOAD_THIS_ZIP=%s\n' "$zip_path"
