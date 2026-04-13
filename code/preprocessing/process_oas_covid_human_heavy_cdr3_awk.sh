#!/usr/bin/env bash
set -u

WORKDIR="${TMPDIR:-$PWD}"
OUTDIR="$PWD/cleaned"
mkdir -p "$OUTDIR"

shopt -s nullglob
files=( *.csv.gz )

if [ ${#files[@]} -eq 0 ]; then
  echo "No .csv.gz files found." >&2
  exit 1
fi

echo "WORKDIR=$WORKDIR"
echo "OUTDIR=$OUTDIR"
echo "Found ${#files[@]} gz files"

first_file="${files[0]}"
echo "Previewing first file: $first_file"

python3 - "$first_file" <<'PY'
import gzip, csv, sys
fn = sys.argv[1]
with gzip.open(fn, "rt", newline="") as fh:
    meta = fh.readline().rstrip()
    header = fh.readline().rstrip()
print("Metadata line:")
print(meta)
print("\nHeader line:")
print(header)
PY

cdr3_col=$(
python3 - "$first_file" <<'PY'
import gzip, csv, sys
fn = sys.argv[1]
with gzip.open(fn, "rt", newline="") as fh:
    fh.readline()
    reader = csv.reader(fh)
    header = next(reader)
for i, name in enumerate(header, 1):
    if name == "cdr3_aa":
        print(i)
        break
else:
    raise SystemExit("cdr3_aa column not found")
PY
)

echo "cdr3_aa column index: $cdr3_col"

tmp_all="$WORKDIR/covid_human_heavy_cdr3_aa_all.txt"
tmp_unique="$WORKDIR/covid_human_heavy_cdr3_aa_unique.txt"
final_unique="$OUTDIR/covid_human_heavy_cdr3_aa_unique.txt"

: > "$tmp_all"

count=0
for f in "${files[@]}"; do
  count=$((count + 1))
  echo "Processing $f ..." >&2

  if gunzip -c -- "$f" 2>/dev/null \
    | awk -F',' -v col="$cdr3_col" '
        NR <= 2 { next }
        {
          seq = $col
          gsub(/^"|"$/, "", seq)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", seq)
          seq = toupper(seq)
          if (seq ~ /^[ACDEFGHIKLMNPQRSTVWY]+$/) print seq
        }
      ' >> "$tmp_all"
  then
    :
  else
    echo "$f    skipped=gunzip_or_awk_error" >&2
    continue
  fi

  if [ $((count % 100)) -eq 0 ]; then
    echo "Processed $count/${#files[@]} files"
  fi
done

echo "Deduplicating globally..."
LC_ALL=C sort -u "$tmp_all" > "$tmp_unique"

cp "$tmp_unique" "$final_unique"

echo "Done."
echo "All extracted rows:"
wc -l "$tmp_all"
echo "Unique CDR3 count:"
wc -l "$final_unique"
echo "Output file:"
echo "$final_unique"
