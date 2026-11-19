"""Consistency run: re-runs the fixed subset of OLD 4-language REL inferences (consistency_subset_ids.json) through the UNCHANGED 6-language REL runner
(mpr_crosslocale.inference.run_rel_followup_6lang), i.e. the same code path as the main fr/ja run, and writes into its own folder.
Only the row filter of that module is replaced (in this process, nothing on disk changes): rows are kept when their input_id is in the subset.

  python consistency_run.py --ids consistency_subset_ids.json -- <arguments of run_rel_followup_6lang.py>

Every listed id must be found as a canonical row (otherwise the run stops before inference)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "01_code/src"))
import mpr_crosslocale.inference.run_rel_followup_6lang as runner  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    if "--ids" not in argv or "--" not in argv:
        raise SystemExit(__doc__)
    ids_path = Path(argv[argv.index("--ids") + 1])
    runner_args = argv[argv.index("--") + 1:]
    ids = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]
    wanted = set(ids)
    if len(wanted) != len(ids):
        raise SystemExit("duplicate ids in the subset file")
    seen = []

    def keep_subset(rows, subset):
        kept = [r for r in rows if runner.input_id(r) in wanted]
        seen.extend(runner.input_id(r) for r in kept)
        if len(kept) != len(wanted) or set(seen) != wanted:
            missing = sorted(wanted - set(seen))
            raise SystemExit(f"consistency subset: {len(kept)} rows found for {len(wanted)} ids; missing e.g. {missing[:3]}")
        return kept

    runner.filter_languages = keep_subset
    runner.main(runner_args)


if __name__ == "__main__":
    main()
