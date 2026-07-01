from __future__ import annotations

import argparse
import sys

from exphub.decode.image_quality import run_image_quality_evaluation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ExpHub decode image quality evaluation.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prepare-result", required=True)
    parser.add_argument("--decode-report", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-details-csv", required=True)
    parser.add_argument("--output-summary-json", default="")
    parser.add_argument("--output-summary-csv", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--decode-profile", default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if int(args.stride) <= 0:
        print("--stride must be > 0", file=sys.stderr)
        return 2
    if int(args.max_frames) < 0:
        print("--max-frames must be >= 0", file=sys.stderr)
        return 2

    try:
        run_image_quality_evaluation(
            run_root=args.run_root,
            prepare_result_path=args.prepare_result,
            decode_report_path=args.decode_report,
            output_report_path=args.output_report,
            output_details_csv_path=args.output_details_csv,
            output_summary_json_path=args.output_summary_json or None,
            output_summary_csv_path=args.output_summary_csv or None,
            identity={
                "dataset": str(args.dataset or ""),
                "sequence": str(args.sequence or ""),
                "tag": str(args.tag or ""),
                "decode_profile": str(args.decode_profile or ""),
            },
            stride=int(args.stride),
            max_frames=int(args.max_frames),
            device=str(args.device),
            python_executable=sys.executable,
            execution_mode="subprocess_decode_python",
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
