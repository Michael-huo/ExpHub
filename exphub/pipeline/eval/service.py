from __future__ import annotations

from exphub.common.logging import debug_info, log_warn
from exphub.contracts import eval as eval_contract


def run(runtime):
    """Formal eval stage entry with a thin bridge to scripts/eval_main.py."""
    contract = eval_contract.build_contract(runtime.paths)
    runtime.remove_in_exp(runtime.paths.eval_dir)
    runtime.paths.eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(runtime.script_path("eval_main.py")),
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--reference",
        str(runtime.paths.slam_traj_path("ori")),
        "--estimate",
        str(runtime.paths.slam_traj_path("gen")),
        "--out_dir",
        str(runtime.paths.eval_dir),
        "--reference_name",
        "ori",
        "--estimate_name",
        "gen",
        "--alignment_mode",
        "se3",
    ]
    if runtime.args.no_viz:
        cmd.append("--skip_plots")

    runtime.step_runner.run_env_python(
        cmd,
        phase_name="slam",
        log_name="eval.log",
        cwd=runtime.exphub_root,
        check=False,
    )

    report_path = contract.artifacts["report"]
    details_path = contract.artifacts["details"]
    metrics_plot_path = contract.artifacts["metrics_overview_plot"]
    if report_path.is_file():
        debug_info("STEP eval: report={}".format(report_path))
    else:
        log_warn("eval report missing: {}".format(report_path))
    if details_path.is_file():
        debug_info("STEP eval: details={}".format(details_path))
    else:
        log_warn("eval details missing: {}".format(details_path))
    if metrics_plot_path.is_file():
        debug_info("STEP eval: metrics_overview={}".format(metrics_plot_path))
    else:
        log_warn("eval metrics overview missing: {}".format(metrics_plot_path))
    return contract.root
