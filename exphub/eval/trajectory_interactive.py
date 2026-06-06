from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _read_json_dict(path_obj):
    if not path_obj:
        return {}
    path = Path(path_obj).resolve()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _float_array(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] == 0:
        return None
    return arr


def _xy_array(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return None
    return arr


def _trajectory_customdata(count, timestamps):
    ts = _float_array(timestamps)
    rows = []
    for idx in range(int(count)):
        value = float(ts[idx]) if ts is not None and idx < int(ts.shape[0]) and np.isfinite(ts[idx]) else None
        rows.append([int(idx), value])
    return rows


def _collect_generation_unit_boundaries(generation_units):
    markers = {}
    total_candidates = 0
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        for key, role in (("start_idx", "start"), ("end_idx", "end")):
            try:
                frame_idx = int(unit.get(key))
            except Exception:
                continue
            if frame_idx < 0:
                continue
            total_candidates += 1
            marker = markers.setdefault(
                frame_idx,
                {
                    "frame_idx": int(frame_idx),
                    "roles": set(),
                    "unit_ids": [],
                },
            )
            marker["roles"].add(str(role))
            if unit_id and unit_id not in marker["unit_ids"]:
                marker["unit_ids"].append(unit_id)
    out = []
    for frame_idx in sorted(markers):
        marker = markers[frame_idx]
        out.append(
            {
                "frame_idx": int(frame_idx),
                "roles": sorted(marker["roles"]),
                "unit_ids": list(marker["unit_ids"]),
            }
        )
    return total_candidates, out


def _prepared_ros_timestamps(prepare_result):
    values = _as_dict(_as_dict(prepare_result).get("frame_index_map")).get("prepared_to_ros_time_sec")
    arr = _float_array(values)
    if arr is None:
        return None
    return arr


def _format_marker_hover(marker, timestamp, x_value, y_value):
    roles = "/".join(list(marker.get("roles") or [])) or "boundary"
    unit_ids = ", ".join(list(marker.get("unit_ids") or [])) or "n/a"
    timestamp_text = "n/a" if timestamp is None else "{:.6f}".format(float(timestamp))
    return (
        "Visual anchors / boundaries"
        "<br>frame_idx={frame_idx}"
        "<br>role={roles}"
        "<br>unit_id={unit_ids}"
        "<br>timestamp={timestamp}"
        "<br>x={x:.4f}"
        "<br>y={y:.4f}"
    ).format(
        frame_idx=int(marker.get("frame_idx")),
        roles=roles,
        unit_ids=unit_ids,
        timestamp=timestamp_text,
        x=float(x_value),
        y=float(y_value),
    )


def _map_boundary_markers(
    prepare_result_path,
    generation_units_path,
    rec_xy,
    rec_timestamps,
    nearest_time_tolerance,
):
    warnings = []
    if not prepare_result_path or not generation_units_path:
        warnings.append("interactive trajectory boundary overlay skipped: prepare_result or generation_units path unavailable")
        return [], 0, 0, warnings

    prepare_result = _read_json_dict(prepare_result_path)
    generation_units = _read_json_dict(generation_units_path)
    if not prepare_result or not generation_units:
        warnings.append("interactive trajectory boundary overlay skipped: boundary metadata unavailable")
        return [], 0, 0, warnings

    prepared_times = _prepared_ros_timestamps(prepare_result)
    if prepared_times is None:
        warnings.append(
            "interactive trajectory boundary overlay skipped: prepare_result.frame_index_map.prepared_to_ros_time_sec unavailable"
        )
        total_candidates, _markers = _collect_generation_unit_boundaries(generation_units)
        return [], int(total_candidates), 0, warnings

    total_candidates, candidates = _collect_generation_unit_boundaries(generation_units)
    if not candidates:
        warnings.append("interactive trajectory boundary overlay skipped: no generation-unit boundaries found")
        return [], int(total_candidates), 0, warnings

    rec_times = _float_array(rec_timestamps)
    rec_points = _xy_array(rec_xy)
    if rec_times is None or rec_points is None:
        warnings.append("interactive trajectory boundary overlay skipped: REC timestamps or coordinates unavailable")
        return [], int(total_candidates), int(len(candidates)), warnings
    if int(rec_times.shape[0]) != int(rec_points.shape[0]):
        warnings.append(
            "interactive trajectory boundary overlay skipped: REC timestamp/count mismatch timestamps={} points={}".format(
                int(rec_times.shape[0]),
                int(rec_points.shape[0]),
            )
        )
        return [], int(total_candidates), int(len(candidates)), warnings

    tolerance = float(nearest_time_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        tolerance = 0.03

    matched = []
    unmatched = 0
    for marker in candidates:
        frame_idx = int(marker["frame_idx"])
        if frame_idx < 0 or frame_idx >= int(prepared_times.shape[0]):
            unmatched += 1
            continue
        timestamp = float(prepared_times[frame_idx])
        if not np.isfinite(timestamp):
            unmatched += 1
            continue
        deltas = np.abs(rec_times - timestamp)
        if deltas.shape[0] == 0:
            unmatched += 1
            continue
        nearest_idx = int(np.argmin(deltas))
        nearest_delta = float(deltas[nearest_idx])
        if nearest_delta > tolerance:
            unmatched += 1
            continue
        x_value = float(rec_points[nearest_idx, 0])
        y_value = float(rec_points[nearest_idx, 1])
        matched.append(
            {
                "x": x_value,
                "y": y_value,
                "hover": _format_marker_hover(marker, timestamp, x_value, y_value),
            }
        )

    if unmatched:
        warnings.append(
            "interactive trajectory boundary overlay unmatched markers: total_candidates={} matched={} unmatched={} tolerance_sec={:.6f}".format(
                int(total_candidates),
                int(len(matched)),
                int(unmatched),
                float(tolerance),
            )
        )
    return matched, int(total_candidates), int(unmatched), warnings


def _add_trajectory_trace(fig, go, name, xy, timestamps, color, dash, line_width):
    points = _xy_array(xy)
    if points is None:
        return
    line = {"color": color, "width": float(line_width)}
    if dash:
        line["dash"] = dash
    fig.add_trace(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines",
            name=str(name),
            line=line,
            customdata=_trajectory_customdata(points.shape[0], timestamps),
            hovertemplate=(
                "%{fullData.name}<br>"
                "sample=%{customdata[0]}<br>"
                "timestamp=%{customdata[1]:.6f}<br>"
                "x=%{x:.4f}<br>"
                "y=%{y:.4f}<extra></extra>"
            ),
        )
    )


def _axis_ranges_for_equal_scale(gt_xy, ori_xy, rec_xy, markers, width, height, margin):
    chunks = []
    for values in (gt_xy, ori_xy, rec_xy):
        points = _xy_array(values)
        if points is None:
            continue
        finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
        if np.any(finite):
            chunks.append(points[finite])

    marker_points = []
    for item in list(markers or []):
        try:
            x_value = float(item["x"])
            y_value = float(item["y"])
        except Exception:
            continue
        if np.isfinite(x_value) and np.isfinite(y_value):
            marker_points.append([x_value, y_value])
    if marker_points:
        chunks.append(np.asarray(marker_points, dtype=np.float64))

    if not chunks:
        return None, None

    points = np.concatenate(chunks, axis=0)
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)
    fallback_span = max(x_span, y_span, 1.0)
    if not np.isfinite(x_span) or x_span <= 0:
        x_span = fallback_span
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = fallback_span

    x_span *= 1.08
    y_span *= 1.08

    plot_width = float(width) - float(margin.get("l", 0)) - float(margin.get("r", 0))
    plot_height = float(height) - float(margin.get("t", 0)) - float(margin.get("b", 0))
    target_aspect = plot_width / plot_height if plot_width > 0 and plot_height > 0 else 1.3
    target_aspect = min(max(float(target_aspect), 1.25), 1.35)

    current_aspect = x_span / y_span if y_span > 0 else target_aspect
    if current_aspect < target_aspect:
        x_span = y_span * target_aspect
    elif current_aspect > target_aspect:
        y_span = x_span / target_aspect

    x_half = 0.5 * x_span
    y_half = 0.5 * y_span
    return [x_center - x_half, x_center + x_half], [y_center - y_half, y_center + y_half]


def write_interactive_trajectory_html(
    output_path,
    exp_dir,
    gt_xy,
    ori_xy,
    rec_xy,
    selected_plane,
    xlabel,
    ylabel,
    gt_timestamps=None,
    ori_timestamps=None,
    rec_timestamps=None,
    prepare_result_path=None,
    generation_units_path=None,
    nearest_time_tolerance=0.03,
):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        return {
            "trajectory_interactive_status": "skipped_plotly_unavailable",
            "trajectory_interactive_path": None,
            "trajectory_interactive_marker_count": 0,
            "trajectory_interactive_unmatched_marker_count": 0,
            "warnings": ["interactive trajectory HTML skipped: Plotly unavailable ({})".format(exc)],
            "marker_warnings": [],
            "messages": [],
        }

    output_path = Path(output_path).resolve()
    warnings = []
    marker_warnings = []
    markers = []
    total_candidates = 0
    unmatched = 0
    try:
        markers, total_candidates, unmatched, marker_warnings = _map_boundary_markers(
            prepare_result_path=prepare_result_path,
            generation_units_path=generation_units_path,
            rec_xy=rec_xy,
            rec_timestamps=rec_timestamps,
            nearest_time_tolerance=nearest_time_tolerance,
        )
    except Exception as exc:
        unmatched = 0
        markers = []
        marker_warnings.append("interactive trajectory boundary overlay skipped: {}".format(exc))

    fig = go.Figure()
    _add_trajectory_trace(fig, go, "GT", gt_xy, gt_timestamps, "#222222", None, line_width=6.0)
    _add_trajectory_trace(fig, go, "ORI", ori_xy, ori_timestamps, "#4C78A8", "dash", line_width=5.0)
    _add_trajectory_trace(fig, go, "REC", rec_xy, rec_timestamps, "#C06C5B", None, line_width=6.0)

    if markers:
        fig.add_trace(
            go.Scattergl(
                x=[float(item["x"]) for item in markers],
                y=[float(item["y"]) for item in markers],
                mode="markers",
                name="Visual anchors / boundaries",
                marker={
                    "color": "#2CA02C",
                    "size": 18,
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 3},
                },
                text=[str(item["hover"]) for item in markers],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    title = "Trajectory Overlay ({})".format(str(selected_plane or "auto").upper())
    figure_width = 1200
    figure_height = 900
    margin = {"l": 70, "r": 50, "t": 30, "b": 70}
    x_range, y_range = _axis_ranges_for_equal_scale(
        gt_xy=gt_xy,
        ori_xy=ori_xy,
        rec_xy=rec_xy,
        markers=markers,
        width=figure_width,
        height=figure_height,
        margin=margin,
    )

    xaxis = {
        "title": {"text": str(xlabel or "x"), "font": {"size": 24}},
        "tickfont": {"size": 20},
        "showline": True,
        "linewidth": 2.5,
        "linecolor": "#222222",
        "ticks": "outside",
        "tickwidth": 2.5,
        "ticklen": 8,
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.12)",
        "gridwidth": 1,
        "zeroline": False,
    }
    yaxis = {
        "title": {"text": str(ylabel or "y"), "font": {"size": 24}},
        "tickfont": {"size": 20},
        "showline": True,
        "linewidth": 2.5,
        "linecolor": "#222222",
        "ticks": "outside",
        "tickwidth": 2.5,
        "ticklen": 8,
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.12)",
        "gridwidth": 1,
        "zeroline": False,
        "scaleanchor": "x",
        "scaleratio": 1,
    }
    if x_range is not None and y_range is not None:
        xaxis["range"] = x_range
        yaxis["range"] = y_range

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        autosize=False,
        width=figure_width,
        height=figure_height,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 20},
        },
        margin=margin,
        xaxis=xaxis,
        yaxis=yaxis,
        modebar={"orientation": "h"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True, config={"responsive": False})

    marker_count = int(len(markers))
    messages = []
    if marker_count > 0:
        messages.append(
            "interactive trajectory boundary overlay: total_candidates={} matched={} unmatched={}".format(
                int(total_candidates),
                int(marker_count),
                int(unmatched),
            )
        )
        status = "success"
    else:
        marker_warnings.append(
            "interactive trajectory HTML saved without boundary overlays because no boundary metadata was found or matched."
        )
        status = "ok_without_markers"

    return {
        "trajectory_interactive_status": status,
        "trajectory_interactive_path": _relative_path(exp_dir, output_path),
        "trajectory_interactive_marker_count": marker_count,
        "trajectory_interactive_unmatched_marker_count": int(unmatched),
        "warnings": warnings,
        "marker_warnings": marker_warnings,
        "messages": messages,
    }
