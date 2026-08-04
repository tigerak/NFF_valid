import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# 코드 내 고정 설정 모드
# -----------------------------------------------------------------------------
# True이면 아래 CODE_INPUTS/CODE_OPTIONS를 사용하고,
# False이면 기존 CLI 인자 입력 방식을 사용합니다.
USE_CODE_INPUTS = True

# 비교할 CSV 소스:
# - list 형태: ["path1.csv", "path2.csv", "folder_path"]
# - dict 형태: {"run_name": "path_or_folder", ...}
def get_path_input(dir_name) -> Path:
	formatted_dir = f"D:\\NFF_ModelDeveloper\\models\\save\\{dir_name}\\train_history.csv"
	return Path(formatted_dir)

CODE_INPUTS: list[str] | dict[str, str] = {
	"attn_fa1": get_path_input("0803_SURFACE_ANODE_DiNO_WtAttn_FA1"),
	"attn_fa05": get_path_input("0803_SURFACE_ANODE_DiNO_WtAttn_FA05"),
	"attn_fa025": get_path_input("0803_SURFACE_ANODE_DiNO_WtAttn_FA025"),
	"concat_fa1": get_path_input("0803_SURFACE_ANODE_DiNO_WtConcat_FA1"),
	"concat_fa05": get_path_input("0803_SURFACE_ANODE_DiNO_WtConcat_FA05"),
	"concat_fa025": get_path_input("0803_SURFACE_ANODE_DiNO_WtConcat_FA025"),
	"concat_hy_fa05_m01_s16": get_path_input("0803_SURFACE_ANODE_DiNO_WtConcat_FA05_Hybrid_ArcFace_M01_S16"),
	"attn_hy_fa05_m01_s16": get_path_input("0804_SURFACE_ANODE_DiNO_WtAttn_FA05_Hybrid_ArcFace_M01_S16"),
}

CODE_OPTIONS = {
	"out_dir": "graph_outputs",
	"compare_metric": "Valid f1",
	"compare_all_metrics": True,
	"smooth_window": 1,
	"dpi": 140,
}


# 기본 metric 매핑: train/valid 짝을 자동 인식하기 위한 기준
METRIC_PAIRS = {
	"Loss": ("Train Loss", "Valid Loss"),
	"Recall": ("Train Recall", "Valid Recall"),
	"f1": ("Train f1", "Valid f1"),
	"ACC": ("Train_ACC", "Valid ACC"),
}


# train_history.csv의 모니터링 컬럼(존재하는 항목만 자동 표시)
MONITOR_COLUMNS = [
	"Grad Norm",
	"Focal Grad Norm",
	"ArcFace Grad Norm",
	"Delta Theta Norm",
	"Update Ratio",
	"Attention Entropy",
	"Grad Cosim Focal ArcFace",
	"Fusion Alpha",
]


def _read_history(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)

	# 저장 포맷에 따라 첫 컬럼이 Unnamed index일 수 있어 정리
	unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
	if unnamed_cols:
		df = df.drop(columns=unnamed_cols)

	# epoch 축이 없으면 행 인덱스를 epoch로 간주
	if "epoch" not in df.columns:
		df = df.copy()
		df["epoch"] = range(1, len(df) + 1)

	return df


def _sanitize_name(text: str) -> str:
	keep = []
	for ch in text:
		if ch.isalnum() or ch in ("-", "_"):
			keep.append(ch)
		else:
			keep.append("_")
	return "".join(keep).strip("_")


def _style_plot(ax, title: str, xlabel: str = "Epoch", ylabel: str = "Value") -> None:
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True, alpha=0.25)


def _smooth(series: pd.Series, window: int) -> pd.Series:
	if window <= 1:
		return series
	return series.rolling(window=window, min_periods=1).mean()


def plot_single_run(
	csv_path: Path,
	out_dir: Path,
	run_label: str | None = None,
	smooth_window: int = 1,
	dpi: int = 140,
) -> None:
	df = _read_history(csv_path)
	run_name = run_label if run_label else csv_path.stem

	# 1) 한 장에 모든 metric pair(Train/Valid) subplot
	fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
	axes = axes.flatten()

	for ax, (base_name, (train_col, valid_col)) in zip(axes, METRIC_PAIRS.items()):
		if train_col not in df.columns or valid_col not in df.columns:
			ax.set_visible(False)
			continue

		ax.plot(df["epoch"], _smooth(df[train_col], smooth_window), label=train_col, linewidth=2)
		ax.plot(df["epoch"], _smooth(df[valid_col], smooth_window), label=valid_col, linewidth=2)
		_style_plot(ax, f"{run_name} | {base_name}")
		ax.legend()

	fig.suptitle(f"Training Curves: {run_name}", fontsize=13)
	save_path = out_dir / f"single_{_sanitize_name(run_name)}_all_metrics.png"
	fig.savefig(save_path, dpi=dpi)
	plt.close(fig)

	# 2) metric별 개별 그림 (보고서 삽입용)
	for base_name, (train_col, valid_col) in METRIC_PAIRS.items():
		if train_col not in df.columns or valid_col not in df.columns:
			continue

		fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
		ax.plot(df["epoch"], _smooth(df[train_col], smooth_window), label=train_col, linewidth=2)
		ax.plot(df["epoch"], _smooth(df[valid_col], smooth_window), label=valid_col, linewidth=2)
		_style_plot(ax, f"{run_name} | {base_name}")
		ax.legend()

		save_path = out_dir / f"single_{_sanitize_name(run_name)}_{_sanitize_name(base_name)}.png"
		fig.savefig(save_path, dpi=dpi)
		plt.close(fig)

	# 3) 모니터링 metric(grad/entropy/cosim) 개별 그림
	monitor_cols = [c for c in MONITOR_COLUMNS if c in df.columns]
	for col in monitor_cols:
		fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
		ax.plot(df["epoch"], _smooth(df[col], smooth_window), label=col, linewidth=2)
		_style_plot(ax, f"{run_name} | {col}")
		ax.legend()

		save_path = out_dir / f"single_{_sanitize_name(run_name)}_{_sanitize_name(col)}.png"
		fig.savefig(save_path, dpi=dpi)
		plt.close(fig)


def plot_multi_run_compare(
	entries: list[tuple[str, Path]],
	out_dir: Path,
	compare_metric: str = "Valid f1",
	smooth_window: int = 1,
	dpi: int = 140,
) -> None:
	# 여러 run에서 한 metric만 겹쳐 비교
	fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
	has_any = False
	missing_labels: list[str] = []
	compare_frames: list[pd.DataFrame] = []

	for run_label, csv_path in entries:
		df = _read_history(csv_path)
		if compare_metric not in df.columns:
			missing_labels.append(run_label)
			continue

		series = _smooth(df[compare_metric], smooth_window)
		ax.plot(
			df["epoch"],
			series,
			label=run_label,
			linewidth=2,
		)
		compare_frames.append(pd.DataFrame({"epoch": df["epoch"], run_label: series}))
		has_any = True

	if not has_any:
		if missing_labels:
			print(
				f"[WARN] compare_metric '{compare_metric}' not found in: "
				+ ", ".join(missing_labels)
			)
		plt.close(fig)
		return

	if missing_labels:
		print(
			f"[WARN] compare_metric '{compare_metric}' not found in: "
			+ ", ".join(missing_labels)
		)

	_style_plot(ax, f"Multi-run Comparison | {compare_metric}")
	ax.legend()

	save_path = out_dir / f"compare_{_sanitize_name(compare_metric)}_labeled.png"
	fig.savefig(save_path, dpi=dpi)
	plt.close(fig)

	# 비교 이미지에 실제 사용된 run_label만 epoch 기준으로 모아 CSV 저장
	compare_df = compare_frames[0]
	for frame in compare_frames[1:]:
		compare_df = compare_df.merge(frame, on="epoch", how="outer")
	compare_df = compare_df.sort_values("epoch").reset_index(drop=True)

	compare_csv_path = out_dir / f"compare_{_sanitize_name(compare_metric)}_labeled.csv"
	compare_df.to_csv(compare_csv_path, index=False)


def get_compare_metrics(entries: list[tuple[str, Path]]) -> list[str]:
	"""다중 run 비교 대상 metric 목록을 구성한다.

	- Train/Valid pair 컬럼
	- 모니터링 컬럼
	중 실제 CSV에 존재하는 항목만 반환한다.
	"""
	candidates: list[str] = []

	for train_col, valid_col in METRIC_PAIRS.values():
		candidates.append(train_col)
		candidates.append(valid_col)

	for col in MONITOR_COLUMNS:
		candidates.append(col)

	# 실제 존재하는 컬럼만 필터링
	present_cols = set()
	for _, csv_path in entries:
		df = _read_history(csv_path)
		present_cols.update(df.columns.tolist())

	metrics = [c for c in candidates if c in present_cols]

	# 순서 유지 중복 제거
	unique = []
	seen = set()
	for m in metrics:
		if m not in seen:
			seen.add(m)
			unique.append(m)

	return unique


def print_summary_table(entries: list[tuple[str, Path]]) -> None:
	rows = []

	for run_label, csv_path in entries:
		df = _read_history(csv_path)
		row = {"run": run_label, "epochs": len(df)}

		for key, (_, valid_col) in METRIC_PAIRS.items():
			if valid_col in df.columns:
				best_idx = df[valid_col].idxmax() if key != "Loss" else df[valid_col].idxmin()
				row[f"best_{valid_col}"] = float(df.loc[best_idx, valid_col])
				row[f"best_epoch_{valid_col}"] = int(df.loc[best_idx, "epoch"])

		rows.append(row)

	if not rows:
		return

	summary = pd.DataFrame(rows)
	print("\n=== Run Summary (best valid metrics) ===")
	print(summary.to_string(index=False))


def collect_csv_paths(inputs: list[str]) -> list[Path]:
	paths: list[Path] = []
	for item in inputs:
		p = Path(item)
		if p.is_file() and p.suffix.lower() == ".csv":
			paths.append(p)
		elif p.is_dir():
			paths.extend(sorted(p.rglob("*.csv")))

	# 중복 제거(순서 보존)
	unique = []
	seen = set()
	for p in paths:
		rp = p.resolve()
		if rp not in seen:
			seen.add(rp)
			unique.append(rp)
	return unique


def collect_labeled_entries(inputs: list[str] | dict[str, str]) -> list[tuple[str, Path]]:
	entries: list[tuple[str, Path]] = []

	if isinstance(inputs, dict):
		for run_name, item in inputs.items():
			p = Path(item)
			if p.is_file() and p.suffix.lower() == ".csv":
				entries.append((run_name, p.resolve()))
			elif p.is_dir():
				for csv_file in sorted(p.rglob("*.csv")):
					label = f"{run_name}:{csv_file.stem}"
					entries.append((label, csv_file.resolve()))
	else:
		for csv_path in collect_csv_paths(inputs):
			entries.append((csv_path.stem, csv_path))

	# (label, path) 중 path 기준 중복 제거
	unique: list[tuple[str, Path]] = []
	seen = set()
	for label, path in entries:
		if path not in seen:
			seen.add(path)
			unique.append((label, path))

	return unique


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"train_history.csv(또는 여러 csv)에서 epoch별 그래프를 생성합니다. "
			"단일 run train/valid 비교 + 여러 run 간 성능 비교를 지원합니다."
		)
	)
	parser.add_argument(
		"inputs",
		nargs="*",
		help="CSV 파일 경로 또는 CSV를 포함한 폴더 경로(여러 개 가능)",
	)
	parser.add_argument(
		"--out-dir",
		default="graph_outputs",
		help="그래프 저장 폴더 (기본: graph_outputs)",
	)
	parser.add_argument(
		"--compare-metric",
		default="Valid f1",
		help="여러 파일 비교 시 겹쳐 그릴 컬럼명 (기본: Valid f1)",
	)
	parser.add_argument(
		"--compare-all-metrics",
		dest="compare_all_metrics",
		action="store_true",
		help="여러 파일 비교 시 가능한 모든 metric을 run 간 비교 그래프로 생성",
	)
	parser.add_argument(
		"--single-compare-metric",
		dest="compare_all_metrics",
		action="store_false",
		help="여러 파일 비교 시 compare-metric 하나만 생성",
	)
	parser.set_defaults(compare_all_metrics=True)
	parser.add_argument(
		"--smooth-window",
		type=int,
		default=1,
		help="이동평균 window (기본: 1, smoothing 없음)",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=140,
		help="그래프 저장 DPI (기본: 140)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if USE_CODE_INPUTS:
		entries = collect_labeled_entries(CODE_INPUTS)
		out_dir = Path(CODE_OPTIONS.get("out_dir", "graph_outputs"))
		compare_metric = str(CODE_OPTIONS.get("compare_metric", "Valid f1"))
		compare_all_metrics = bool(CODE_OPTIONS.get("compare_all_metrics", True))
		smooth_window = int(CODE_OPTIONS.get("smooth_window", 1))
		dpi = int(CODE_OPTIONS.get("dpi", 140))
	else:
		entries = collect_labeled_entries(args.inputs)
		out_dir = Path(args.out_dir)
		compare_metric = args.compare_metric
		compare_all_metrics = bool(args.compare_all_metrics)
		smooth_window = args.smooth_window
		dpi = args.dpi

	if not entries:
		print("No CSV files found. Check inputs (CODE_INPUTS or CLI args).")
		return

	out_dir.mkdir(parents=True, exist_ok=True)

	print(f"Found {len(entries)} csv file(s).")
	for run_label, p in entries:
		print(f" - [{run_label}] {p}")

	# 각 run별 train/valid 추이 그래프 생성
	for run_label, csv_path in entries:
		plot_single_run(
			csv_path=csv_path,
			out_dir=out_dir,
			run_label=run_label,
			smooth_window=max(1, smooth_window),
			dpi=dpi,
		)

	# 여러 run 비교 그래프 생성
	if len(entries) >= 2:
		if compare_all_metrics:
			all_metrics = get_compare_metrics(entries)
			if not all_metrics:
				print("[WARN] No comparable metrics found across input CSV files.")
			for metric in all_metrics:
				plot_multi_run_compare(
					entries=entries,
					out_dir=out_dir,
					compare_metric=metric,
					smooth_window=max(1, smooth_window),
					dpi=dpi,
				)
		else:
			plot_multi_run_compare(
				entries=entries,
				out_dir=out_dir,
				compare_metric=compare_metric,
				smooth_window=max(1, smooth_window),
				dpi=dpi,
			)

	# 콘솔 요약표 출력
	print_summary_table(entries)

	print(f"\nGraphs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
	main()
