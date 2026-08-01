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
CODE_INPUTS: list[str] | dict[str, str] = {
	# "sum": r"D:\\NFF_ModelDeveloper\\models\\save\\SURFACE_ANODE_DiNO_Sum_0730\\train_history.csv",
	# "arcface": r"D:\\NFF_ModelDeveloper\\models\\save\\SURFACE_ANODE_DiNO_Concat_SCArc_0730\\train_history.csv",
}

CODE_OPTIONS = {
	"out_dir": "graph_outputs",
	"compare_metric": "Valid f1",
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
	smooth_window: int = 1,
	dpi: int = 140,
) -> None:
	df = _read_history(csv_path)
	run_name = csv_path.stem

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
	save_path = out_dir / f"{_sanitize_name(run_name)}_all_metrics.png"
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

		save_path = out_dir / f"{_sanitize_name(run_name)}_{_sanitize_name(base_name)}.png"
		fig.savefig(save_path, dpi=dpi)
		plt.close(fig)


def plot_multi_run_compare(
	csv_paths: list[Path],
	out_dir: Path,
	compare_metric: str = "Valid f1",
	smooth_window: int = 1,
	dpi: int = 140,
) -> None:
	# 여러 run에서 한 metric만 겹쳐 비교
	fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
	has_any = False

	for csv_path in csv_paths:
		df = _read_history(csv_path)
		if compare_metric not in df.columns:
			continue

		run_name = csv_path.stem
		ax.plot(
			df["epoch"],
			_smooth(df[compare_metric], smooth_window),
			label=run_name,
			linewidth=2,
		)
		has_any = True

	if not has_any:
		plt.close(fig)
		return

	_style_plot(ax, f"Multi-run Comparison | {compare_metric}")
	ax.legend()

	save_path = out_dir / f"compare_{_sanitize_name(compare_metric)}.png"
	fig.savefig(save_path, dpi=dpi)
	plt.close(fig)


def print_summary_table(csv_paths: list[Path]) -> None:
	rows = []

	for csv_path in csv_paths:
		df = _read_history(csv_path)
		row = {"run": csv_path.stem, "epochs": len(df)}

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
		smooth_window = int(CODE_OPTIONS.get("smooth_window", 1))
		dpi = int(CODE_OPTIONS.get("dpi", 140))
	else:
		entries = collect_labeled_entries(args.inputs)
		out_dir = Path(args.out_dir)
		compare_metric = args.compare_metric
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
			smooth_window=max(1, smooth_window),
			dpi=dpi,
		)

		# 파일명 stem 대신 사용자가 넣은 라벨로 별도 대표 그래프 1장 추가
		df = _read_history(csv_path)
		fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
		axes = axes.flatten()
		for ax, (base_name, (train_col, valid_col)) in zip(axes, METRIC_PAIRS.items()):
			if train_col not in df.columns or valid_col not in df.columns:
				ax.set_visible(False)
				continue
			ax.plot(df["epoch"], _smooth(df[train_col], max(1, smooth_window)), label=train_col, linewidth=2)
			ax.plot(df["epoch"], _smooth(df[valid_col], max(1, smooth_window)), label=valid_col, linewidth=2)
			_style_plot(ax, f"{run_label} | {base_name}")
			ax.legend()
		fig.suptitle(f"Training Curves: {run_label}", fontsize=13)
		fig.savefig(out_dir / f"{_sanitize_name(run_label)}_all_metrics.png", dpi=dpi)
		plt.close(fig)

	# 여러 run 비교 그래프 생성
	if len(entries) >= 2:
		# 기존 함수 재사용을 위해 path list를 구성하되,
		# 별칭 라벨 비교 그래프는 아래에서 한 번 더 그린다.
		csv_paths = [p for _, p in entries]
		plot_multi_run_compare(
			csv_paths=csv_paths,
			out_dir=out_dir,
			compare_metric=compare_metric,
			smooth_window=max(1, smooth_window),
			dpi=dpi,
		)

		fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
		has_any = False
		for run_label, csv_path in entries:
			df = _read_history(csv_path)
			if compare_metric not in df.columns:
				continue
			ax.plot(df["epoch"], _smooth(df[compare_metric], max(1, smooth_window)), label=run_label, linewidth=2)
			has_any = True
		if has_any:
			_style_plot(ax, f"Multi-run Comparison | {compare_metric}")
			ax.legend()
			fig.savefig(out_dir / f"compare_{_sanitize_name(compare_metric)}_labeled.png", dpi=dpi)
		plt.close(fig)

	# 콘솔 요약표 출력
	csv_paths = [p for _, p in entries]
	print_summary_table(csv_paths)

	print(f"\nGraphs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
	main()
