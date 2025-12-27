#!/usr/bin/env python3
"""
Gather results from all experiment log files and JSON results into a report.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORT_FILE = PROJECT_ROOT / "results_report.txt"


def gather_results():
    """Gather all results and create a report."""

    report_lines = []
    report_lines.append("=" * 160)
    report_lines.append("COMMUNITY DETECTION SPARSIFICATION EXPERIMENT - RESULTS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 160)
    report_lines.append("")

    # Collect all dataset results
    all_results = []

    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        results_file = dataset_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results.append(json.load(f))

    if not all_results:
        report_lines.append("No results found.")
        return "\n".join(report_lines)

    # Summary statistics
    report_lines.append(f"Total datasets processed: {len(all_results)}")
    report_lines.append("")

    # Per-dataset summaries
    for result in all_results:
        dataset = result['dataset']
        has_gt = result.get('has_ground_truth', False)

        report_lines.append("")
        report_lines.append("=" * 160)
        report_lines.append(f"DATASET: {dataset}")
        report_lines.append("=" * 160)
        report_lines.append(f"  Nodes: {result['n_nodes']:,}")
        report_lines.append(f"  Edges: {result['n_edges_original']:,}")
        report_lines.append(f"  Connected Components: {result['n_cc_original']}")
        report_lines.append(f"  Ground Truth: {'Yes' if has_gt else 'No'}")
        report_lines.append("")

        # Main table header
        report_lines.append(f"{'Config':<20} {'Edge%':>8} {'CC':>6} {'Mod':>8} {'NMI':>8} {'ARI':>8} {'CommSim':>8} {'Intra%':>8} {'Inter%':>8} {'Ratio':>8}")
        report_lines.append("-" * 110)

        for config_name, config in result['configs'].items():
            if 'error' in config:
                report_lines.append(f"{config_name:<20} ERROR: {config['error']}")
                continue

            edge_pct = config['edge_ratio'] * 100
            cc = config.get('n_cc', '-')
            mod = config['metrics'].get('modularity_original', '-')
            nmi = config['metrics'].get('nmi')
            ari = config['metrics'].get('ari')
            comm_sim = config.get('community_similarity')

            # Edge preservation stats (Leiden labels)
            edge_pres = config.get('edge_preservation', {})
            intra_pres = edge_pres.get('intra_preservation_rate')
            inter_pres = edge_pres.get('inter_preservation_rate')
            pres_ratio = edge_pres.get('preservation_ratio')

            cc_str = f"{cc:.0f}" if isinstance(cc, (int, float)) else str(cc)
            mod_str = f"{mod:.4f}" if isinstance(mod, float) else str(mod)
            nmi_str = f"{nmi:.4f}" if nmi is not None else "-"
            ari_str = f"{ari:.4f}" if ari is not None else "-"
            comm_sim_str = f"{comm_sim:.4f}" if comm_sim is not None else "-"
            intra_str = f"{intra_pres*100:.1f}%" if intra_pres is not None else "-"
            inter_str = f"{inter_pres*100:.1f}%" if inter_pres is not None else "-"
            ratio_str = f"{pres_ratio:.3f}" if pres_ratio is not None else "-"

            report_lines.append(f"{config_name:<20} {edge_pct:>7.1f}% {cc_str:>6} {mod_str:>8} {nmi_str:>8} {ari_str:>8} {comm_sim_str:>8} {intra_str:>8} {inter_str:>8} {ratio_str:>8}")

        # Ground truth analysis table (only if has ground truth)
        if has_gt:
            report_lines.append("")
            report_lines.append("--- Ground Truth Analysis ---")
            report_lines.append(f"{'Config':<20} {'GT_Intra%':>10} {'GT_Inter%':>10} {'GT_Ratio':>10} {'Misclass%':>10} {'GT_Mod':>10}")
            report_lines.append("-" * 80)

            for config_name, config in result['configs'].items():
                if 'error' in config:
                    continue

                gt_pres = config.get('gt_edge_preservation', {})
                gt_intra = gt_pres.get('gt_intra_preservation_rate') if gt_pres else None
                gt_inter = gt_pres.get('gt_inter_preservation_rate') if gt_pres else None
                gt_ratio = gt_pres.get('gt_preservation_ratio') if gt_pres else None

                misclass = config.get('misclassification', {})
                misclass_rate = misclass.get('removed_misclassification_rate') if misclass else None

                gt_mod = config.get('gt_modularity')

                gt_intra_str = f"{gt_intra*100:.1f}%" if gt_intra is not None else "-"
                gt_inter_str = f"{gt_inter*100:.1f}%" if gt_inter is not None else "-"
                gt_ratio_str = f"{gt_ratio:.3f}" if gt_ratio is not None else "-"
                misclass_str = f"{misclass_rate*100:.1f}%" if misclass_rate is not None else "-"
                gt_mod_str = f"{gt_mod:.4f}" if gt_mod is not None else "-"

                report_lines.append(f"{config_name:<20} {gt_intra_str:>10} {gt_inter_str:>10} {gt_ratio_str:>10} {misclass_str:>10} {gt_mod_str:>10}")

        report_lines.append("")

    # Combined summary table
    report_lines.append("")
    report_lines.append("=" * 160)
    report_lines.append("COMBINED SUMMARY (ALL DATASETS)")
    report_lines.append("=" * 160)
    report_lines.append("")
    report_lines.append(f"{'Dataset':<18} {'Config':<20} {'Edge%':>8} {'CC':>6} {'Mod':>8} {'NMI':>8} {'ARI':>8} {'CommSim':>8} {'Intra%':>8} {'Inter%':>8} {'Ratio':>8}")
    report_lines.append("-" * 130)

    for result in all_results:
        dataset = result['dataset']

        for config_name, config in result['configs'].items():
            if 'error' in config:
                continue

            edge_pct = config['edge_ratio'] * 100
            cc = config.get('n_cc', '-')
            mod = config['metrics'].get('modularity_original', '-')
            nmi = config['metrics'].get('nmi')
            ari = config['metrics'].get('ari')
            comm_sim = config.get('community_similarity')

            edge_pres = config.get('edge_preservation', {})
            intra_pres = edge_pres.get('intra_preservation_rate')
            inter_pres = edge_pres.get('inter_preservation_rate')
            pres_ratio = edge_pres.get('preservation_ratio')

            cc_str = f"{cc:.0f}" if isinstance(cc, (int, float)) else str(cc)
            mod_str = f"{mod:.4f}" if isinstance(mod, float) else str(mod)
            nmi_str = f"{nmi:.4f}" if nmi is not None else "-"
            ari_str = f"{ari:.4f}" if ari is not None else "-"
            comm_sim_str = f"{comm_sim:.4f}" if comm_sim is not None else "-"
            intra_str = f"{intra_pres*100:.1f}%" if intra_pres is not None else "-"
            inter_str = f"{inter_pres*100:.1f}%" if inter_pres is not None else "-"
            ratio_str = f"{pres_ratio:.3f}" if pres_ratio is not None else "-"

            report_lines.append(f"{dataset:<18} {config_name:<20} {edge_pct:>7.1f}% {cc_str:>6} {mod_str:>8} {nmi_str:>8} {ari_str:>8} {comm_sim_str:>8} {intra_str:>8} {inter_str:>8} {ratio_str:>8}")

        report_lines.append("-" * 130)

    # Ground truth summary for all datasets with GT
    gt_datasets = [r for r in all_results if r.get('has_ground_truth', False)]
    if gt_datasets:
        report_lines.append("")
        report_lines.append("=" * 160)
        report_lines.append("GROUND TRUTH ANALYSIS SUMMARY")
        report_lines.append("=" * 160)
        report_lines.append("")
        report_lines.append(f"{'Dataset':<18} {'Config':<20} {'GT_Intra%':>10} {'GT_Inter%':>10} {'GT_Ratio':>10} {'Misclass%':>10} {'GT_Mod':>10}")
        report_lines.append("-" * 100)

        for result in gt_datasets:
            dataset = result['dataset']

            for config_name, config in result['configs'].items():
                if 'error' in config:
                    continue

                gt_pres = config.get('gt_edge_preservation', {})
                gt_intra = gt_pres.get('gt_intra_preservation_rate') if gt_pres else None
                gt_inter = gt_pres.get('gt_inter_preservation_rate') if gt_pres else None
                gt_ratio = gt_pres.get('gt_preservation_ratio') if gt_pres else None

                misclass = config.get('misclassification', {})
                misclass_rate = misclass.get('removed_misclassification_rate') if misclass else None

                gt_mod = config.get('gt_modularity')

                gt_intra_str = f"{gt_intra*100:.1f}%" if gt_intra is not None else "-"
                gt_inter_str = f"{gt_inter*100:.1f}%" if gt_inter is not None else "-"
                gt_ratio_str = f"{gt_ratio:.3f}" if gt_ratio is not None else "-"
                misclass_str = f"{misclass_rate*100:.1f}%" if misclass_rate is not None else "-"
                gt_mod_str = f"{gt_mod:.4f}" if gt_mod is not None else "-"

                report_lines.append(f"{dataset:<18} {config_name:<20} {gt_intra_str:>10} {gt_inter_str:>10} {gt_ratio_str:>10} {misclass_str:>10} {gt_mod_str:>10}")

            report_lines.append("-" * 100)

    report_lines.append("=" * 160)
    report_lines.append("")
    report_lines.append("END OF REPORT")

    return "\n".join(report_lines)


def main():
    report = gather_results()

    # Print to console
    print(report)

    # Save to file
    with open(REPORT_FILE, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == '__main__':
    main()
