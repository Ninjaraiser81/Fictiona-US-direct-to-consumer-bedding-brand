from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_generates_required_outputs():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'run_analysis.py')],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    required = [
        'qa_report.csv',
        'month_1_baseline.csv',
        'month_comparison_matched.csv',
        'prompt_family_deltas.csv',
        'platform_deltas.csv',
        'fix_assessment.csv',
        'next_sprint_plan.csv',
        'supporting_analytics_summary.csv',
        'runs_clean.csv',
        'outcomes_clean.csv',
        'citations_clean.csv',
    ]

    for name in required:
        assert (ROOT / 'outputs' / name).exists(), f'Missing {name}'
