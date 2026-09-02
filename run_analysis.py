from src.analysis_pipeline import export_outputs
from src.analysis_pipeline import OUTPUT_DIR


if __name__ == '__main__':
    export_outputs()
    print(f'Analysis complete. Outputs written to {OUTPUT_DIR} ({len(list(OUTPUT_DIR.glob("*.csv")))} CSV files).')
