from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OFFERS_DIR = OUTPUT_DIR / "offers"

RESUME_CSV = DATA_DIR / "resume_dataset_1200.csv"
LEAVE_XLSX = DATA_DIR / "employee_leave_tracking_data.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OFFERS_DIR.mkdir(parents=True, exist_ok=True)
