from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
RAW_DIR = RESULTS_DIR / "plr_raw"

# --- Test protocol timing (seconds) ---
SETTLE_DURATION_S = 3.0       # shown but NOT recorded - lets you get into position first
BASELINE_DURATION_S = 3.0     # dim screen, recorded, pupil should be stable by now
FLASH_DURATION_S = 0.3        # bright flash duration
RECOVERY_DURATION_S = 7.0     # observe re-dilation after flash
TOTAL_DURATION_S = BASELINE_DURATION_S + FLASH_DURATION_S + RECOVERY_DURATION_S

# Only search for peak constriction within this many seconds after flash onset.
# Real PLR constriction peaks within ~1s of light onset - searching the full
# recovery window risks picking up blinks or end-of-recording artifacts instead.
CONSTRICTION_SEARCH_WINDOW_S = 2.0

# --- Display ---
DIM_COLOR_BGR = (90, 90, 90)
FLASH_COLOR_BGR = (255, 255, 255)
STIMULUS_WINDOW_SIZE = (900, 600)  # width, height

# --- Signal processing ---
SMOOTHING_WINDOW = 5          # frames, simple moving average
MIN_VALID_FRAME_RATIO = 0.5   # warn if fewer than this fraction of frames had a detection

for d in (RESULTS_DIR, PLOTS_DIR, METRICS_DIR, RAW_DIR):
    d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Raw output dir:", RAW_DIR)
