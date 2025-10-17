import csv
import os
from datetime import datetime

class DialogLogger:
    """
    Logs dialog turns and autocorrections for the experiment.
    Creates timestamped log files for each participant session.
    """
    def __init__(self, participant_id="P000", log_dir=None):
        self.participant_id = participant_id

        if log_dir is None:
            home_dir = os.path.expanduser("~")  # C:\Users\nnif0
            mair_dir = os.path.join(home_dir, "MAIR")
            log_dir = os.path.join(mair_dir, "logs")

        os.makedirs(log_dir, exist_ok=True)
        print(f"📁 Log files created in: {os.path.abspath(log_dir)}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dialog_log_path = os.path.join(
            log_dir, f"{participant_id}_dialog_{timestamp}.csv"
        )
        self.autocorrect_log_path = os.path.join(
            log_dir, f"{participant_id}_autocorrect_{timestamp}.csv"
        )

        # Initialize log files with headers
        with open(self.dialog_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "participant_id", "state", "speaker", "utterance"])

        with open(self.autocorrect_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "participant_id", "original", "corrected", "utterance"])

    def log_turn(self, state, speaker, utterance):
        """Record one dialog turn."""
        with open(self.dialog_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), self.participant_id, state, speaker, utterance])

    def log_autocorrect(self, original, corrected, utterance):
        """Record one autocorrection (only if correction occurred)."""
        if original != corrected:
            with open(self.autocorrect_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), self.participant_id, original, corrected, utterance])
