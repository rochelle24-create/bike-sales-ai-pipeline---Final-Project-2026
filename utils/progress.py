# utils/progress.py
# Live progress display and logging for the Bike Sales AI Pipeline
# - Hebrew University 2026 Final Project
# Author: Rachel Barazani - AI Developer
# Course: AI Developer Program - Hebrew University 2026

import time
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Progress:
    """
    Live progress tracker for the Bike Sales AI Pipeline - Hebrew University 2026 Final Project.
    Prints real-time status to terminal and logs to logs/pipeline.log
    """

    def __init__(self):
        self.pipeline_start = None
        self.stage_start = None
        self.crew1_time = 0
        self.crew2_time = 0
        self.validation_time = 0

    def pipeline_start_msg(self, backend: str):
        """Print pipeline startup banner"""
        self.pipeline_start = time.time()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "+" + "=" * 58 + "+")
        print("|" + "BIKE SALES AI PIPELINE - HEBREW UNIVERSITY 2026".center(58) + "|")
        print("|" + "FINAL PROJECT - RUNNING".center(58) + "|")
        print("+" + "=" * 58 + "+")
        print(f"|  Backend: {backend:<47}|")
        print(f"|  Started: {now:<47}|")
        print("+" + "=" * 58 + "+")
        logger.info(f"Pipeline started - Backend: {backend}")

    def crew_header(self, crew_name: str):
        """Print crew section header"""
        print(f"\n{'=' * 60}")
        print(f" {crew_name}")
        print(f"{'=' * 60}\n")
        logger.info(f"STARTED: {crew_name}")

    def agent_start(self, agent_name: str):
        """Record agent start time and print status"""
        self.stage_start = time.time()
        print(f"  > {agent_name:<45} [STARTED]")
        logger.info(f"Agent started: {agent_name}")

    def step(self, message: str):
        """Print a sub-step progress message"""
        print(f"    -> {message}")
        logger.info(f"    {message}")

    def agent_done(self, agent_name: str):
        """Calculate elapsed time and print completion"""
        elapsed = time.time() - self.stage_start
        print(f"  [OK] {agent_name:<45} [{self._fmt(elapsed)}]")
        logger.info(f"Agent completed: {agent_name} in {self._fmt(elapsed)}")
        return elapsed

    def validation_header(self):
        """Print validation checkpoint header"""
        print(f"\n{'=' * 60}")
        print(f" CREWAI FLOW - VALIDATION CHECKPOINT")
        print(f"{'=' * 60}\n")
        self.stage_start = time.time()
        logger.info("Validation checkpoint started")

    def validation_check(self, check_name: str, passed: bool):
        """Print validation check result - raise on failure"""
        status = "[OK]" if passed else "[X]"
        print(f"  > {check_name:<45} {status}")
        logger.info(f"Validation: {check_name} - {'PASS' if passed else 'FAIL'}")
        if not passed:
            error_msg = f"Validation FAILED: {check_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def validation_passed(self):
        """Print validation success and time"""
        elapsed = time.time() - self.stage_start
        self.validation_time = elapsed
        print(f"\n  [OK] ALL CHECKS PASSED - Starting Crew 2  [{self._fmt(elapsed)}]")
        logger.info(f"Validation passed in {self._fmt(elapsed)}")

    def crew_summary(self, crew_name: str, elapsed: float):
        """Print crew completion summary"""
        print(f"\n{'=' * 60}")
        print(f" {crew_name} TOTAL TIME: {self._fmt(elapsed)}")
        print(f"{'=' * 60}")
        logger.info(f"{crew_name} completed in {self._fmt(elapsed)}")

    def pipeline_complete(self, crew1_time: float, crew2_time: float):
        """Print final pipeline completion summary"""
        total = crew1_time + self.validation_time + crew2_time
        print("\n" + "+" + "=" * 58 + "+")
        print("|" + "PIPELINE COMPLETE [OK]".center(58) + "|")
        print("+" + "=" * 58 + "+")
        print(f"|  Crew 1:      {self._fmt(crew1_time):<43}|")
        print(f"|  Validation:  {self._fmt(self.validation_time):<43}|")
        print(f"|  Crew 2:      {self._fmt(crew2_time):<43}|")
        print(f"|  TOTAL:       {self._fmt(total):<43}|")
        print("+" + "=" * 58 + "+")
        print(f"|  Artifacts saved to: artifacts/           |")
        print(f"|  Logs saved to:      logs/pipeline.log    |")
        print("+" + "=" * 58 + "+")
        print(f"|  Run Streamlit app:                       |")
        print(f"|  streamlit run app/streamlit_app.py       |")
        print("+" + "=" * 58 + "+\n")
        logger.info(f"Pipeline complete. Total time: {self._fmt(total)}")

    def error(self, agent_name: str, error_msg: str):
        """Print and log a graceful failure message"""
        print(f"\n{'=' * 60}")
        print(f"  [X] PIPELINE FAILED")
        print(f"  Agent:   {agent_name}")
        print(f"  Reason:  {error_msg}")
        print(f"  Check:   logs/pipeline.log for details")
        print(f"{'=' * 60}\n")
        logger.error(f"Pipeline failed at {agent_name}: {error_msg}")

    def _fmt(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


# Global instance - imported and used by all agents
progress = Progress()