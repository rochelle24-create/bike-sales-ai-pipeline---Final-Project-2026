# flow/pipeline.py
# CrewAI Flow - Main Pipeline Orchestrator
# Author: Rachel Barazani - AI Developer
# Course: AI Developer Program - Hebrew University 2026

import os
import sys
import json
import pathlib
import time
import pandas as pd

# Allow running as: python flow/pipeline.py
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from crewai.flow.flow import Flow, listen, start
from utils.progress import progress
from utils.llm_selector import get_llm
from crew1_analysts.tools import (
    load_and_clean_data,
    generate_eda_report,
    generate_insights_and_contract,
)
from crew2_datascientists.tools import (
    engineer_features,
    train_models,
    evaluate_models,
    write_model_card,
)


def validate_crew1_outputs():
    """Validation Checkpoint 1 - Between Crew 1 and Crew 2."""
    progress.validation_header()

    progress.validation_check(
        "clean_data.csv exists",
        os.path.exists("artifacts/clean_data.csv"),
    )
    progress.validation_check(
        "dataset_contract.json exists",
        os.path.exists("artifacts/dataset_contract.json"),
    )
    progress.validation_check(
        "insights.md exists",
        os.path.exists("artifacts/insights.md"),
    )
    progress.validation_check(
        "eda_report.html exists",
        os.path.exists("artifacts/eda_report.html"),
    )

    df = pd.read_csv("artifacts/clean_data.csv")
    with open("artifacts/dataset_contract.json", "r") as f:
        contract = json.load(f)

    contract_cols = list(contract["columns"].keys())
    missing = [c for c in contract_cols if c not in df.columns]
    progress.validation_check(
        "Contract columns match clean_data.csv",
        len(missing) == 0,
    )
    progress.validation_check(
        "No nulls in Payment_Method",
        df["Payment_Method"].isna().sum() == 0,
    )
    progress.validation_check(
        "No nulls in Customer_Age",
        df["Customer_Age"].isna().sum() == 0,
    )
    progress.validation_check(
        f"Clean data has sufficient rows ({len(df):,})",
        len(df) > 1000,
    )
    progress.validation_passed()


def validate_crew2_inputs():
    """Validation Checkpoint 2 - Before Model Training."""
    progress.validation_header()

    progress.validation_check(
        "features.csv exists",
        os.path.exists("artifacts/features.csv"),
    )

    df = pd.read_csv("artifacts/features.csv")
    required_features = [
        "Quantity", "Bike_Model_enc", "Is_Cash",
        "Month", "Season_enc", "Age_Group_enc",
        "Price_Tier_enc", "Is_Weekend", "Day_of_Week",
    ]
    missing = [f for f in required_features if f not in df.columns]
    progress.validation_check(
        "All required feature columns exist",
        len(missing) == 0,
    )
    for target in ["Quantity", "Bike_Model_enc", "Is_Cash"]:
        progress.validation_check(
            f"Target column exists: {target}",
            target in df.columns,
        )
    clean_df = pd.read_csv("artifacts/clean_data.csv")
    progress.validation_check(
        f"Feature row count matches clean data ({len(df):,})",
        len(df) == len(clean_df),
    )
    progress.validation_passed()


class BikeSalesPipeline(Flow):
    """Orchestrates Crew 1 -> Validation -> Crew 2."""

    @start()
    def run_crew1(self):
        """Run Crew 1 - Data Analyst tools directly."""
        crew1_start = time.time()
        progress.crew_header("CREW 1 - DATA ANALYST CREW")

        result1 = load_and_clean_data.run("data/bike_sales_dirty.csv")
        progress.step(result1)

        result2 = generate_eda_report.run("artifacts/clean_data.csv")
        progress.step(result2)

        result3 = generate_insights_and_contract.run("artifacts/clean_data.csv")
        progress.step(result3)

        crew1_time = time.time() - crew1_start
        progress.crew_summary("CREW 1", crew1_time)
        return {"crew1_time": crew1_time}

    @listen(run_crew1)
    def validate_and_handoff(self, crew1_result):
        """Validation checkpoint between crews."""
        try:
            validate_crew1_outputs()
            return {
                "crew1_time": crew1_result["crew1_time"],
                "validation_passed": True,
            }
        except ValueError as e:
            progress.error("Validation Checkpoint 1", str(e))
            raise

    @listen(validate_and_handoff)
    def run_crew2(self, validation_result):
        """Run Crew 2 - Data Scientist tools directly."""
        crew2_start = time.time()
        progress.crew_header("CREW 2 - DATA SCIENTIST CREW")

        result4 = engineer_features.run("artifacts/clean_data.csv")
        progress.step(result4)

        validate_crew2_inputs()

        result5 = train_models.run("artifacts/features.csv")
        progress.step(result5)

        result6 = evaluate_models.run("artifacts/features.csv")
        progress.step(result6)

        result7 = write_model_card.run("artifacts/evaluation_report.md")
        progress.step(result7)

        crew2_time = time.time() - crew2_start
        progress.crew_summary("CREW 2", crew2_time)
        return {
            "crew1_time": validation_result["crew1_time"],
            "crew2_time": crew2_time,
        }

    @listen(run_crew2)
    def pipeline_done(self, result):
        """Final pipeline completion summary."""
        progress.pipeline_complete(
            crew1_time=result["crew1_time"],
            crew2_time=result["crew2_time"],
        )


def main():
    """Main entry point - run the full pipeline."""
    try:
        llm = get_llm()
        os.environ["SELECTED_BACKEND"] = (
            "Ollama"
            if hasattr(llm, "model") and "ollama" in str(llm.model).lower()
            else "Anthropic"
        )
        progress.pipeline_start_msg(os.environ.get("SELECTED_BACKEND", "Unknown"))
        BikeSalesPipeline().kickoff()
    except KeyboardInterrupt:
        print("\n\n[!]  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n\n[X] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
