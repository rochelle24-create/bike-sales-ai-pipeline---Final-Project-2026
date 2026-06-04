# flow/pipeline.py
# CrewAI Flow — Main Pipeline Orchestrator
# Author: Rachel Barazani — AI Developer
# Course: AI Developer Program — Hebrew University 2026

import os
import json
import pathlib
import time
import pandas as pd
from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, start
from utils.progress import progress
from utils.llm_selector import get_llm

# Ensure working directory is always the project root
os.chdir(pathlib.Path(__file__).parent.parent)

# Crew 1 imports
from crew1_analysts.agents import (
    data_ingestion_agent,
    eda_agent,
    contract_agent
)
from crew1_analysts.tasks import (
    task_ingest_clean,
    task_eda,
    task_contract
)
from crew1_analysts.tools import (
    load_and_clean_data,
    generate_eda_report,
    generate_insights_and_contract
)

# Crew 2 imports
from crew2_datascientists.agents import (
    feature_engineer,
    model_trainer,
    model_evaluator,
    model_card_writer
)
from crew2_datascientists.tasks import (
    task_feature_engineering,
    task_model_training,
    task_model_evaluation,
    task_model_card
)
from crew2_datascientists.tools import (
    engineer_features,
    train_models,
    evaluate_models,
    write_model_card
)


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_crew1_outputs():
    """
    Validation Checkpoint 1 — Between Crew 1 and Crew 2.
    Confirms all required artifacts exist and match the contract.
    """
    progress.validation_header()

    # Check 1 — clean_data.csv exists
    progress.validation_check(
        "clean_data.csv exists",
        os.path.exists("artifacts/clean_data.csv")
    )

    # Check 2 — dataset_contract.json exists
    progress.validation_check(
        "dataset_contract.json exists",
        os.path.exists("artifacts/dataset_contract.json")
    )

    # Check 3 — insights.md exists
    progress.validation_check(
        "insights.md exists",
        os.path.exists("artifacts/insights.md")
    )

    # Check 4 — eda_report.html exists
    progress.validation_check(
        "eda_report.html exists",
        os.path.exists("artifacts/eda_report.html")
    )

    # Check 5 — Contract columns match clean data
    df = pd.read_csv("artifacts/clean_data.csv")
    with open("artifacts/dataset_contract.json", "r") as f:
        contract = json.load(f)

    contract_cols = list(contract["columns"].keys())
    missing = [c for c in contract_cols if c not in df.columns]
    progress.validation_check(
        f"Contract columns match clean_data.csv",
        len(missing) == 0
    )

    # Check 6 — No nulls in critical columns
    progress.validation_check(
        "No nulls in Payment_Method",
        df["Payment_Method"].isna().sum() == 0
    )
    progress.validation_check(
        "No nulls in Customer_Age",
        df["Customer_Age"].isna().sum() == 0
    )

    # Check 7 — Row count is reasonable
    progress.validation_check(
        f"Clean data has sufficient rows ({len(df):,})",
        len(df) > 1000
    )

    progress.validation_passed()


def validate_crew2_inputs():
    """
    Validation Checkpoint 2 — Before Model Training.
    Confirms features.csv exists with all required columns.
    """
    progress.validation_header()

    # Check 1 — features.csv exists
    progress.validation_check(
        "features.csv exists",
        os.path.exists("artifacts/features.csv")
    )

    # Check 2 — Required feature columns exist
    df = pd.read_csv("artifacts/features.csv")
    required_features = [
        "Quantity", "Bike_Model_enc", "Is_Cash",
        "Month", "Season_enc", "Age_Group_enc",
        "Price_Tier_enc", "Is_Weekend", "Day_of_Week"
    ]
    missing = [f for f in required_features if f not in df.columns]
    progress.validation_check(
        "All required feature columns exist",
        len(missing) == 0
    )

    # Check 3 — Target columns exist
    for target in ["Quantity", "Bike_Model_enc", "Is_Cash"]:
        progress.validation_check(
            f"Target column exists: {target}",
            target in df.columns
        )

    # Check 4 — Row count matches
    clean_df = pd.read_csv("artifacts/clean_data.csv")
    progress.validation_check(
        f"Feature row count matches clean data ({len(df):,})",
        len(df) == len(clean_df)
    )

    progress.validation_passed()


# ============================================================
# CREWAI FLOW
# ============================================================

class BikeSalesPipeline(Flow):
    """
    Main CrewAI Flow for the Bike Sales AI Pipeline.
    Orchestrates Crew 1 → Validation → Crew 2.
    """

    @start()
    def run_crew1(self):
        """Run Crew 1 — Data Analyst Crew"""
        crew1_start = time.time()

        progress.crew_header("CREW 1 — DATA ANALYST CREW")

        # Assign tools to agents
        data_ingestion_agent.tools = [load_and_clean_data]
        eda_agent.tools = [generate_eda_report]
        contract_agent.tools = [generate_insights_and_contract]

        crew1 = Crew(
            agents=[data_ingestion_agent, eda_agent, contract_agent],
            tasks=[task_ingest_clean, task_eda, task_contract],
            process=Process.sequential,
            verbose=True
        )

        crew1.kickoff(inputs={"filepath": "data/bike_sales_dirty.csv"})

        crew1_time = time.time() - crew1_start
        progress.crew_summary("CREW 1", crew1_time)
        return {"crew1_time": crew1_time}

    @listen(run_crew1)
    def validate_and_handoff(self, crew1_result):
        """Validation Checkpoint — Between Crew 1 and Crew 2"""
        try:
            validate_crew1_outputs()
            return {
                "crew1_time": crew1_result["crew1_time"],
                "validation_passed": True
            }
        except ValueError as e:
            progress.error("Validation Checkpoint 1", str(e))
            raise

    @listen(validate_and_handoff)
    def run_crew2(self, validation_result):
        """Run Crew 2 — Data Scientist Crew"""
        crew2_start = time.time()

        progress.crew_header("CREW 2 — DATA SCIENTIST CREW")

        # Assign tools to agents
        feature_engineer.tools = [engineer_features]
        model_trainer.tools = [train_models]
        model_evaluator.tools = [evaluate_models]
        model_card_writer.tools = [write_model_card]

        crew2 = Crew(
            agents=[
                feature_engineer,
                model_trainer,
                model_evaluator,
                model_card_writer
            ],
            tasks=[
                task_feature_engineering,
                task_model_training,
                task_model_evaluation,
                task_model_card
            ],
            process=Process.sequential,
            verbose=True
        )

        # Validate inputs before training
        validate_crew2_inputs()

        crew2.kickoff(inputs={"filepath": "artifacts/clean_data.csv"})

        crew2_time = time.time() - crew2_start
        progress.crew_summary("CREW 2", crew2_time)

        return {
            "crew1_time": validation_result["crew1_time"],
            "crew2_time": crew2_time
        }

    @listen(run_crew2)
    def pipeline_done(self, result):
        """Final pipeline completion summary"""
        progress.pipeline_complete(
            crew1_time=result["crew1_time"],
            crew2_time=result["crew2_time"]
        )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point — run the full pipeline"""
    try:
        # Display startup banner and get LLM selection
        llm = get_llm()

        # Record backend name for logging
        os.environ["SELECTED_BACKEND"] = "Ollama" if hasattr(
            llm, 'model') and 'ollama' in str(llm.model) else "Anthropic"

        # Start pipeline
        progress.pipeline_start_msg(os.environ.get("SELECTED_BACKEND", "Unknown"))

        # Run the flow
        pipeline = BikeSalesPipeline()
        pipeline.kickoff()

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()