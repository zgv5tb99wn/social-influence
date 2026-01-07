import os
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel


# Define the structured output schemas for OpenAI

class PsychologicalDepthAnalysis(BaseModel):
    """Analysis of psychological depth in responses."""
    before_level: Literal["powierzchowny", "średni", "głęboki"]
    after_level: Literal["powierzchowny", "średni", "głęboki"]
    before_focus: Literal["psychologiczny", "mieszany", "behawioralny/zewnętrzny"]
    after_focus: Literal["psychologiczny", "mieszany", "behawioralny/zewnętrzny"]
    change_direction: Literal["bardziej psychologiczny", "bardziej behawioralny", "bez zmian"]


class ReactionTypes(BaseModel):
    """Categorization of reaction types."""
    emocjonalne: int
    kognitywne: int
    behawioralne_manifestacje: int
    zewnętrzne_wyniki: int


class ReactionTypesAnalysis(BaseModel):
    """Analysis of reaction types before and after."""
    before_types: ReactionTypes
    after_types: ReactionTypes
    emotional_granularity_change: str


class ConceptualCoverage(BaseModel):
    """Coverage of conceptual categories."""
    before_categories: List[str]
    after_categories: List[str]
    maintained_categories: List[str]
    new_categories: List[str]
    lost_categories: List[str]


class EvolutionAssessment(BaseModel):
    """Assessment of how understanding evolved."""
    type: str
    description: str
    quality_change: float


class ReactionAnalysisResult(BaseModel):
    """Schema for reaction analysis result from OpenAI."""
    semantic_similarity: float
    psychological_depth_analysis: PsychologicalDepthAnalysis
    reaction_types_analysis: ReactionTypesAnalysis
    conceptual_coverage: ConceptualCoverage
    evolution_assessment: EvolutionAssessment
    key_insights: str


# Nested models for Intention Analysis
class ConceptualDepthAnalysis(BaseModel):
    """Analysis of conceptual depth in responses."""
    before_level: Literal["powierzchowny", "średni", "głęboki"]
    after_level: Literal["powierzchowny", "średni", "głęboki"]
    before_focus: Literal["koncepcyjny", "mieszany", "konkretny"]
    after_focus: Literal["koncepcyjny", "mieszany", "konkretny"]
    change_direction: Literal["bardziej koncepcyjny", "bardziej konkretny", "bez zmian"]


class IntentionAnalysisResult(BaseModel):
    """Schema for intention analysis result from OpenAI."""
    semantic_similarity: float
    conceptual_depth_analysis: ConceptualDepthAnalysis
    conceptual_coverage: ConceptualCoverage
    evolution_assessment: EvolutionAssessment
    key_insights: str


class AnnotatorCompetenceAnalyzer:
    """Analyzes annotator competences using OpenAI structured outputs."""

    def __init__(
        self,
        openai_api_key: str,
        excel_path: str,
        max_workers: int = 2,
        request_delay: float = 1.0,
        max_retries: int = 5
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.excel_path = excel_path
        self.max_workers = max_workers
        self.request_delay = request_delay  # Delay between requests in seconds
        self.max_retries = max_retries
        self.template_dir = Path(__file__).parent
        self.last_request_time = 0  # Track last request time for rate limiting

        # Set up Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Load templates
        self.reaction_system_template = self.jinja_env.get_template('reaction_prompt.j2')
        self.reaction_user_template = self.jinja_env.get_template('reaction_user_prompt.j2')
        self.intention_system_template = self.jinja_env.get_template('intention_prompt.j2')
        self.intention_user_template = self.jinja_env.get_template('intention_user_prompt.j2')

    def load_data(self) -> pd.DataFrame:
        return pd.read_excel(self.excel_path)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [
            'first_question_4_intention',
            'repeated_question_4_intention',
            'first_question_9_reaction',
            'repeated_question_9_reaction',
        ]

        additional_columns = [
            'id',
            'example',
            'annotator',
            'first_question_10_certainty',
            'repeated_question_10_certainty',
            'annotator_group',
            'first_question_3_intention_clarity',
            'repeated_question_3_intention_clarity',
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Include additional columns if they exist
        all_columns = required_columns + [col for col in additional_columns if col in df.columns]
        return df[all_columns].copy()

    def render_prompts(
        self,
        row: pd.Series,
        analysis_type: Literal["reaction", "intention"]
    ) -> tuple[str, str]:
        """Render system and user prompts for a given analysis type."""
        if analysis_type == "reaction":
            system_prompt = self.reaction_system_template.render()
            user_prompt = self.reaction_user_template.render(
                before=row['first_question_9_reaction'],
                after=row['repeated_question_9_reaction']
            )
        elif analysis_type == "intention":
            system_prompt = self.intention_system_template.render()
            user_prompt = self.intention_user_template.render(
                before=row['first_question_4_intention'],
                after=row['repeated_question_4_intention']
            )
        else:
            raise ValueError(f"Invalid analysis type: {analysis_type}")

        return system_prompt, user_prompt

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Recursively flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings or join if simple strings
                if v and isinstance(v[0], str):
                    items.append((new_key, '; '.join(v)))
                else:
                    items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _rate_limited_api_call(self, system_prompt: str, user_prompt: str, response_format):
        """Make API call with rate limiting and retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting: ensure minimum delay between requests
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.request_delay:
                    sleep_time = self.request_delay - time_since_last_request
                    time.sleep(sleep_time)

                response = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=response_format,
                    temperature=0
                )

                # Update last request time
                self.last_request_time = time.time()
                return response

            except RateLimitError as e:
                # Exponential backoff for rate limit errors
                wait_time = (2 ** attempt) * 2  
                print(f"  Rate limit hit (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)

                if attempt == self.max_retries - 1:
                    raise  # Re-raise on final attempt

            except APIError as e:
                # Retry on API errors with exponential backoff
                wait_time = (2 ** attempt) * 1  
                print(f"  API error (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)

                if attempt == self.max_retries - 1:
                    raise  # Re-raise on final attempt

    def analyze_single_row(
        self,
        row: pd.Series,
        index: int,
        analysis_type: Literal["reaction", "intention"]
    ) -> dict:
        system_prompt, user_prompt = self.render_prompts(row, analysis_type)
        response_format = (
            ReactionAnalysisResult if analysis_type == "reaction"
            else IntentionAnalysisResult
        )

        try:
            response = self._rate_limited_api_call(system_prompt, user_prompt, response_format)
            result = response.choices[0].message.parsed

            # Convert the result to a flat dictionary
            result_dict = self._flatten_dict(result.model_dump())

            # Add index and analysis type
            result_dict['index'] = index
            result_dict['analysis_type'] = analysis_type

            return result_dict

        except Exception as e:
            print(f"Error processing row {index} for {analysis_type}: {e}")
            return {
                'index': index,
                'analysis_type': analysis_type,
                'error': str(e)
            }

    def _get_checkpoint_path(self, analysis_type: str) -> Path:
        """Get the checkpoint file path for a given analysis type."""
        return Path(__file__).parent / f"checkpoint_{analysis_type}.json"

    def _load_checkpoint(self, analysis_type: str) -> dict:
        """Load checkpoint if it exists."""
        checkpoint_path = self._get_checkpoint_path(analysis_type)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, analysis_type: str, results: List[dict]):
        """Save checkpoint to disk."""
        checkpoint_path = self._get_checkpoint_path(analysis_type)
        checkpoint_data = {str(r['index']): r for r in results}
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    def analyze_all_rows(
        self,
        df: pd.DataFrame,
        analysis_type: Literal["reaction", "intention"],
        save_every: int = 5
    ) -> List[dict]:
        """
        Process all rows in parallel using ThreadPoolExecutor.

        Args:
            df: DataFrame to process
            analysis_type: Type of analysis to run
            save_every: Save checkpoint every N completed rows
        """
        # Load existing checkpoint
        checkpoint = self._load_checkpoint(analysis_type)
        completed_indices = set(int(idx) for idx in checkpoint.keys())

        if completed_indices:
            print(f"Found checkpoint with {len(completed_indices)} completed rows. Resuming...")
            results = list(checkpoint.values())
        else:
            results = []

        # Filter out already completed rows
        pending_rows = [(idx, row) for idx, row in df.iterrows() if idx not in completed_indices]

        if not pending_rows:
            print(f"All rows already completed for {analysis_type} analysis!")
            return results

        print(f"Processing {len(pending_rows)} remaining rows...")
        completed_count = len(completed_indices)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.analyze_single_row, row, idx, analysis_type): idx
                for idx, row in pending_rows
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    print(f"Completed {analysis_type} analysis for row {idx + 1}/{len(df)} ({completed_count} total)")

                    if completed_count % save_every == 0:
                        self._save_checkpoint(analysis_type, results)
                        print(f"  → Checkpoint saved ({completed_count} rows)")

                except Exception as e:
                    print(f"Exception for row {idx} ({analysis_type}): {e}")
                    results.append({
                        'index': idx,
                        'analysis_type': analysis_type,
                        'error': str(e)
                    })
                    completed_count += 1

        self._save_checkpoint(analysis_type, results)
        print(f"Final checkpoint saved for {analysis_type}")

        return results

    def clear_checkpoints(self):
        """Remove all checkpoint files."""
        for analysis_type in ["reaction", "intention"]:
            checkpoint_path = self._get_checkpoint_path(analysis_type)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Removed checkpoint: {checkpoint_path}")

    def run(self, clear_checkpoints: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main execution method.

        Args:
            clear_checkpoints: If True, remove existing checkpoints and start fresh
        """
        df = self.load_data()
        print(f"Loaded {len(df)} rows")

        if clear_checkpoints:
            print("\nClearing existing checkpoints...")
            self.clear_checkpoints()

        print("Preparing data...")
        prepared_df = self.prepare_data(df)

        # Run reaction analysis
        print(f"\nRunning reaction analysis for {len(prepared_df)} rows...")
        reaction_results = self.analyze_all_rows(prepared_df, "reaction")
        reaction_df = pd.DataFrame(reaction_results)
        reaction_df = reaction_df.set_index('index')

        # Run intention analysis
        print(f"\nRunning intention analysis for {len(prepared_df)} rows...")
        intention_results = self.analyze_all_rows(prepared_df, "intention")
        intention_df = pd.DataFrame(intention_results)
        intention_df = intention_df.set_index('index')

        # Add prefixes to column names to distinguish them
        reaction_df = reaction_df.add_prefix('reaction_')
        intention_df = intention_df.add_prefix('intention_')

        # Join with original data
        final_df = prepared_df.join(reaction_df).join(intention_df)

        print(f"\nCompleted! Final DataFrame has {len(final_df)} rows and {len(final_df.columns)} columns")

        # Clean up checkpoints after successful completion
        print("\nCleaning up checkpoints...")
        self.clear_checkpoints()

        return final_df, reaction_df, intention_df


def main():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    excel_path = "social_influence/experiments/annotators_competences/zbior_30_AB_do_porownania.xlsx"
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    analyzer = AnnotatorCompetenceAnalyzer(openai_api_key, excel_path)
    final_df, reaction_df, intention_df = analyzer.run()

    output_dir = Path(__file__).parent
    final_output_path = output_dir / "annotated_results_full.csv"
    reaction_output_path = output_dir / "reaction_analysis.csv"
    intention_output_path = output_dir / "intention_analysis.csv"

    final_df.to_csv(final_output_path, index=False)
    reaction_df.to_csv(reaction_output_path)
    intention_df.to_csv(intention_output_path)

    print(f"\nResults saved to:")
    print(f"  - Full results: {final_output_path}")
    print(f"  - Reaction analysis: {reaction_output_path}")
    print(f"  - Intention analysis: {intention_output_path}")

    return final_df


if __name__ == "__main__":
    result = main()
    print("\nFirst few rows of results:")
    print(result.head())
