# Annotators Competence Analysis

This project analyzes how annotator competences evolved over time by comparing their annotations before and after a time gap. It uses OpenAI's structured outputs to evaluate changes in reaction and intention annotations, then provides comprehensive statistical analysis through a Jupyter notebook.

## Overview

The analysis consists of two main components:

1. **Data Processing Script** (`annotators_competences.py`) - Processes Excel data using OpenAI API to analyze annotation quality changes
2. **Analysis Notebook** (`analysis_notebook.ipynb`) - Performs exploratory data analysis with visualizations and statistical tests

## Setup

### 1. Install Dependencies

This project uses Poetry for dependency management:

```bash
poetry install
```

Or if you prefer pip:

```bash
pip install openai pandas jinja2 openpyxl matplotlib seaborn scipy jupyter
```

### 2. Set up OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

### 3. Prepare Your Data

Place your Excel file at:
```
social_influence/experiments/annotators_competences/zbior_30_AB_do_porownania.xlsx
```

The Excel file should contain these columns:
- `id` - Unique identifier
- `example` - Example text
- `annotator` - Annotator identifier
- `annotator_group` - Group classification (e.g., "psycholog", "nauczyciel", "rodzic", "nastolatek", "komunikacja")
- `first_question_3_intention_clarity` - Initial intention clarity score
- `repeated_question_3_intention_clarity` - Repeated intention clarity score
- `first_question_4_intention` - Initial intention annotation
- `repeated_question_4_intention` - Repeated intention annotation
- `first_question_9_reaction` - Initial reaction annotation
- `repeated_question_9_reaction` - Repeated reaction annotation
- `first_question_10_certainty` - Initial certainty level (Polish text: "Niska pewność", "Umiarkowana pewność", "Wysoka pewność", "Całkowita pewność")
- `repeated_question_10_certainty` - Repeated certainty level

## Usage

### Step 1: Run the Analysis Script

```bash
cd /Users/ola.sawczuk/Code/priv/social-influence
poetry run python social_influence/experiments/annotators_competences/annotators_competences.py
```

This will:
- Load data from the Excel file
- Analyze reactions and intentions using OpenAI API with structured outputs
- Save results to CSV files in the same directory

**Features:**
- **Parallel processing** - Uses ThreadPoolExecutor for faster processing (configurable workers)
- **Rate limiting** - Built-in delays and exponential backoff for API rate limits
- **Checkpoint system** - Automatically saves progress every 5 rows and can resume from checkpoints
- **Structured outputs** - Uses Pydantic models for type-safe OpenAI responses
- **Nested analysis** - Evaluates psychological depth, reaction types, conceptual coverage, and quality changes

**Output files:**
- `annotated_results_full.csv` - Complete results with all original and analysis columns
- `reaction_analysis.csv` - Reaction-specific analysis results
- `intention_analysis.csv` - Intention-specific analysis results
- `checkpoint_reaction.json` - Checkpoint for reaction analysis (temporary)
- `checkpoint_intention.json` - Checkpoint for intention analysis (temporary)

### Step 2: Run the Analysis Notebook

```bash
cd social_influence/experiments/annotators_competences
poetry run jupyter notebook analysis_notebook.ipynb
```

The notebook provides comprehensive analysis:

1. **Data Overview** - Dataset statistics, missing values, annotator group distribution
2. **Reaction Analysis** - Quality change distributions (overall and per-group)
3. **Intention Analysis** - Quality change distributions (overall and per-group)
4. **Comparison** - Reaction vs Intention changes with correlations
5. **Depth Analysis** - Psychological and conceptual depth changes (before/after, per-group)
6. **Certainty Analysis** - Certainty level changes and distributions (overall and per-group)
7. **Correlations** - Certainty vs quality changes with correlation matrices and scatter plots
8. **Group Comparison** - Statistical comparisons across annotator groups
9. **Summary Statistics** - Overall metrics and key findings
10. **Data Export** - Processed data with all calculated metrics

## Files

### Core Files
- `annotators_competences.py` - Main processing script with OpenAI integration
- `analysis_notebook.ipynb` - Jupyter notebook for data analysis and visualization
- `pyproject.toml` - Poetry dependencies configuration

### Prompt Templates (Jinja2)
- `reaction_prompt.j2` - System prompt for reaction analysis (Polish)
- `reaction_user_prompt.j2` - User prompt template for reactions
- `intention_prompt.j2` - System prompt for intention analysis (Polish)
- `intention_user_prompt.j2` - User prompt template for intentions

### Output Files (Generated)
- `annotated_results_full.csv` - Complete annotated dataset
- `reaction_analysis.csv` - Reaction-specific results
- `intention_analysis.csv` - Intention-specific results
- `annotated_results_with_analysis.csv` - Dataset with computed metrics from notebook
- `analysis_summary.csv` - Summary statistics table

## Analysis Methodology

### Reaction Analysis

The system evaluates:
- **Semantic Similarity** - Cosine similarity between before/after annotations
- **Psychological Depth** - Level (powierzchowny/średni/głęboki) and focus (psychologiczny/mieszany/behawioralny)
- **Reaction Types** - Distribution across emotional, cognitive, behavioral, and external categories
- **Conceptual Coverage** - Categories maintained, added, or lost
- **Quality Change** - Score from -1 (decline) to +1 (improvement)

### Intention Analysis

The system evaluates:
- **Semantic Similarity** - Cosine similarity between before/after annotations
- **Conceptual Depth** - Level (powierzchowny/średni/głęboki) and focus (koncepcyjny/mieszany/konkretny)
- **Conceptual Coverage** - Categories maintained, added, or lost
- **Quality Change** - Score from -1 (decline) to +1 (improvement)

### Certainty Mapping

Polish text certainty levels are mapped to numeric scale:
- "Niska pewność" → 1 (Low)
- "Umiarkowana pewność" → 2 (Moderate)
- "Wysoka pewność" → 3 (High)
- "Całkowita pewność" → 4 (Complete)

## Customization

### Modify the Prompts

Edit the Jinja2 template files:
- `reaction_prompt.j2` - System instructions for reaction analysis
- `intention_prompt.j2` - System instructions for intention analysis
- `reaction_user_prompt.j2` - User message format for reactions
- `intention_user_prompt.j2` - User message format for intentions

### Modify the Output Schema

Edit the Pydantic models in `annotators_competences.py`:
- `ReactionAnalysisResult` - Schema for reaction analysis
- `IntentionAnalysisResult` - Schema for intention analysis
- Nested models like `PsychologicalDepthAnalysis`, `ConceptualDepthAnalysis`, etc.

### Change Processing Parameters

In `annotators_competences.py`, adjust the `AnnotatorCompetenceAnalyzer` initialization:

```python
analyzer = AnnotatorCompetenceAnalyzer(
    openai_api_key=openai_api_key,
    excel_path=excel_path,
    max_workers=2,      # Number of parallel workers
    request_delay=1.0,  # Delay between requests (seconds)
    max_retries=5       # Max retries on errors
)
```

### Change the Model

In `_rate_limited_api_call` method, change the model parameter:
- `gpt-4o-mini` - Current default (faster and cheaper)
- `gpt-4o` - Higher quality
- `gpt-4-turbo` - Alternative option

## Output Schema

### Flattened Column Structure

All nested results are flattened with underscore separators:

**Reaction columns:**
- `reaction_semantic_similarity` (float)
- `reaction_psychological_depth_analysis_before_level` (str)
- `reaction_psychological_depth_analysis_after_level` (str)
- `reaction_psychological_depth_analysis_before_focus` (str)
- `reaction_psychological_depth_analysis_after_focus` (str)
- `reaction_psychological_depth_analysis_change_direction` (str)
- `reaction_types_analysis_before_types_emocjonalne` (int)
- `reaction_types_analysis_before_types_kognitywne` (int)
- `reaction_types_analysis_before_types_behawioralne_manifestacje` (int)
- `reaction_types_analysis_before_types_zewnętrzne_wyniki` (int)
- `reaction_types_analysis_after_types_emocjonalne` (int)
- `reaction_types_analysis_after_types_kognitywne` (int)
- `reaction_types_analysis_after_types_behawioralne_manifestacje` (int)
- `reaction_types_analysis_after_types_zewnętrzne_wyniki` (int)
- `reaction_types_analysis_emotional_granularity_change` (str)
- `reaction_conceptual_coverage_before_categories` (str, semicolon-separated)
- `reaction_conceptual_coverage_after_categories` (str, semicolon-separated)
- `reaction_conceptual_coverage_maintained_categories` (str, semicolon-separated)
- `reaction_conceptual_coverage_new_categories` (str, semicolon-separated)
- `reaction_conceptual_coverage_lost_categories` (str, semicolon-separated)
- `reaction_evolution_assessment_type` (str)
- `reaction_evolution_assessment_description` (str)
- `reaction_evolution_assessment_quality_change` (float)
- `reaction_key_insights` (str)

**Intention columns:**
- `intention_semantic_similarity` (float)
- `intention_conceptual_depth_analysis_before_level` (str)
- `intention_conceptual_depth_analysis_after_level` (str)
- `intention_conceptual_depth_analysis_before_focus` (str)
- `intention_conceptual_depth_analysis_after_focus` (str)
- `intention_conceptual_depth_analysis_change_direction` (str)
- `intention_conceptual_coverage_before_categories` (str, semicolon-separated)
- `intention_conceptual_coverage_after_categories` (str, semicolon-separated)
- `intention_conceptual_coverage_maintained_categories` (str, semicolon-separated)
- `intention_conceptual_coverage_new_categories` (str, semicolon-separated)
- `intention_conceptual_coverage_lost_categories` (str, semicolon-separated)
- `intention_evolution_assessment_type` (str)
- `intention_evolution_assessment_description` (str)
- `intention_evolution_assessment_quality_change` (float)
- `intention_key_insights` (str)

## Troubleshooting

### Rate Limit Errors

If you encounter rate limit errors:
1. Increase `request_delay` parameter (e.g., 2.0 or 3.0 seconds)
2. Reduce `max_workers` to 1 for sequential processing
3. The script will automatically retry with exponential backoff

### Checkpoint Recovery

If the script fails mid-processing:
1. Simply run it again - it will automatically resume from the last checkpoint
2. To start fresh, delete the checkpoint files:
   ```bash
   rm checkpoint_reaction.json checkpoint_intention.json
   ```

### Notebook Errors

If you get KeyError for `certainty_change`:
1. Make sure to run Cell 1 first (data loading cell)
2. Or run "Run All" from the Jupyter menu
3. The certainty analysis cells now include safety checks to handle this automatically

### Data Type Issues

The notebook automatically:
- Converts Polish certainty text to numeric values (1-4)
- Converts all numeric columns using `pd.to_numeric()`
- Handles missing values with `.dropna()` calls

## License

This project is for research purposes.
