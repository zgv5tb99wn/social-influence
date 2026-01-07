# Annotators Competence Analysis

This script fetches data from Google Sheets, processes it using OpenAI's structured outputs with a Jinja2 template, and saves the annotated results.

## Setup

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Set up Google Sheets API

To access Google Sheets, you need to set up a service account:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Sheets API
4. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name and description
   - Click "Create and Continue"
   - Skip the optional steps
5. Create a key for the service account:
   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the key file
6. Save the key file as `credentials.json` in `~/.config/gspread/service_account.json` or in the same directory as the script
7. Share your Google Sheet with the service account email (found in the JSON file)

### 3. Set up OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the script:

```bash
python social_influence/experiments/annotators_competences/annotators_competences.py
```

## Files

- `annotators_competences.py` - Main script
- `annotation_prompt.j2` - Jinja2 template for the annotation prompt
- `annotated_results.csv` - Output file with results (generated after running)

## Customization

### Modify the Annotation Prompt

Edit `annotation_prompt.j2` to change the prompt sent to OpenAI.

### Modify the Output Schema

Edit the `AnnotationResult` class in `annotators_competences.py` to change the structured output fields.

### Change the Model

In the `annotate_single_row` method, change the model parameter:
- `gpt-4o-mini` - Faster and cheaper
- `gpt-4o` - Higher quality
- `gpt-4-turbo` - Alternative option

## Output

The script generates a CSV file with:
- All original columns from the Google Sheet
- Additional columns from the OpenAI annotation:
  - `consistency_score` - Consistency across responses (0-1)
  - `clarity_rating` - Overall clarity (1-5)
  - `intention_alignment` - Whether intention is clear (boolean)
  - `reaction_appropriateness` - Assessment of reaction
  - `certainty_level` - High/Medium/Low
  - `overall_notes` - Additional observations
