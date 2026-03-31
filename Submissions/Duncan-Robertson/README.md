# One-Shot Hybrid Pipeline

Hybrid SDRF metadata extraction pipeline for the **Harmonizing the Data of Your Data** Kaggle competition. The approach combines training-set priors, external repository metadata, rule-based extraction, prompt-based Gemini extraction, ontology/value normalization, and filename-aware row assignment to produce competition-format SDRF predictions.

This implementation was adapted from the Kaggle notebook by **ashsihkumar** and then modified into a local script/config workflow with a different prompt, path structure, normalization layer, and post-processing logic.

## Method

### Overview
The pipeline is a hybrid metadata extraction system built around six main stages:

1. **Training-data priors**
   - Load the harmonized training SDRFs.
   - Build per-column vocabularies and frequency counters.
   - Compute global fallback values for columns that are populated in most training files.
   - Reuse exact per-PXD SDRF values when a training PXD overlap exists.

2. **External metadata retrieval**
   - Query PRIDE metadata.
   - Query ProteomeXchange XML metadata.
   - Convert retrieved metadata into SDRF-style candidate values.

3. **Rule-based extraction from publication text**
   - Use curated normalization dictionaries for organism, organism part/tissue, and instrument mapping.
   - Apply regex-based extraction for structured metadata such as organism, tissue, cell line, cell type, digestion enzyme, labeling strategy, instrument, mass tolerances, gradient time, fractions, and other experimental settings.
   - Use conservative gating for noisier fields such as disease.

4. **LLM-based extraction**
   - Read a prompt from the configured prompt file.
   - Build a focused publication text context from title, abstract, methods, and method-related sections.
   - Call Gemini with JSON output enforced at generation time.
   - Normalize the returned JSON so that all fields are flat SDRF keys with list-valued outputs.

5. **Normalization and candidate merging**
   - Merge candidates in priority order:
     1. training-PXD overlap
     2. PRIDE metadata
     3. ProteomeXchange XML metadata
     4. Gemini extraction
     5. regex extraction
     6. global fallback values
   - Snap extracted values toward training-set vocabularies using RapidFuzz token-based matching.
   - Cap noisy multi-value fields to reduce over-generation.
   - Expand modification values into `Characteristics[Modification]`, `Characteristics[Modification].1`, etc.

6. **Submission assembly**
   - Parse raw data filenames for file-level metadata such as fraction identifiers, biological replicates, technical replicates, and labels.
   - Combine PXD-level candidate sets with filename-derived row-level assignments.
   - Fill the final sample submission schema and write the competition submission file.

### Data preprocessing
- The script loads paths from `config.yaml` and environment variables from `.env`.
- Training SDRFs are scanned to build vocabulary and frequency statistics for each target SDRF column.
- Test publication JSON files are loaded from the configured PubText directory.
- Only the SDRF target columns in the sample submission schema are predicted.

### Feature extraction / engineering
- **Training priors:** per-column vocabularies, global modes, and per-PXD observed SDRF values.
- **Repository metadata:** PRIDE and ProteomeXchange dataset metadata.
- **Publication text features:** method-focused text slices assembled from title/abstract/method sections.
- **Rule-based signals:** curated regex patterns and normalization dictionaries for proteomics-specific metadata.
- **Filename features:** fraction, replicate, and labeling cues parsed directly from raw file names.

### Model architecture or rules
This is not a single statistical model. It is a **hybrid extraction pipeline** with:
- deterministic metadata lookup from training SDRFs and repository APIs,
- regex and dictionary-based extraction for structured fields,
- prompt-based LLM extraction for semantically harder fields,
- vocabulary snapping and post-processing to harmonize output values.

### Post-processing
- Deduplicate candidates within each PXD.
- Apply RapidFuzz-based value snapping against training vocabularies.
- Limit noisy multi-valued fields such as organism part, cell line, cell type, disease, and modifications.
- Split modification predictions across suffixed SDRF columns.
- Use filename-derived values when row-level differentiation is available.
- Default remaining empty outputs to `Not Applicable` when writing the final submission table.

### Key hyperparameters
- `PRIDE_TIMEOUT = 12`
- `PX_TIMEOUT = 12`
- `OLS_TIMEOUT = 8`
- `FUZZY_CUTOFF = 82`
- `MULTI_VAL_CAP = [FILL IN]`
- `gemini_model = gemini-2.0-flash`
- `gemini_temperature = 0.0`

## Prompting strategy
The LLM stage uses a long-form instruction prompt tailored to SDRF extraction. The prompt requires:
- flat SDRF keys such as `Characteristics[...]`, `Comment[...]`, and `FactorValue[...]`,
- omission of fields that are not explicitly supported by the paper,
- list-valued JSON outputs only,
- conservative distinction between sample descriptors and true experimental factors,
- field-by-field extraction guidance to reduce hallucination and schema drift.

## Results
- Local F1 Score: N/A
- Kaggle Score: **0.34937**
- Key findings:
  - Hybrid extraction worked better than relying on only regex or only LLM output.
  - Repository metadata and training-set priors provided strong anchors for many columns.
  - Prompt-constrained JSON extraction improved recall on fields that are difficult to recover with rules alone.
  - Filename-aware assignment helped map PXD-level metadata back to per-file SDRF rows.

## Installation & Usage

### 1. Install dependencies
```bash
pip install -r Submissions/Duncan-Robertson/requirements.txt
```

### 2. Configure files
Place the following in `Submissions/Duncan-Robertson/`:
- `config.yaml`
- `.env`
- your prompt file referenced by `prompt1_file` in `config.yaml`

Your `.env` file should contain:
```bash
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the pipeline
```bash
python Submissions/Duncan-Robertson/pipeline.py
```

### 4. Output
The pipeline writes:
- intermediate extracted metadata JSON files to the configured extract directory,
- the final submission CSV to the configured submission directory.

## Repository structure
```text
Submissions/
  Duncan-Robertson/
    data/
      prompt.txt
      training_extract.json
    pipeline.py
    config.yaml
    .env
    README.md
    requirements.txt
```

## Acknowledgements / References
- Kaggle competition: **Harmonizing the Data of Your Data**
- Source notebook inspiration: **ashsihkumar — harmonizing-the-data-of-your-data**
- The final system in this folder is a modified local-script implementation rather than a direct copy of the notebook.
- The LLM extraction behavior is driven by a custom SDRF-focused prompt designed for conservative JSON-only metadata extraction.
