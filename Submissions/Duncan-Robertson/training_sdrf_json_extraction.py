from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
INPUT_DIR = REPO_ROOT / "data" / "TrainingSDRFs"
OUTPUT_DIR = SCRIPT_DIR / "data" / "Training-SDRF"
FILE_GLOB = "PXD*_cleaned.sdrf.tsv"
MISSING_VALUE = "Not Applicable"
PXD_RE = re.compile(r"(PXD\d{6})", re.IGNORECASE)

# Competition schema columns that matter for the extracted JSON.
COMPETITION_COLUMNS = {
    "PXD",
    "Raw Data File",
    "Characteristics[Age]",
    "Characteristics[AlkylationReagent]",
    "Characteristics[AnatomicSiteTumor]",
    "Characteristics[AncestryCategory]",
    "Characteristics[BMI]",
    "Characteristics[Bait]",
    "Characteristics[BiologicalReplicate]",
    "Characteristics[CellLine]",
    "Characteristics[CellPart]",
    "Characteristics[CellType]",
    "Characteristics[CleavageAgent]",
    "Characteristics[Compound]",
    "Characteristics[ConcentrationOfCompound]",
    "Characteristics[Depletion]",
    "Characteristics[DevelopmentalStage]",
    "Characteristics[DiseaseTreatment]",
    "Characteristics[Disease]",
    "Characteristics[GeneticModification]",
    "Characteristics[Genotype]",
    "Characteristics[GrowthRate]",
    "Characteristics[Label]",
    "Characteristics[MaterialType]",
    "Characteristics[Modification]",
    "Characteristics[NumberOfBiologicalReplicates]",
    "Characteristics[NumberOfSamples]",
    "Characteristics[NumberOfTechnicalReplicates]",
    "Characteristics[OrganismPart]",
    "Characteristics[Organism]",
    "Characteristics[OriginSiteDisease]",
    "Characteristics[PooledSample]",
    "Characteristics[ReductionReagent]",
    "Characteristics[SamplingTime]",
    "Characteristics[Sex]",
    "Characteristics[Specimen]",
    "Characteristics[SpikedCompound]",
    "Characteristics[Staining]",
    "Characteristics[Strain]",
    "Characteristics[SyntheticPeptide]",
    "Characteristics[Temperature]",
    "Characteristics[Time]",
    "Characteristics[Treatment]",
    "Characteristics[TumorCellularity]",
    "Characteristics[TumorGrade]",
    "Characteristics[TumorSite]",
    "Characteristics[TumorSize]",
    "Characteristics[TumorStage]",
    "Comment[AcquisitionMethod]",
    "Comment[CollisionEnergy]",
    "Comment[EnrichmentMethod]",
    "Comment[FlowRateChromatogram]",
    "Comment[FractionIdentifier]",
    "Comment[FractionationMethod]",
    "Comment[FragmentMassTolerance]",
    "Comment[FragmentationMethod]",
    "Comment[GradientTime]",
    "Comment[Instrument]",
    "Comment[IonizationType]",
    "Comment[MS2MassAnalyzer]",
    "Comment[NumberOfFractions]",
    "Comment[NumberOfMissedCleavages]",
    "Comment[PrecursorMassTolerance]",
    "Comment[Separation]",
    "FactorValue[Bait]",
    "FactorValue[CellPart]",
    "FactorValue[Compound]",
    "FactorValue[ConcentrationOfCompound].1",
    "FactorValue[Disease]",
    "FactorValue[FractionIdentifier]",
    "FactorValue[GeneticModification]",
    "FactorValue[Temperature]",
    "FactorValue[Treatment]",
}

# Direct raw-header -> competition-header mappings.
# Only map to actual competition columns; anything else should be dropped.
RAW_TO_CANONICAL: Dict[str, Optional[str]] = {
    # explicitly dropped
    "sourcename": None,
    "source name": None,
    "assayname": None,
    "assay name": None,
    "experiment": None,
    "technology type": None,
    "comment[proteomexchange accession number]": None,
    "characteristics[proteomexchange accession number]": None,
    "comment[file uri]": None,
    "comment[file url]": None,
    "comment[associated file uri]": None,
    "comment[tool metadata]": None,
    "comment[batch]": None,
    "comment[donor]": None,
    "comment[ptm]": None,
    "comment[scan window lower limit]": None,
    "comment[scan window upper limit]": None,
    "comment[isolation width]": None,
    "factor value[isolation width]": None,
    "characteristics[isolation width]": None,
    "characteristics[multiplicities of infection]": None,
    "characteristics[multiplicities of infection].1": None,
    "factor value[multiplicities of infection]": None,
    "characteristics[overproduction]": None,
    "characteristics[overproduction].1": None,
    "factor value[overproduction]": None,
    "factor value[overproduction].1": None,
    "characteristics[phenotype]": None,
    "characteristics[phenotype].1": None,
    "factor value[phenotype]": None,
    "factor value[phenotype].1": None,
    "characteristics[pool]": None,
    "factor value[pool]": None,
    "characteristics[protocol]": None,
    "factor value[protocol]": None,
    "characteristics[subtype]": None,
    "factor value[subtype]": None,
    "characteristics[hla]": None,
    "factor value[hla]": None,
    "characteristics[individual]": None,
    "characteristics[individual].1": None,
    "factor value[individual]": None,
    "characteristics[infect]": None,
    "characteristics[cultured cell]": None,
    "comment[rice ssp.]": None,
    "comment[rice strain]": None,
    "characteristics[rice strain]": None,
    "comment[modification parameters1]": None,

    # raw data / tracking
    "raw data file": "Raw Data File",
    "comment[data file]": "Raw Data File",
    "comment[data file].1": "Raw Data File",

    # direct plain-header mappings
    "age": "Characteristics[Age]",
    "alkylationreagent": "Characteristics[AlkylationReagent]",
    "anatomicsitetumor": "Characteristics[AnatomicSiteTumor]",
    "ancestrycategory": "Characteristics[AncestryCategory]",
    "bmi": "Characteristics[BMI]",
    "bait": "Characteristics[Bait]",
    "biologicalreplicate": "Characteristics[BiologicalReplicate]",
    "biological replicate": "Characteristics[BiologicalReplicate]",
    "cellline": "Characteristics[CellLine]",
    "cell line": "Characteristics[CellLine]",
    "cellpart": "Characteristics[CellPart]",
    "cell part": "Characteristics[CellPart]",
    "celltype": "Characteristics[CellType]",
    "cell type": "Characteristics[CellType]",
    "cleavageagent": "Characteristics[CleavageAgent]",
    "cleavage agent": "Characteristics[CleavageAgent]",
    "collisionenergy": "Comment[CollisionEnergy]",
    "compound": "Characteristics[Compound]",
    "concentrationofcompound": "Characteristics[ConcentrationOfCompound]",
    "depletion": "Characteristics[Depletion]",
    "developmentalstage": "Characteristics[DevelopmentalStage]",
    "disease": "Characteristics[Disease]",
    "diseasetreatment": "Characteristics[DiseaseTreatment]",
    "disease treatment": "Characteristics[DiseaseTreatment]",
    "enrichmentmethod": "Comment[EnrichmentMethod]",
    "fractionidentifier": "Comment[FractionIdentifier]",
    "fractionationmethod": "Comment[FractionationMethod]",
    "fragmentmasstolerance": "Comment[FragmentMassTolerance]",
    "fragmentationmethod": "Comment[FragmentationMethod]",
    "growthrate": "Characteristics[GrowthRate]",
    "instrument": "Comment[Instrument]",
    "label": "Characteristics[Label]",
    "ms2massanalyzer": "Comment[MS2MassAnalyzer]",
    "materialtype": "Characteristics[MaterialType]",
    "material type": "Characteristics[MaterialType]",
    "modification": "Characteristics[Modification]",
    "numberofmissedcleavages": "Comment[NumberOfMissedCleavages]",
    "organism": "Characteristics[Organism]",
    "organismpart": "Characteristics[OrganismPart]",
    "organism part": "Characteristics[OrganismPart]",
    "pooledsample": "Characteristics[PooledSample]",
    "precursormasstolerance": "Comment[PrecursorMassTolerance]",
    "reductionreagent": "Characteristics[ReductionReagent]",
    "samplingtime": "Characteristics[SamplingTime]",
    "separation": "Comment[Separation]",
    "sex": "Characteristics[Sex]",
    "specimen": "Characteristics[Specimen]",
    "spikedcompound": "Characteristics[SpikedCompound]",
    "staining": "Characteristics[Staining]",
    "strain": "Characteristics[Strain]",
    "syntheticpeptide": "Characteristics[SyntheticPeptide]",
    "technicalreplicate": "Characteristics[NumberOfTechnicalReplicates]",
    "technical replicate": "Characteristics[NumberOfTechnicalReplicates]",
    "temperature": "Characteristics[Temperature]",
    "time": "Characteristics[Time]",
    "treatment": "Characteristics[Treatment]",
    "tumorcellularity": "Characteristics[TumorCellularity]",
    "tumorgrade": "Characteristics[TumorGrade]",
    "tumorsite": "Characteristics[TumorSite]",
    "tumorsize": "Characteristics[TumorSize]",
    "tumorstage": "Characteristics[TumorStage]",

    # comment aliases
    "comment[technical replicate]": "Characteristics[NumberOfTechnicalReplicates]",
    "comment[plasmodium strain]": "Characteristics[Strain]",
    "comment[fraction identifier]": "Comment[FractionIdentifier]",
    "comment[fragmentation method]": "Comment[FragmentationMethod]",
    "comment[fragment mass tolerance]": "Comment[FragmentMassTolerance]",
    "comment[precursor mass tolerance]": "Comment[PrecursorMassTolerance]",
    "comment[instrument]": "Comment[Instrument]",
    "comment[ionization type]": "Comment[IonizationType]",
    "comment[ms2 mass analyzer]": "Comment[MS2MassAnalyzer]",
    "comment[cleavage agent details]": "Characteristics[CleavageAgent]",
    "comment[cid collision energy]": "Comment[CollisionEnergy]",
    "comment[etcid collision energy]": "Comment[CollisionEnergy]",
    "comment[ethcd collision energy]": "Comment[CollisionEnergy]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomexchange accession number]": None,
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomexchange accession number]": None,
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomexchange accession number]": None,
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[proteomexchange accession number]": None,
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",

    # lowercase bracketed characteristics -> competition columns
    "characteristics[age]": "Characteristics[Age]",
    "characteristics[age at death]": "Characteristics[Age]",
    "characteristics[alkylationreagent]": "Characteristics[AlkylationReagent]",
    "characteristics[anatomicsitetumor]": "Characteristics[AnatomicSiteTumor]",
    "characteristics[ancestrycategory]": "Characteristics[AncestryCategory]",
    "characteristics[bmi]": "Characteristics[BMI]",
    "characteristics[bait]": "Characteristics[Bait]",
    "characteristics[biologicalreplicate]": "Characteristics[BiologicalReplicate]",
    "characteristics[cellline]": "Characteristics[CellLine]",
    "characteristics[cellpart]": "Characteristics[CellPart]",
    "characteristics[celltype]": "Characteristics[CellType]",
    "characteristics[cleavageagent]": "Characteristics[CleavageAgent]",
    "characteristics[compound]": "Characteristics[Compound]",
    "characteristics[concentrationofcompound]": "Characteristics[ConcentrationOfCompound]",
    "characteristics[depletion]": "Characteristics[Depletion]",
    "characteristics[developmentalstage]": "Characteristics[DevelopmentalStage]",
    "characteristics[disease]": "Characteristics[Disease]",
    "characteristics[diseasetreatment]": "Characteristics[DiseaseTreatment]",
    "characteristics[geneticmodification]": "Characteristics[GeneticModification]",
    "characteristics[genotype]": "Characteristics[Genotype]",
    "characteristics[growthrate]": "Characteristics[GrowthRate]",
    "characteristics[label]": "Characteristics[Label]",
    "characteristics[materialtype]": "Characteristics[MaterialType]",
    "characteristics[modification]": "Characteristics[Modification]",
    "characteristics[numberofbiologicalreplicates]": "Characteristics[NumberOfBiologicalReplicates]",
    "characteristics[numberofsamples]": "Characteristics[NumberOfSamples]",
    "characteristics[numberoftechnicalreplicates]": "Characteristics[NumberOfTechnicalReplicates]",
    "characteristics[organism]": "Characteristics[Organism]",
    "characteristics[organismpart]": "Characteristics[OrganismPart]",
    "characteristics[originsitedisease]": "Characteristics[OriginSiteDisease]",
    "characteristics[pooledsample]": "Characteristics[PooledSample]",
    "characteristics[reductionreagent]": "Characteristics[ReductionReagent]",
    "characteristics[samplingtime]": "Characteristics[SamplingTime]",
    "characteristics[sex]": "Characteristics[Sex]",
    "characteristics[specimen]": "Characteristics[Specimen]",
    "characteristics[spikedcompound]": "Characteristics[SpikedCompound]",
    "characteristics[staining]": "Characteristics[Staining]",
    "characteristics[strain]": "Characteristics[Strain]",
    "characteristics[syntheticpeptide]": "Characteristics[SyntheticPeptide]",
    "characteristics[temperature]": "Characteristics[Temperature]",
    "characteristics[time]": "Characteristics[Time]",
    "characteristics[treatment]": "Characteristics[Treatment]",
    "characteristics[tumorcellularity]": "Characteristics[TumorCellularity]",
    "characteristics[tumorgrade]": "Characteristics[TumorGrade]",
    "characteristics[tumorsite]": "Characteristics[TumorSite]",
    "characteristics[tumorsize]": "Characteristics[TumorSize]",
    "characteristics[tumorstage]": "Characteristics[TumorStage]",
    "characteristics[plasmodium strain]": "Characteristics[Strain]",
    "characteristics[chemical entity]": "Characteristics[Compound]",
    "characteristics[disease response]": "Characteristics[Disease]",
    "characteristics[induced by]": "Characteristics[Treatment]",

    # lowercase factor value -> competition factor columns
    "factor value[bait]": "FactorValue[Bait]",
    "factor value[cellpart]": "FactorValue[CellPart]",
    "factor value[compound]": "FactorValue[Compound]",
    "factor value[concentrationofcompound].1": "FactorValue[ConcentrationOfCompound].1",
    "factor value[concentrationofcompound]": "FactorValue[ConcentrationOfCompound].1",
    "factor value[disease]": "FactorValue[Disease]",
    "factor value[disease response]": "FactorValue[Disease]",
    "factor value[fractionidentifier]": "FactorValue[FractionIdentifier]",
    "factor value[geneticmodification]": "FactorValue[GeneticModification]",
    "factor value[temperature]": "FactorValue[Temperature]",
    "factor value[treatment]": "FactorValue[Treatment]",
    "factor value[chemical entity]": "FactorValue[Compound]",
    "factor value[induced by]": "FactorValue[Treatment]",

    # standardized factor aliases that may show up
    "factorvalue[bait]": "FactorValue[Bait]",
    "factorvalue[cellpart]": "FactorValue[CellPart]",
    "factorvalue[compound]": "FactorValue[Compound]",
    "factorvalue[concentrationofcompound].1": "FactorValue[ConcentrationOfCompound].1",
    "factorvalue[disease]": "FactorValue[Disease]",
    "factorvalue[fractionidentifier]": "FactorValue[FractionIdentifier]",
    "factorvalue[geneticmodification]": "FactorValue[GeneticModification]",
    "factorvalue[temperature]": "FactorValue[Temperature]",
    "factorvalue[treatment]": "FactorValue[Treatment]",
}

# Preferred output ordering in each JSON object.
OUTPUT_ORDER = [
    "PXD",
    "Raw Data File",
    "Characteristics[Age]",
    "Characteristics[AlkylationReagent]",
    "Characteristics[AnatomicSiteTumor]",
    "Characteristics[AncestryCategory]",
    "Characteristics[BMI]",
    "Characteristics[Bait]",
    "Characteristics[BiologicalReplicate]",
    "Characteristics[CellLine]",
    "Characteristics[CellPart]",
    "Characteristics[CellType]",
    "Characteristics[CleavageAgent]",
    "Characteristics[Compound]",
    "Characteristics[ConcentrationOfCompound]",
    "Characteristics[Depletion]",
    "Characteristics[DevelopmentalStage]",
    "Characteristics[DiseaseTreatment]",
    "Characteristics[Disease]",
    "Characteristics[GeneticModification]",
    "Characteristics[Genotype]",
    "Characteristics[GrowthRate]",
    "Characteristics[Label]",
    "Characteristics[MaterialType]",
    "Characteristics[Modification]",
    "Characteristics[NumberOfBiologicalReplicates]",
    "Characteristics[NumberOfSamples]",
    "Characteristics[NumberOfTechnicalReplicates]",
    "Characteristics[OrganismPart]",
    "Characteristics[Organism]",
    "Characteristics[OriginSiteDisease]",
    "Characteristics[PooledSample]",
    "Characteristics[ReductionReagent]",
    "Characteristics[SamplingTime]",
    "Characteristics[Sex]",
    "Characteristics[Specimen]",
    "Characteristics[SpikedCompound]",
    "Characteristics[Staining]",
    "Characteristics[Strain]",
    "Characteristics[SyntheticPeptide]",
    "Characteristics[Temperature]",
    "Characteristics[Time]",
    "Characteristics[Treatment]",
    "Characteristics[TumorCellularity]",
    "Characteristics[TumorGrade]",
    "Characteristics[TumorSite]",
    "Characteristics[TumorSize]",
    "Characteristics[TumorStage]",
    "FactorValue[Bait]",
    "FactorValue[CellPart]",
    "FactorValue[Compound]",
    "FactorValue[ConcentrationOfCompound].1",
    "FactorValue[Disease]",
    "FactorValue[FractionIdentifier]",
    "FactorValue[GeneticModification]",
    "FactorValue[Temperature]",
    "FactorValue[Treatment]",
    "Comment[AcquisitionMethod]",
    "Comment[CollisionEnergy]",
    "Comment[EnrichmentMethod]",
    "Comment[FlowRateChromatogram]",
    "Comment[FractionIdentifier]",
    "Comment[FractionationMethod]",
    "Comment[FragmentMassTolerance]",
    "Comment[FragmentationMethod]",
    "Comment[GradientTime]",
    "Comment[Instrument]",
    "Comment[IonizationType]",
    "Comment[MS2MassAnalyzer]",
    "Comment[NumberOfFractions]",
    "Comment[NumberOfMissedCleavages]",
    "Comment[PrecursorMassTolerance]",
    "Comment[Separation]",
]

def normalize(value: str | None) -> str:
    if value is None:
        return MISSING_VALUE
    text = str(value).strip()
    if not text:
        return MISSING_VALUE
    lowered = text.lower()
    if lowered in {"na", "n/a", "nan", "none", "null", "not applicable", "not available"}:
        return MISSING_VALUE
    return text


def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        norm = normalize(value)
        if norm == MISSING_VALUE or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _normalize_header_key(header: str) -> str:
    """
    Normalize arbitrary input headers so lower/upper case, repeated spaces,
    factor value vs FactorValue, and duplicate suffixes are handled consistently.
    """
    raw = header.strip().replace("\ufeff", "")
    raw = re.sub(r"\.(\d+)$", "", raw)  # remove duplicate suffixes for lookup
    raw = raw.replace("Factor Value[", "FactorValue[").replace("factor value[", "FactorValue[")
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"\[\s*", "[", raw)
    raw = re.sub(r"\s*\]", "]", raw)
    return raw.strip()


def canonicalize_header(header: str) -> Optional[str]:
    raw = _normalize_header_key(header)
    if not raw:
        return None

    lowered = raw.lower()
    if lowered in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[lowered]

    compact = re.sub(r"[^a-z0-9\[\]\.]+", "", lowered)
    if compact in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[compact]

    # Handle already-standardized or semi-standardized bracketed forms, case-insensitively.
    m = re.match(r"^(characteristics|comment|factorvalue)\[(.+?)\]$", raw, flags=re.IGNORECASE)
    if not m:
        return None

    kind = m.group(1).lower()
    category = m.group(2).strip()
    category_lower = category.lower()

    # Route common lowercase/raw bracketed names through RAW_TO_CANONICAL first.
    fallback_key = f"{kind}[{category_lower}]"
    if fallback_key in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[fallback_key]

    # Already competition-conformant names.
    if kind == "characteristics":
        candidate = f"Characteristics[{category}]"
        return candidate if candidate in COMPETITION_COLUMNS else None

    if kind == "comment":
        candidate = f"Comment[{category}]"
        return candidate if candidate in COMPETITION_COLUMNS else None

    if kind == "factorvalue":
        # Preserve only competition factor columns.
        if category == "ConcentrationOfCompound.1":
            candidate = "FactorValue[ConcentrationOfCompound].1"
        else:
            candidate = f"FactorValue[{category}]"
        return candidate if candidate in COMPETITION_COLUMNS else None

    return None


def parse_tsv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty TSV file: {path}")

    headers = [h.strip() for h in rows[0]]
    width = len(headers)

    data_rows: List[List[str]] = []
    for row in rows[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        data_rows.append(row)

    return headers, data_rows


def extract_pxd_from_path(path: Path) -> str:
    match = PXD_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not find a PXD accession in filename: {path.name}")
    return match.group(1).upper()


def row_to_metadata(headers: List[str], row: List[str], pxd: str) -> Dict[str, object]:
    grouped: Dict[str, List[str]] = defaultdict(list)

    for header, raw_value in zip(headers, row):
        canonical = canonicalize_header(header)
        if canonical is None:
            continue

        value = normalize(raw_value)
        if value == MISSING_VALUE:
            continue

        grouped[canonical].append(value)

    grouped["PXD"].append(pxd)

    result: Dict[str, object] = {}
    for key in OUTPUT_ORDER:
        if key not in grouped:
            continue
        values = dedupe_preserve_order(grouped[key])
        if values:
            result[key] = values

    return result


def convert_training_sdrf_file(tsv_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    pxd = extract_pxd_from_path(tsv_path)
    headers, rows = parse_tsv(tsv_path)

    metadata_dict: Dict[str, object] = {}
    fallback_counter = 0

    for row in rows:
        meta = row_to_metadata(headers, row, pxd)
        raw_files = meta.get("Raw Data File")

        if isinstance(raw_files, list) and raw_files:
            # Use the first raw file as the object key; JSON body still stores it as a list.
            key = str(raw_files[0])
        else:
            key = f"row_{fallback_counter}"
            fallback_counter += 1

        metadata_dict[key] = meta

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pxd}_metadata.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_dict, handle, indent=2, ensure_ascii=False)

    return output_path


def convert_all_training_sdrfs(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> List[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths = sorted(input_dir.glob(FILE_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matching {FILE_GLOB!r} were found in {input_dir}")

    outputs: List[Path] = []
    for tsv_path in paths:
        outputs.append(convert_training_sdrf_file(tsv_path, output_dir=output_dir))
    return outputs


def main() -> None:
    outputs = convert_all_training_sdrfs()
    print(f"Converted {len(outputs)} training SDRF files to JSON in: {OUTPUT_DIR}")
    for path in outputs[:10]:
        print(path)
    if len(outputs) > 10:
        print("...")


if __name__ == "__main__":
    main()