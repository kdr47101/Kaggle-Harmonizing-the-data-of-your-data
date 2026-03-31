import os, re, gc, json, time
from rapidfuzz import fuzz, process as rfprocess
from collections import defaultdict, Counter
from pathlib import Path

import requests
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# 0. CONFIG
# ────────────────────────────────────────────────────────────
PRIDE_TIMEOUT = 12
PX_TIMEOUT    = 12
OLS_TIMEOUT   = 8

FUZZY_CUTOFF = 82
MULTI_VAL_CAP = 7

CAPPED_MULTI_VALUE_COLS = {
    "Characteristics[OrganismPart]",
    "Characteristics[CellLine]",
    "Characteristics[CellType]",
    "Characteristics[Modification]",
    "Characteristics[Disease]",
}

# ────────────────────────────────────────────────────────────
# 1. PATHS
# ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

load_dotenv(HERE / ".env")

TEST_PUBTEXT_DIR = PROJECT_ROOT / "data" / "TestPubText"
TEST_EXTRACT_DIR = HERE / "data" / "Gemini-Extract"
TRAINING_EXTRACT_DIR = HERE / "data" / "Training-SDRF"
SUBMISSION_DIR = HERE
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "data" / "SampleSubmission.csv"
PROMPT_PATH = HERE / "data" / "prompt.txt"
TRAINING_VALUES_PATH = HERE / "data" / "training_values.json"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
USE_GEMINI = bool(GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_TEMPERATURE = 0.0
BASE_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

TEST_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
ID_COLS = ["ID", "PXD", "Raw Data File", "Usage"]
target_cols = [c for c in sample_sub.columns if c not in ID_COLS and "Unnamed" not in c]
base_tgt = sorted(set(re.sub(r"\.\d+$", "", c) for c in target_cols))
all_base = set(base_tgt)

# ────────────────────────────────────────────────────────────
# 2. VOCABULARY & TRAINING DATA
# ────────────────────────────────────────────────────────────

if not TRAINING_VALUES_PATH.exists():
    raise FileNotFoundError(
        f"Missing {TRAINING_VALUES_PATH}. "
        f"Run: python {HERE / 'build_training_values.py'}"
    )

with open(TRAINING_VALUES_PATH, "r", encoding="utf-8") as f:
    training_values = json.load(f)

col_vocab = defaultdict(
    set,
    {k: set(v) for k, v in training_values["col_vocab"].items()}
)
global_modes = training_values["global_modes"]
non_na_ratio = {k: float(v) for k, v in training_values["non_na_ratio"].items()}
train_pxd_sdrf = training_values["train_pxd_sdrf"]

# ────────────────────────────────────────────────────────────
# 3. NORMALISATION DICTIONARIES
# ────────────────────────────────────────────────────────────
organism_norm = {
    "human": "9606 (Homo sapiens)", "homo sapiens": "9606 (Homo sapiens)",
    "mouse": "10090 (Mus musculus)", "mice": "10090 (Mus musculus)",
    "murine": "10090 (Mus musculus)", "mus musculus": "10090 (Mus musculus)",
    "rat": "10116 (Rattus norvegicus)", "rats": "10116 (Rattus norvegicus)",
    "rattus norvegicus": "10116 (Rattus norvegicus)",
    "yeast": "4932 (Saccharomyces cerevisiae)",
    "saccharomyces cerevisiae": "4932 (Saccharomyces cerevisiae)",
    "e. coli": "562 (Escherichia coli)", "e.coli": "562 (Escherichia coli)",
    "escherichia coli": "562 (Escherichia coli)",
    "drosophila melanogaster": "7227 (Drosophila melanogaster)",
    "fruit fly": "7227 (Drosophila melanogaster)",
    "zebrafish": "7955 (Danio rerio)", "danio rerio": "7955 (Danio rerio)",
    "arabidopsis thaliana": "3702 (Arabidopsis thaliana)",
    "pig": "9823 (Sus scrofa)", "porcine": "9823 (Sus scrofa)", "sus scrofa": "9823 (Sus scrofa)",
    "bovine": "9913 (Bos taurus)", "cow": "9913 (Bos taurus)", "bos taurus": "9913 (Bos taurus)",
    "chicken": "9031 (Gallus gallus)", "gallus gallus": "9031 (Gallus gallus)",
    "rabbit": "9986 (Oryctolagus cuniculus)",
    "c. elegans": "6239 (Caenorhabditis elegans)",
    "caenorhabditis elegans": "6239 (Caenorhabditis elegans)",
    "xenopus laevis": "8355 (Xenopus laevis)",
    "dog": "9615 (Canis lupus familiaris)",
}

tissue_norm = {
    "brain": "NT=brain;AC=UBERON:0000955",
    "liver": "NT=liver;AC=UBERON:0002107",
    "lung": "NT=lung;AC=UBERON:0002048",
    "heart": "NT=heart;AC=UBERON:0000948",
    "kidney": "NT=kidney;AC=UBERON:0002113",
    "muscle": "NT=skeletal muscle;AC=UBERON:0001134",
    "colon": "NT=colon;AC=UBERON:0001155",
    "breast": "NT=breast;AC=UBERON:0000310",
    "prostate": "NT=prostate gland;AC=UBERON:0002367",
    "pancreas": "NT=pancreas;AC=UBERON:0001264",
    "ovary": "NT=ovary;AC=UBERON:0000992",
    "ovaries": "NT=ovary;AC=UBERON:0000992",
    "skin": "NT=skin of body;AC=UBERON:0002097",
    "bone marrow": "NT=bone marrow;AC=UBERON:0002371",
    "spleen": "NT=spleen;AC=UBERON:0002106",
    "plasma": "NT=blood plasma;AC=UBERON:0001969",
    "serum": "NT=blood serum;AC=UBERON:0001977",
    "blood": "NT=blood;AC=UBERON:0000178",
    "urine": "NT=urine;AC=UBERON:0001088",
    "cerebrospinal fluid": "NT=cerebrospinal fluid;AC=UBERON:0001359",
    "csf": "NT=cerebrospinal fluid;AC=UBERON:0001359",
    "saliva": "NT=saliva;AC=UBERON:0001836",
    "thymus": "NT=thymus;AC=UBERON:0002370",
    "lymph node": "NT=lymph node;AC=UBERON:0000029",
    "adipose": "NT=adipose tissue;AC=UBERON:0001013",
    "testis": "NT=testis;AC=UBERON:0000473",
    "testes": "NT=testis;AC=UBERON:0000473",
    "stomach": "NT=stomach;AC=UBERON:0000945",
    "intestine": "NT=intestine;AC=UBERON:0000160",
    "small intestine": "NT=small intestine;AC=UBERON:0002108",
    "large intestine": "NT=large intestine;AC=UBERON:0000059",
    "thyroid": "NT=thyroid gland;AC=UBERON:0002046",
    "retina": "NT=retina;AC=UBERON:0000966",
    "hippocampus": "NT=hippocampal formation;AC=UBERON:0002421",
    "cortex": "NT=cerebral cortex;AC=UBERON:0000956",
    "cerebellum": "NT=cerebellum;AC=UBERON:0002037",
    "frontal lobe": "NT=frontal lobe;AC=UBERON:0001870",
    "temporal lobe": "NT=temporal lobe;AC=UBERON:0001871",
    "striatum": "NT=striatum;AC=UBERON:0002435",
    "substantia nigra": "NT=substantia nigra;AC=UBERON:0002038",
    "thalamus": "NT=thalamus;AC=UBERON:0001897",
    "cervix": "NT=uterine cervix;AC=UBERON:0000002",
    "endometrium": "NT=endometrium;AC=UBERON:0001295",
    "placenta": "NT=placenta;AC=UBERON:0001987",
    "umbilical cord": "NT=umbilical cord;AC=UBERON:0002331",
    "peripheral blood": "NT=peripheral blood;AC=UBERON:0000178",
    "pbmc": "NT=peripheral blood mononuclear cell;AC=CL:0000057",
    "platelet": "NT=platelet;AC=CL:0000233",
    "prostate gland": "NT=prostate gland;AC=UBERON:0002367",
    "blood plasma": "NT=blood plasma;AC=UBERON:0001969",
    "blood serum": "NT=blood serum;AC=UBERON:0001977",
    "adipose tissue": "NT=adipose tissue;AC=UBERON:0001013",
    "thyroid gland": "NT=thyroid gland;AC=UBERON:0002046",
}

instrument_norm = {
    "q exactive hf-x": "AC=MS:1003027;NT=Q Exactive HF-X",
    "q exactive hf":   "AC=MS:1002523;NT=Q Exactive HF",
    "q exactive plus":  "AC=MS:1002634;NT=Q Exactive Plus",
    "q-exactive plus":  "AC=MS:1002634;NT=Q Exactive Plus",
    "q exactive":       "AC=MS:1001911;NT=Q Exactive",
    "q-exactive":       "AC=MS:1001911;NT=Q Exactive",
    "orbitrap fusion lumos": "AC=MS:1002732;NT=Orbitrap Fusion Lumos",
    "orbitrap fusion":  "AC=MS:1002416;NT=Orbitrap Fusion",
    "orbitrap eclipse": "AC=MS:1003029;NT=Orbitrap Eclipse",
    "exploris 480":     "AC=MS:1003094;NT=Orbitrap Exploris 480",
    "orbitrap exploris":"AC=MS:1003094;NT=Orbitrap Exploris 480",
    "ltq orbitrap velos":"AC=MS:1001742;NT=LTQ Orbitrap Velos",
    "ltq orbitrap elite":"AC=MS:1001910;NT=LTQ Orbitrap Elite",
    "ltq orbitrap xl":  "AC=MS:1000556;NT=LTQ Orbitrap XL",
    "ltq orbitrap":     "AC=MS:1000449;NT=LTQ Orbitrap",
    "timstof pro":      "AC=MS:1003231;NT=timsTOF Pro",
    "timstof":          "AC=MS:1002817;NT=timsTOF",
    "impact ii":        "AC=MS:1002817;NT=impact II",
    "maxi speed":       "AC=MS:1002817;NT=maXis Speed",
    "synapt g2":        "AC=MS:1002726;NT=Synapt G2-Si",
    "triple tof 6600":  "AC=MS:1000931;NT=TripleTOF 6600",
    "triple tof 5600":  "AC=MS:1000931;NT=TripleTOF 5600",
    "triple tof":       "AC=MS:1000931;NT=TripleTOF 6600",
    "sciex 6600":       "AC=MS:1000931;NT=TripleTOF 6600",
    "velos pro":        "AC=MS:1001909;NT=LTQ Velos Pro",
    "eclipse":          "AC=MS:1003029;NT=Orbitrap Eclipse",
    "astral":           "AC=MS:1003378;NT=Orbitrap Astral",
    "exploris 240":     "AC=MS:1003028;NT=Orbitrap Exploris 240",
    "timstof ht":       "AC=MS:1003405;NT=timsTOF HT",
    "zeno tof 7600":    "AC=MS:1003027;NT=Zeno TOF 7600",
    "synapt":       "AC=MS:1002726;NT=Synapt G2-Si",
}

CELL_LINES = [
    "HEK293", "HEK-293", "HEK293T", "HEK 293", "HeLa", "U2OS", "MCF7", "MCF-7",
    "A549", "Jurkat", "K562", "HCT116", "HepG2", "CHO", "PC3", "LNCaP", "THP-1",
    "SH-SY5Y", "Caco-2", "NIH3T3", "RAW264.7", "RAW 264.7", "U87", "U251", "T47D",
    "MDA-MB-231", "MDA-MB-468", "PANC-1", "MiaPaCa", "AsPC-1", "OVCAR", "SKOV3",
    "HL-60", "Ramos", "Karpas-299", "NCI-H1299", "NCI-H460", "SW480", "SW620",
    "LoVo", "HT-29", "BV2", "Vero", "HUVEC", "B16", "C2C12", "3T3-L1", "U937",
    "RPMI 8226", "MEF", "iPSC", "DLD-1", "RKO", "HepG2", "Huh7", "SNU-398",
    "PC-9", "H1299", "H460", "H1975", "A375", "SK-MEL", "WM266", "SKBR3",
    "BT474", "MDA-MB-453", "ZR-75-1", "Cal51", "HCC1954",
]

# ────────────────────────────────────────────────────────────
# 4. FORMATTING HELPERS
# ────────────────────────────────────────────────────────────
def fmt_label(n):
    n = str(n).lower()
    if any(x in n for x in ["label free", "label-free", "lfq", "label_free", "unlab"]):
        return "AC=MS:1002038;NT=label free sample"
    if "tmt" in n:
        m = re.search(r"tmt[\s\-]?(\d+)", n)
        p = m.group(1) if m else "6"
        acc = {"2":"MS:1002456","6":"MS:1002453","10":"MS:1002454",
               "11":"MS:1002454","16":"MS:1003998","18":"MS:1003999"}
        return f"AC={acc.get(p,'MS:1002453')};NT=TMT{p}plex"
    if "itraq" in n:
        m = re.search(r"itraq[\s\-]?(\d+)", n)
        p = m.group(1) if m else "4"
        return f"AC={'MS:1001985' if p=='4' else 'MS:1002519'};NT=iTRAQ{p}plex"
    if "silac" in n: return "AC=MS:1002791;NT=SILAC"
    if "dimethyl" in n: return "AC=MS:1002457;NT=Dimethyl"
    return str(n)

def fmt_instrument(raw):
    n = raw.lower().strip()
    # longest match first (order matters in dict)
    for key, val in instrument_norm.items():
        if key in n:
            return val
    return raw

def _fuzzy_norm(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def fuzzy_snap(value, base_col, cutoff=FUZZY_CUTOFF):
    if not value or base_col not in col_vocab:
        return value

    valid_set = list(col_vocab[base_col])
    if not valid_set:
        return value

    match = rfprocess.extractOne(
        str(value).strip(),
        valid_set,
        scorer=fuzz.token_sort_ratio,
        processor=_fuzzy_norm,
        score_cutoff=cutoff,
    )
    return match[0] if match else value


# ────────────────────────────────────────────────────────────
# 5. TEXT EXTRACTION HELPER
# ────────────────────────────────────────────────────────────
PRIORITY_SECTIONS = [
    "METHODS", "MATERIALS AND METHODS", "MATERIALS", "EXPERIMENTAL",
    "SAMPLE PREPARATION", "PROTEIN DIGESTION", "MASS SPECTROMETRY",
    "LC-MS", "LC-MS/MS", "DATA ACQUISITION", "DATA ANALYSIS", "CELL CULTURE",
]
METHOD_KWS = ["method", "material", "protocol", "procedure", "digest",
              "spectr", "chromat", "lc", "ms", "prep", "enrichment", "culture"]

def get_text(pub_dict, include_abstract=True, max_chars=None):
    parts = []
    for key in PRIORITY_SECTIONS:
        val = pub_dict.get(key, "")
        if isinstance(val, list): val = " ".join(str(x) for x in val)
        if val.strip(): parts.append(val.strip())
    for key, val in pub_dict.items():
        if key.upper() in PRIORITY_SECTIONS: continue
        if any(kw in key.lower() for kw in METHOD_KWS):
            if isinstance(val, list): val = " ".join(str(x) for x in val)
            if val.strip(): parts.append(val.strip())
    if include_abstract:
        for key in ["ABSTRACT", "TITLE"]:
            val = pub_dict.get(key, "")
            if isinstance(val, list): val = " ".join(str(x) for x in val)
            if val.strip(): parts.append(val.strip())
    text = " ".join(parts)
    return text[:max_chars] if max_chars else text

# ────────────────────────────────────────────────────────────
# 6. REGEX EXTRACTION
# ────────────────────────────────────────────────────────────
_NEG = r"(?<!without\s)(?<!no\s)(?<!not\s)"

def _first(m):
    return m[0] if isinstance(m, tuple) else m

# Clinical/patient context signal — used to gate Disease extraction
_CLINICAL = re.compile(
    r"\b(patient|cohort|biopsy|tumor|tumour|cancer|carcinoma|malignant|"
    r"diagnosed|clinical|disease|healthy\s+(?:control|donor)|specimen|"
    r"hospital|surgical|resection)\b", re.I
)

def regex_extraction(pub_dict):
    text      = get_text(pub_dict)
    text_low  = text.lower()
    extracted = {}

    def add(col, val):
        if val:
            extracted.setdefault(col, [])
            if val not in extracted[col]:
                extracted[col].append(val)

    # ── Organism ──────────────────────────────────────────────────────
    for pattern, norm_key in [
        (re.compile(r"\b(homo\s+sapiens|human(?:s)?)\b", re.I), "human"),
        (re.compile(r"\b(mus\s+musculus|mouse|mice|murine)\b", re.I), "mouse"),
        (re.compile(r"\b(rattus\s+norvegicus|rat(?:s)?)\b", re.I), "rat"),
        (re.compile(r"\b(saccharomyces\s+cerevisiae|(?<!\w)yeast(?!\w))\b", re.I), "yeast"),
        (re.compile(r"\b(escherichia\s+coli|e\.?\s*coli)\b", re.I), "e. coli"),
        (re.compile(r"\b(drosophila\s+melanogaster|fruit\s+fly)\b", re.I), "fruit fly"),
        (re.compile(r"\b(danio\s+rerio|zebrafish)\b", re.I), "zebrafish"),
        (re.compile(r"\b(arabidopsis\s+thaliana)\b", re.I), "arabidopsis thaliana"),
        (re.compile(r"\b(sus\s+scrofa|porcine|pig(?:s)?)\b", re.I), "pig"),
        (re.compile(r"\b(bos\s+taurus|bovine)\b", re.I), "bovine"),
        (re.compile(r"\b(caenorhabditis\s+elegans|c\.\s*elegans)\b", re.I), "c. elegans"),
        (re.compile(r"\b(xenopus\s+laevis)\b", re.I), "xenopus laevis"),
    ]:
        if pattern.search(text):
            add("Characteristics[Organism]", organism_norm[norm_key])

    # ── OrganismPart — collect ALL mentioned tissues (not just first) ──
    for tissue, norm in tissue_norm.items():
        if re.search(r'\b' + re.escape(tissue) + r'\b', text_low):
            add("Characteristics[OrganismPart]", norm)

    # ── CellLine ──────────────────────────────────────────────────────
    for cl in CELL_LINES:
        if re.search(r'\b' + re.escape(cl.lower()) + r'\b', text_low):
            add("Characteristics[CellLine]", cl)

    # ── CellType (primary cells) ──────────────────────────────────────
    for ct_pat, ct_val in [
        (re.compile(r"\b(neurons?|neuronal\s+cells?)\b", re.I), "neurons"),
        (re.compile(r"\b(astrocytes?)\b", re.I), "astrocytes"),
        (re.compile(r"\b(microglia)\b", re.I), "microglia"),
        (re.compile(r"\b(macrophages?)\b", re.I), "macrophages"),
        (re.compile(r"\b(dendritic\s+cells?)\b", re.I), "dendritic cells"),
        (re.compile(r"\b(fibroblasts?)\b", re.I), "fibroblasts"),
        (re.compile(r"\b(t[\s\-]cells?|cd4\+|cd8\+)\b", re.I), "T cells"),
        (re.compile(r"\b(b[\s\-]cells?)\b", re.I), "B cells"),
        (re.compile(r"\b(monocytes?)\b", re.I), "monocytes"),
        (re.compile(r"\b(nk\s+cells?|natural\s+killer)\b", re.I), "NK cells"),
        (re.compile(r"\b(pbmc|peripheral\s+blood\s+mononuclear)\b", re.I), "PBMC"),
        (re.compile(r"\b(hepatocytes?)\b", re.I), "hepatocytes"),
        (re.compile(r"\b(cardiomyocytes?)\b", re.I), "cardiomyocytes"),
        (re.compile(r"\b(adipocytes?)\b", re.I), "adipocytes"),
        (re.compile(r"\b(osteoblasts?)\b", re.I), "osteoblasts"),
        (re.compile(r"\b(platelets?|thrombocytes?)\b", re.I), "platelets"),
    ]:
        if ct_pat.search(text):
            add("Characteristics[CellType]", ct_val)

    # ── CleavageAgent ─────────────────────────────────────────────────
    for pat, val in [
        (re.compile(_NEG + r"\b(trypsin(?:/lys[\s\-]?c)?)\b", re.I), "AC=MS:1001251;NT=Trypsin"),
        (re.compile(_NEG + r"\b(lys[\s\-]?c)\b", re.I), "AC=MS:1001255;NT=Lys-C"),
        (re.compile(_NEG + r"\b(glu[\s\-]?c)\b", re.I), "AC=MS:1001917;NT=Glu-C"),
        (re.compile(_NEG + r"\b(chymotrypsin)\b", re.I), "AC=MS:1001306;NT=Chymotrypsin"),
        (re.compile(_NEG + r"\b(asp[\s\-]?n)\b", re.I), "AC=MS:1001267;NT=Asp-N"),
        (re.compile(_NEG + r"\b(arg[\s\-]?c)\b", re.I), "AC=MS:1001303;NT=Arg-C"),
        (re.compile(_NEG + r"\b(cnbr|cyanogen\s+bromide)\b", re.I), "AC=MS:1001325;NT=CNBr"),
    ]:
        if pat.search(text):
            add("Characteristics[CleavageAgent]", val)

    # ── Label ─────────────────────────────────────────────────────────
    for pat, fn in [
        (re.compile(r"\b(tmt[\s\-]?(?:10|11|16|18|2|6)?(?:plex)?)\b", re.I), lambda m: fmt_label(_first(m))),
        (re.compile(r"\b(itraq[\s\-]?(?:4|8)?(?:plex)?)\b", re.I), lambda m: fmt_label(_first(m))),
        (re.compile(r"\b(silac)\b", re.I), lambda m: "AC=MS:1002791;NT=SILAC"),
        (re.compile(r"\b(label[\s\-]free|lfq)\b", re.I), lambda m: "AC=MS:1002038;NT=label free sample"),
        (re.compile(r"\b(dimethyl\s+label(?:ing)?|reductive\s+dimethylation)\b", re.I), lambda m: "AC=MS:1002457;NT=Dimethyl"),
    ]:
        m = pat.search(text)
        if m:
            add("Characteristics[Label]", fn(m.group(1) if m.lastindex else m.group(0)))
            break  # only one label type per paper

    # ── ReductionReagent ──────────────────────────────────────────────
    for pat, val in [
        (re.compile(_NEG + r"\b(dtt|dithiothreitol)\b", re.I), "AC=MS:1000578;NT=DTT"),
        (re.compile(_NEG + r"\b(tcep)\b", re.I), "AC=MS:1001135;NT=TCEP"),
        (re.compile(_NEG + r"\b(beta[\s\-]?mercaptoethanol|bme)\b", re.I), "AC=MS:1000382;NT=beta-mercaptoethanol"),
    ]:
        if pat.search(text):
            add("Characteristics[ReductionReagent]", val)
            break

    # ── AlkylationReagent ─────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(iodoacetamide|iaa)\b", re.I), "AC=PRIDE:0000126;NT=Iodoacetamide"),
        (re.compile(r"\b(n[\s\-]?ethylmaleimide|nem)\b", re.I), "AC=PRIDE:0000459;NT=N-ethylmaleimide"),
        (re.compile(r"\b(chloroacetamide|caa)\b", re.I), "AC=PRIDE:0000126;NT=Chloroacetamide"),
        (re.compile(r"\b(4[\s\-]?vinylpyridine)\b", re.I), "AC=PRIDE:0000101;NT=4-vinylpyridine"),
    ]:
        if pat.search(text):
            add("Characteristics[AlkylationReagent]", val)
            break

    # ── Modifications (collect all) ───────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(carbamidomethyl(?:ation)?|iodoacetamide)\b", re.I),
         "NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed"),
        (re.compile(r"\b(oxidation(?:\s+of\s+methionine)?)\b", re.I),
         "NT=Oxidation;AC=UNIMOD:35;TA=M;MT=Variable"),
        (re.compile(r"\b(phospho(?:rylation)?)\b", re.I),
         "NT=Phospho;AC=UNIMOD:21;TA=S,T,Y;MT=Variable"),
        (re.compile(r"\b(acetyl(?:ation)?(?:\s+of\s+(?:lysine|n.?term))?)\b", re.I),
         "NT=Acetyl;AC=UNIMOD:1;TA=K;MT=Variable"),
        (re.compile(r"\b(ubiquitin(?:ation)?|di[\s\-]?glycine|gg[\s\-]?remnant)\b", re.I),
         "NT=GlyGly;AC=UNIMOD:121;TA=K;MT=Variable"),
        (re.compile(r"\b(methylation(?:\s+of\s+(?:lysine|arginine))?)\b", re.I),
         "NT=Methyl;AC=UNIMOD:34;TA=K,R;MT=Variable"),
        (re.compile(r"\b(sumoylation)\b", re.I),
         "NT=SUMO;AC=UNIMOD:3;TA=K;MT=Variable"),
        (re.compile(r"\b(deamidation(?:\s+of\s+(?:asparagine|glutamine))?)\b", re.I),
         "NT=Deamidated;AC=UNIMOD:7;TA=N,Q;MT=Variable"),
    ]:
        if pat.search(text):
            add("Characteristics[Modification]", val)

    # ── AcquisitionMethod ─────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(dda|data[\s\-]dependent\s+acquisition)\b", re.I), "AC=MS:1003215;NT=DDA"),
        (re.compile(r"\b(dia|data[\s\-]independent\s+acquisition|swath)\b", re.I), "AC=MS:1003215;NT=DIA"),
        (re.compile(r"\b(prm|parallel\s+reaction\s+monitoring)\b", re.I), "AC=MS:1001501;NT=PRM"),
        (re.compile(r"\b(srm|mrm|selected\s+reaction\s+monitoring)\b", re.I), "AC=MS:1001501;NT=SRM"),
    ]:
        if pat.search(text):
            add("Comment[AcquisitionMethod]", val)
            break

    # ── Instrument ────────────────────────────────────────────────────
    for pat in [
        re.compile(r"\b(Q[\s\-]?Exactive[\s\-]?HF[\s\-]?X)\b", re.I),
        re.compile(r"\b(Q[\s\-]?Exactive[\s\-]?HF)\b", re.I),
        re.compile(r"\b(Q[\s\-]?Exactive[\s\-]?Plus)\b", re.I),
        re.compile(r"\b(Q[\s\-]?Exactive)\b", re.I),
        re.compile(r"\b(Orbitrap\s+Astral)\b", re.I),
        re.compile(r"\b(Orbitrap\s+Fusion\s+Lumos)\b", re.I),
        re.compile(r"\b(Orbitrap\s+Fusion)\b", re.I),
        re.compile(r"\b(Orbitrap\s+Eclipse)\b", re.I),
        re.compile(r"\b(Orbitrap\s+Exploris\s+480|Exploris\s+480)\b", re.I),
        re.compile(r"\b(LTQ[\s\-]?Orbitrap\s+Velos)\b", re.I),
        re.compile(r"\b(LTQ[\s\-]?Orbitrap\s+Elite)\b", re.I),
        re.compile(r"\b(LTQ[\s\-]?Orbitrap\s+XL)\b", re.I),
        re.compile(r"\b(LTQ[\s\-]?Orbitrap)\b", re.I),
        re.compile(r"\b(timsTOF\s+Pro)\b", re.I),
        re.compile(r"\b(timsTOF)\b", re.I),
        re.compile(r"\b(Triple[\s\-]?TOF\s+6600)\b", re.I),
        re.compile(r"\b(Triple[\s\-]?TOF\s+5600)\b", re.I),
        re.compile(r"\b(Triple[\s\-]?TOF)\b", re.I),
        re.compile(r"\b(Impact\s+II)\b", re.I),
        re.compile(r"\b(maXis\s+Speed)\b", re.I),
    ]:
        m = pat.search(text)
        if m:
            add("Comment[Instrument]", fmt_instrument(m.group(1)))
            break

    # ── FragmentationMethod ───────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(hcd)\b", re.I), "AC=MS:1002481;NT=HCD"),
        (re.compile(r"\b(cid)\b", re.I), "AC=MS:1001880;NT=CID"),
        (re.compile(r"\b(etd)\b", re.I), "AC=MS:1001526;NT=ETD"),
        (re.compile(r"\b(ecd)\b", re.I), "AC=MS:1001872;NT=ECD"),
        (re.compile(r"\b(uvpd)\b", re.I), "AC=MS:1003246;NT=UVPD"),
    ]:
        if pat.search(text):
            add("Comment[FragmentationMethod]", val)

    # ── IonizationType ────────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(nano[\s\-]?esi|nesi)\b", re.I), "AC=MS:1000398;NT=nanoESI"),
        (re.compile(r"\b(electrospray|(?<!\bnano)esi)\b", re.I), "AC=MS:1000073;NT=ESI"),
        (re.compile(r"\b(maldi)\b", re.I), "AC=MS:1000075;NT=MALDI"),
    ]:
        if pat.search(text):
            add("Comment[IonizationType]", val)
            break

    # ── MS2MassAnalyzer ───────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(orbitrap)\b", re.I), "AC=MS:1000484;NT=Orbitrap"),
        (re.compile(r"\b(ion\s*trap)\b", re.I), "AC=MS:1000264;NT=ion trap"),
        (re.compile(r"\b(tof)\b", re.I), "AC=MS:1000084;NT=TOF"),
        (re.compile(r"\b(quadrupole)\b", re.I), "AC=MS:1000081;NT=Quadrupole"),
    ]:
        if pat.search(text):
            add("Comment[MS2MassAnalyzer]", val)
            break

    # ── FractionationMethod ───────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(sds[\s\-]?page)\b", re.I), "AC=PRIDE:0000672;NT=SDS-PAGE"),
        (re.compile(r"\b(scx|strong\s+cation\s+exchange)\b", re.I), "AC=PRIDE:0000228;NT=SCX"),
        (re.compile(r"\b(sax|strong\s+anion\s+exchange)\b", re.I), "AC=PRIDE:0000229;NT=SAX"),
        (re.compile(r"\b(hprp|high[\s\-]ph\s+rp|high[\s\-]ph\s+reversed[\s\-]phase)\b", re.I),
         "AC=PRIDE:0000550;NT=High-pH Reversed-Phase"),
        (re.compile(r"\b(isoelectric\s+focusing|ief)\b", re.I), "AC=PRIDE:0000006;NT=IEF"),
        (re.compile(r"\b(size[\s\-]exclusion\s+chromatography|sec[\s\-]hplc)\b", re.I), "AC=PRIDE:0000020;NT=SEC"),
        (re.compile(r"\b(offgel)\b", re.I), "AC=PRIDE:0000006;NT=Off-gel IEF"),
    ]:
        if pat.search(text):
            add("Comment[FractionationMethod]", val)
            break

    # ── EnrichmentMethod ──────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(tio2?|titanium\s+dioxide)\b", re.I), "AC=MS:1002088;NT=TiO2"),
        (re.compile(r"\b(imac|immobilized\s+metal\s+affinity)\b", re.I), "AC=MS:1001923;NT=IMAC"),
        (re.compile(r"\b(immunoaffinity|immunoprecipitation|ip[\s\-]ms)\b", re.I),
         "AC=MS:1002090;NT=Immunoprecipitation"),
        (re.compile(r"\b(glycopeptide\s+enrichment|lectin)\b", re.I), "Glycopeptide enrichment"),
        (re.compile(r"\b(streptavidin|avidin[\s\-]biotin)\b", re.I), "Streptavidin affinity"),
    ]:
        if pat.search(text):
            add("Comment[EnrichmentMethod]", val)
            break

    # ── Separation ────────────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(nano[\s\-]?lc)\b", re.I), "AC=PRIDE:0000565;NT=nanoLC"),
        (re.compile(r"\b(uplc|uhplc)\b", re.I), "UPLC"),
        (re.compile(r"\b(rplc|reversed[\s\-]phase\s+lc)\b", re.I), "AC=PRIDE:0000550;NT=Reversed-Phase"),
        (re.compile(r"\b(c18\s+column)\b", re.I), "AC=PRIDE:0000550;NT=Reversed-Phase"),
    ]:
        if pat.search(text):
            add("Comment[Separation]", val)
            break

    # ── Sex ───────────────────────────────────────────────────────────
    if re.search(r"\b(male\s+and\s+female|both\s+sexes|mixed\s+sex|male/female)\b", text, re.I):
        add("Characteristics[Sex]", "male and female")
    elif re.search(r"\b(male\s+(?:donors?|subjects?|patients?|volunteers?|mice|rats?)|(?:^|\s)men\b|male\s+samples?)\b", text, re.I):
        add("Characteristics[Sex]", "male")
    elif re.search(r"\b(female\s+(?:donors?|subjects?|patients?|volunteers?|mice|rats?)|(?:^|\s)women\b|female\s+samples?)\b", text, re.I):
        add("Characteristics[Sex]", "female")

    # ── Disease — only when clinical context is present ───────────────
    if _CLINICAL.search(text):
        for pat, val in [
            (re.compile(r"\b(alzheimer[\s']?s?\s+disease)\b", re.I), "Alzheimer disease"),
            (re.compile(r"\b(parkinson[\s']?s?\s+disease)\b", re.I), "Parkinson disease"),
            (re.compile(r"\b(type\s+2\s+diabetes|t2d(?:m)?)\b", re.I), "type 2 diabetes mellitus"),
            (re.compile(r"\b(type\s+1\s+diabetes|t1d(?:m)?)\b", re.I), "type 1 diabetes mellitus"),
            (re.compile(r"\b(breast\s+(?:cancer|carcinoma))\b", re.I), "breast carcinoma"),
            (re.compile(r"\b(colorectal\s+(?:cancer|carcinoma)|colon\s+cancer)\b", re.I), "colorectal carcinoma"),
            (re.compile(r"\b(lung\s+(?:cancer|carcinoma))\b", re.I), "lung carcinoma"),
            (re.compile(r"\b(glioblastoma|gbm)\b", re.I), "glioblastoma"),
            (re.compile(r"\b(melanoma)\b", re.I), "melanoma"),
            (re.compile(r"\b(prostate\s+(?:cancer|carcinoma))\b", re.I), "prostate carcinoma"),
            (re.compile(r"\b(ovarian\s+(?:cancer|carcinoma))\b", re.I), "ovarian carcinoma"),
            (re.compile(r"\b(hepatocellular\s+carcinoma|hcc)\b", re.I), "hepatocellular carcinoma"),
            (re.compile(r"\b(pancreatic\s+(?:cancer|carcinoma|ductal\s+adenocarcinoma)|pdac)\b", re.I), "pancreatic ductal adenocarcinoma"),
            (re.compile(r"\b(covid[\s\-]?19|sars[\s\-]?cov[\s\-]?2)\b", re.I), "COVID-19"),
            (re.compile(r"\b(multiple\s+myeloma)\b", re.I), "multiple myeloma"),
            (re.compile(r"\b(acute\s+myeloid\s+leukemia|aml)\b", re.I), "acute myeloid leukemia"),
            (re.compile(r"\b(non[\s\-]small[\s\-]cell\s+lung|nsclc)\b", re.I), "non-small cell lung carcinoma"),
            (re.compile(r"\b(triple[\s\-]negative\s+breast|tnbc)\b", re.I), "triple-negative breast carcinoma"),
            (re.compile(r"\b(healthy\s+(?:controls?|donors?|volunteers?|individuals?))\b", re.I), "normal"),
        ]:
            if pat.search(text):
                add("Characteristics[Disease]", val)

    # ── MaterialType — infer from other fields ────────────────────────
    if "Characteristics[CellLine]" in extracted:
        add("Characteristics[MaterialType]", "cell line")
    elif "Characteristics[CellType]" in extracted:
        add("Characteristics[MaterialType]", "primary cells")
    elif re.search(r"\b(tissue(?:s)?(?!\s+culture)|biopsy|biopsies|tumor|tumour)\b", text, re.I):
        add("Characteristics[MaterialType]", "tissue")
    elif re.search(r"\b(plasma|serum|urine|csf|saliva|blood)\b", text, re.I):
        add("Characteristics[MaterialType]", "biofluid")
    elif re.search(r"\b(organoid)\b", text, re.I):
        add("Characteristics[MaterialType]", "organoid")

    # ── Specimen ──────────────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(ffpe|formalin[\s\-]fixed)\b", re.I), "FFPE"),
        (re.compile(r"\b(fresh[\s\-]frozen)\b", re.I), "fresh frozen tissue"),
        (re.compile(r"\b(whole\s+blood)\b", re.I), "whole blood"),
        (re.compile(r"\b(cell\s+lysate(?:s)?)\b", re.I), "cell lysate"),
        (re.compile(r"\b(biopsy|biopsies)\b", re.I), "biopsy"),
        (re.compile(r"\b(urine)\b", re.I), "urine"),
        (re.compile(r"\b(serum)\b", re.I), "serum"),
        (re.compile(r"\b(plasma)\b", re.I), "plasma"),
    ]:
        if pat.search(text):
            add("Characteristics[Specimen]", val)
            break

    # ── Strain ────────────────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(C57BL/6J?)\b"), "C57BL/6J"),
        (re.compile(r"\b(BALB/c)\b"), "BALB/c"),
        (re.compile(r"\b(FVB/N)\b"), "FVB/N"),
        (re.compile(r"\b(129/Sv(?:ev)?)\b", re.I), "129/SvEv"),
        (re.compile(r"\b(Sprague[\s\-]Dawley)\b", re.I), "Sprague-Dawley"),
        (re.compile(r"\b(Wistar)\b", re.I), "Wistar"),
        (re.compile(r"\b(nude\s+mice?|athymic\s+nude)\b", re.I), "nude"),
        (re.compile(r"\b(NOD/SCID)\b", re.I), "NOD/SCID"),
        (re.compile(r"\b(ob/ob)\b", re.I), "ob/ob"),
        (re.compile(r"\b(db/db)\b", re.I), "db/db"),
    ]:
        m = pat.search(text)
        if m:
            add("Characteristics[Strain]", val)
            break

    # ── Genotype ──────────────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(wild[\s\-]?type|wt(?:\s+cells?|\s+mice)?)\b", re.I), "wild-type"),
        (re.compile(r"\b(knockout|knock[\s\-]out|ko(?:\s+cells?|\s+mice)?)\b", re.I), "knockout"),
        (re.compile(r"\b(knock[\s\-]?in)\b", re.I), "knock-in"),
        (re.compile(r"\b(transgenic)\b", re.I), "transgenic"),
        (re.compile(r"\b(heterozygous)\b", re.I), "heterozygous"),
        (re.compile(r"\b(homozygous)\b", re.I), "homozygous"),
    ]:
        if pat.search(text):
            add("Characteristics[Genotype]", val)
            break

    # ── DevelopmentalStage ────────────────────────────────────────────
    for pat, val in [
        (re.compile(r"\b(adult(?:s)?)\b", re.I), "adult"),
        (re.compile(r"\b(embryo(?:nic)?|E\d{1,2}\.?\d?)\b", re.I), "embryo"),
        (re.compile(r"\b(fetal|fetus|foetal)\b", re.I), "fetal"),
        (re.compile(r"\b(neonatal|newborn|postnatal\s+day\s+[0-5])\b", re.I), "neonatal"),
        (re.compile(r"\b(juvenile|adolescent)\b", re.I), "juvenile"),
    ]:
        if pat.search(text):
            add("Characteristics[DevelopmentalStage]", val)
            break

    # ── GradientTime ──────────────────────────────────────────────────
    for pat in [
        re.compile(r"(\d+)[\s\-]min(?:ute)?\s+(?:gradient|lc[\s\-]?ms?|linear\s+gradient|run)", re.I),
        re.compile(r"gradient\s+(?:of\s+)?(\d+)[\s\-]?min", re.I),
        re.compile(r"(\d+)[\s\-]min(?:ute)?\s+(?:separation|elution)", re.I),
        re.compile(r"over\s+(\d+)\s+min(?:utes?)?", re.I),
    ]:
        m = pat.search(text)
        if m:
            add("Comment[GradientTime]", f"{m.group(1)} min")
            break

    # ── FlowRateChromatogram ──────────────────────────────────────────
    m = re.search(r"(\d+(?:\.\d+)?)\s*(nl|nL|ul|uL|µl|µL)/min", text, re.I)
    if m:
        unit = "nL" if m.group(2).lower() == "nl" else "µL"
        add("Comment[FlowRateChromatogram]", f"{m.group(1)} {unit}/min")

    # ── PrecursorMassTolerance ────────────────────────────────────────
    for pat in [
        re.compile(r"(?:precursor|ms1|parent)\s+(?:mass\s+)?tolerance(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*(ppm|da)", re.I),
        re.compile(r"(\d+(?:\.\d+)?)\s*ppm\s+(?:precursor|ms1|parent|for\s+ms1)", re.I),
        re.compile(r"ms1\s+(?:mass\s+)?accuracy\s+of\s+(\d+(?:\.\d+)?)\s*(ppm)", re.I),
    ]:
        m = pat.search(text)
        if m:
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 else "ppm"
            add("Comment[PrecursorMassTolerance]", f"{m.group(1)} {unit}")
            break

    # ── FragmentMassTolerance ─────────────────────────────────────────
    for pat in [
        re.compile(r"(?:fragment|ms2|product)\s+(?:mass\s+)?tolerance(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*(ppm|da|mda)", re.I),
        re.compile(r"(\d+(?:\.\d+)?)\s*(da|mda)\s+(?:for\s+)?(?:fragment|ms2|product)", re.I),
        re.compile(r"ms2\s+(?:mass\s+)?accuracy\s+of\s+(\d+(?:\.\d+)?)\s*(da|ppm)", re.I),
    ]:
        m = pat.search(text)
        if m:
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 else "Da"
            add("Comment[FragmentMassTolerance]", f"{m.group(1)} {unit}")
            break

    # ── NumberOfMissedCleavages ───────────────────────────────────────
    for pat in [
        re.compile(r"(?:up\s+to\s+|allowing\s+(?:up\s+to\s+)?)(\d)\s+missed\s+cleavages?", re.I),
        re.compile(r"(\d)\s+missed\s+cleavages?\s+(?:were\s+)?allowed", re.I),
        re.compile(r"missed\s+cleavages?\s*[=:]\s*(\d)", re.I),
        re.compile(r"maximum\s+(?:of\s+)?(\d)\s+missed\s+cleavages?", re.I),
    ]:
        m = pat.search(text)
        if m:
            add("Comment[NumberOfMissedCleavages]", m.group(1))
            break

    # ── NumberOfBiologicalReplicates ──────────────────────────────────
    for pat in [
        re.compile(r"(\d+)\s+(?:independent\s+)?biological\s+replicates?", re.I),
        re.compile(r"biological\s+replicates?\s+\(n\s*[=≥]\s*(\d+)\)", re.I),
        re.compile(r"n\s*=\s*(\d+)\s+(?:independent\s+)?(?:biological\s+)?replicates?", re.I),
        re.compile(r"performed\s+in\s+(triplicate|duplicate|quadruplicate)", re.I),
        re.compile(r"(?:biological\s+)?triplicates?\b", re.I),
    ]:
        m = pat.search(text)
        if m:
            word_map = {"triplicate": "3", "duplicate": "2", "quadruplicate": "4"}
            val = word_map.get(m.group(1).lower(), m.group(1)) if m.lastindex else "3"
            add("Characteristics[NumberOfBiologicalReplicates]", val)
            break

    # ── NumberOfTechnicalReplicates ───────────────────────────────────
    for pat in [
        re.compile(r"(\d+)\s+technical\s+replicates?", re.I),
        re.compile(r"technical\s+(triplicates?|duplicates?)", re.I),
        re.compile(r"injected\s+(\d+)\s+times?", re.I),
    ]:
        m = pat.search(text)
        if m:
            word_map = {"triplicate": "3", "triplicates": "3", "duplicate": "2", "duplicates": "2"}
            val = word_map.get(m.group(1).lower(), m.group(1)) if m.lastindex else "2"
            add("Characteristics[NumberOfTechnicalReplicates]", val)
            break

    # ── NumberOfSamples ───────────────────────────────────────────────
    for pat in [
        re.compile(r"(\d+)\s+(?:clinical\s+)?(?:patient|tumor|tissue|plasma|serum|urine)\s+samples?", re.I),
        re.compile(r"cohort\s+of\s+(\d+)\s+(?:patients?|subjects?|individuals?|participants?)", re.I),
        re.compile(r"(\d+)\s+(?:patients?|subjects?|individuals?|participants?)\s+(?:were|with|diagnosed)", re.I),
        re.compile(r"total\s+of\s+(\d+)\s+samples?", re.I),
    ]:
        m = pat.search(text)
        if m:
            add("Characteristics[NumberOfSamples]", m.group(1))
            break

    # ── NumberOfFractions ─────────────────────────────────────────────
    for pat in [
        re.compile(r"(?:fractionated\s+into|collected\s+into|divided\s+into)\s+(\d+)\s+fractions?", re.I),
        re.compile(r"(\d+)\s+(?:scx|hprp|rp|sds[\s\-]?page|gel)?\s*fractions?\s+(?:were|of)", re.I),
        re.compile(r"(\d+)[\s\-]fraction\s+(?:separation|fractionation)", re.I),
    ]:
        m = pat.search(text)
        if m:
            add("Comment[NumberOfFractions]", m.group(1))
            break

    # ── TumorStage ────────────────────────────────────────────────────
    m = re.search(r"\b(?:ajcc\s+)?(?:clinical\s+)?stage\s+([IViv]+)\b", text, re.I)
    if m:
        add("Characteristics[TumorStage]", f"Stage {m.group(1).upper()}")

    # ── TumorGrade ────────────────────────────────────────────────────
    for pat in [
        re.compile(r"\bgrade\s+([IViv\d]+)\b", re.I),
        re.compile(r"\bgleason\s+(?:score\s+)?(\d+)\b", re.I),
        re.compile(r"\b(poorly|well|moderately)\s+differentiated\b", re.I),
    ]:
        m = pat.search(text)
        if m:
            add("Characteristics[TumorGrade]", f"Grade {m.group(1)}")
            break

    # Cap noisy multi-value fields
    for col in CAPPED_MULTI_VALUE_COLS:
        if col in extracted:
            extracted[col] = extracted[col][:MULTI_VAL_CAP]

    return extracted

# ────────────────────────────────────────────────────────────
# 7. API FETCHERS
# ────────────────────────────────────────────────────────────
http = requests.Session()
http.headers.update({"User-Agent": "SDRF-Extractor/8.0"})

def fetch_pride(pxd):
    try:
        r = http.get(f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}",
                     timeout=PRIDE_TIMEOUT)
        if r.status_code != 200: return {}
        d = r.json()
        out = defaultdict(list)
        for o in d.get("organisms", []):
            name = o.get("name","")
            if name:
                norm = organism_norm.get(name.lower().strip())
                out["Characteristics[Organism]"].append(norm or name.strip())
        for op in (d.get("organisms_part") or d.get("tissues") or []):
            name = op.get("name",""); acc = op.get("accession","")
            if name and name.lower() not in ("not available","n/a",""):
                out["Characteristics[OrganismPart]"].append(
                    f"NT={name};AC={acc}" if acc else f"NT={name}"
                )
        for dis in d.get("diseases",[]):
            name = dis.get("name","")
            if name and name.lower() not in ("not available","n/a","none","normal",""):
                out["Characteristics[Disease]"].append(name)
        for inst in d.get("instruments",[]):
            name = inst.get("name",""); acc = inst.get("accession","")
            if name:
                fmt = fmt_instrument(name)
                out["Comment[Instrument]"].append(
                    fmt if "AC=" in fmt else (f"AC={acc};NT={name}" if acc else name)
                )
        for qm in d.get("quantification_methods",[]):
            name = qm.get("name","")
            if name: out["Characteristics[Label]"].append(fmt_label(name))
        return {k: list(dict.fromkeys(v)) for k,v in out.items() if v}
    except Exception:
        return {}

def fetch_px_xml(pxd):
    out = defaultdict(list)
    try:
        url = f"https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID={pxd}&outputMode=XML&test=no"
        r = http.get(url, timeout=PX_TIMEOUT)
        if r.status_code != 200: return {}
        xml = r.text
        for m in re.finditer(r'<cvParam[^>]+accession="(MS:\d+)"[^>]+name="([^"]+)"', xml):
            acc, name = m.group(1), m.group(2)
            if "instrument" in name.lower():
                out["Comment[Instrument]"].append(fmt_instrument(name))
        for m in re.finditer(r'<cvParam[^>]+accession="(NEWT:\d+)"[^>]+name="([^"]+)"', xml):
            tax = m.group(1).replace("NEWT:","")
            name = m.group(2)
            norm = organism_norm.get(name.lower().strip())
            out["Characteristics[Organism]"].append(norm if norm else f"{tax} ({name})")
    except Exception:
        pass
    return {k: list(dict.fromkeys(v)) for k,v in out.items() if v}

# ────────────────────────────────────────────────────────────
# 8. FILENAME TOKEN PARSER
# ────────────────────────────────────────────────────────────
def parse_filename_tokens(raw_files):
    """
    Extract per-file metadata from filename tokens.
    Returns dict: col -> list of per-file values (same order as raw_files).
    """
    n = len(raw_files)
    results = {
        "Comment[FractionIdentifier]":       [None] * n,
        "Characteristics[BiologicalReplicate]": [None] * n,
        "Characteristics[TechnicalReplicate]":  [None] * n,
        "Characteristics[Label]":              [None] * n,
    }

    for i, rf in enumerate(raw_files):
        rf_str = str(rf)
        rf_up  = rf_str.upper()

        # Fraction
        m = re.search(r"(?:_f|_fr|_frac(?:tion)?)[_\s]?(\d+)", rf_str, re.I)
        if m: results["Comment[FractionIdentifier]"][i] = m.group(1)

        # Biological replicate
        m = re.search(r"(?:_rep|_br|_biol?rep|_biorep|[_\-]r)(\d+)", rf_str, re.I)
        if m: results["Characteristics[BiologicalReplicate]"][i] = f"biological replicate {m.group(1)}"

        # Technical replicate
        m = re.search(r"(?:_tr|_tech(?:rep)?|_inj|_techrep)(\d+)", rf_str, re.I)
        if m: results["Characteristics[TechnicalReplicate]"][i] = f"technical replicate {m.group(1)}"

        # Label from filename
        if re.search(r"TMT", rf_up):
            # Try to get plex from filename e.g. TMT10, TMT16
            mp = re.search(r"TMT(\d+)", rf_up)
            results["Characteristics[Label]"][i] = fmt_label(f"tmt{mp.group(1)}" if mp else "tmt")
        elif re.search(r"SILAC|HEAVY|_H_|_L_|LIGHT", rf_up):
            results["Characteristics[Label]"][i] = "AC=MS:1002791;NT=SILAC"
        elif re.search(r"LFQ|LF_|_LF\d|LABELFREE", rf_up):
            results["Characteristics[Label]"][i] = "AC=MS:1002038;NT=label free sample"

    return results

# ────────────────────────────────────────────────────────────
# 9. GEMINI API EXTRACTION
# ────────────────────────────────────────────────────────────

def build_gemini_text(pub_dict):
    parts = []

    for key in ["TITLE", "ABSTRACT"]:
        v = pub_dict.get(key, "")
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if str(v).strip():
            parts.append(f"[{key}]\n{str(v).strip()}")

    method_keys = [
        "METHODS", "MATERIALS AND METHODS", "EXPERIMENTAL",
        "SAMPLE PREPARATION", "MASS SPECTROMETRY", "LC-MS",
        "PROTEIN DIGESTION", "DATA ACQUISITION", "CELL CULTURE"
    ]
    for key in method_keys:
        v = pub_dict.get(key, "")
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if str(v).strip():
            parts.append(f"[{key}]\n{str(v).strip()}")

    method_kws = [
        "method", "material", "protocol", "digest", "spectr",
        "chromat", "prep", "enrichment", "culture", "experimental"
    ]
    seen = {k.upper() for k in ["TITLE", "ABSTRACT"] + method_keys}
    for key, v in pub_dict.items():
        if key.upper() in seen:
            continue
        if any(kw in key.lower() for kw in method_kws):
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            if str(v).strip():
                parts.append(f"[{key}]\n{str(v).strip()}")

    return "\n\n".join(parts)

def gemini_extract(pub_dict, pxd):
    if not USE_GEMINI:
        return {}

    text = build_gemini_text(pub_dict)
    if not text.strip():
        return {}

    full_prompt = f"{BASE_PROMPT}\n\nPAPER TEXT:\n{text}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": GEMINI_TEMPERATURE,
            "responseMimeType": "application/json",
        }
    }

    try:
        print(f"Calling Gemini for {pxd} ...", flush=True)
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=(30, 600),
        )
        print(f"Gemini returned for {pxd}: {resp.status_code}", flush=True)
        resp.raise_for_status()
        data = resp.json()

        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        result = json.loads(raw)

        normalized = {}
        for k, v in result.items():
            if isinstance(v, list):
                cleaned = [str(x).strip() for x in v if str(x).strip()]
                if cleaned:
                    normalized[k] = cleaned
            elif v is not None and str(v).strip():
                normalized[k] = [str(v).strip()]

        out_path = TEST_EXTRACT_DIR / f"{pxd}_metadata.json"
        out_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(1)  # rate limiting safety
        return normalized

    except Exception as e:
        print(f"Gemini error for {pxd}: {e}")
        return {}
    
# ────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ────────────────────────────────────────────────────────────
print("Loading test texts...")
test_docs = {}
for path in sorted(TEST_PUBTEXT_DIR.glob("PXD*_PubText.json")):
    pxd = path.stem.replace("_PubText", "")
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if raw: test_docs[pxd] = json.loads(raw)
    except Exception:
        pass
print(f"Loaded {len(test_docs)} test papers")

predicted_sets = defaultdict(dict)

for pxd, pub_dict in tqdm(test_docs.items(), desc="Extracting"):
    col_vals = defaultdict(list)

    def add(col, val):
        if not val or str(val).strip().lower() in ("not applicable","na","n/a",""): return
        base_col    = re.sub(r"\.\d+$", "", col)
        snapped_val = fuzzy_snap(str(val).strip(), base_col)
        if snapped_val not in col_vals[col]:
            col_vals[col].append(snapped_val)

    def add_list(col, vals):
        for v in (vals or []): add(col, v)

    # Priority 1: Ground truth overlap
    if pxd in train_pxd_sdrf:
        for col, vals in train_pxd_sdrf[pxd].items():
            add_list(col, vals)

    # Priority 2: PRIDE API
    for col, vals in fetch_pride(pxd).items():
        add_list(col, vals)

    # Priority 3: ProteomeXchange XML
    for col, vals in fetch_px_xml(pxd).items():
        add_list(col, vals)

    # Priority 4: Gemini LLM
    if USE_GEMINI:
        for col, vals in gemini_extract(pub_dict, pxd).items():
            add_list(col, vals)

    # Priority 5: Regex extraction
    for col, vals in regex_extraction(pub_dict).items():
        add_list(col, vals)

    # Priority 6: Global fallback
    found_base = set(re.sub(r"\.\d+$", "", c) for c in col_vals.keys())
    for col in list(all_base - found_base):
        if non_na_ratio.get(col, 0.0) > 0.80:
            add(col, global_modes.get(col, "Not Applicable"))

    # Cap noisy multi-value fields after all sources have been merged
    for col in CAPPED_MULTI_VALUE_COLS:
        if col in col_vals:
            col_vals[col] = list(dict.fromkeys(col_vals[col]))[:MULTI_VAL_CAP]

    # Deduplicate Modification slots
    mods = list(dict.fromkeys(col_vals.pop("Characteristics[Modification]", [])))[:MULTI_VAL_CAP]
    for i, mod in enumerate(mods):
        slot = "Characteristics[Modification]" if i == 0 else f"Characteristics[Modification].{i}"
        col_vals[slot] = [mod]

    for col, vals in col_vals.items():
        predicted_sets[pxd][col] = list(dict.fromkeys([v for v in vals if v]))

# ────────────────────────────────────────────────────────────
# 11. BUILD SUBMISSION
# ────────────────────────────────────────────────────────────
def assign_values_to_rows(n_rows, values):
    return [values[i % len(values)] for i in range(n_rows)]

final_sub = sample_sub.copy()
for col in target_cols:
    final_sub[col] = "Not Applicable"

for pxd, pxd_df in final_sub.groupby("PXD"):
    idx      = pxd_df.index
    extr     = predicted_sets.get(pxd, {})
    raw_files = pxd_df["Raw Data File"].tolist()

    # Parse filename tokens for this PXD
    fn_tokens = parse_filename_tokens(raw_files)

    for col in target_cols:
        base_col = re.sub(r"\.\d+$", "", col)

        # ── Filename-derived per-row columns ──────────────────────────
        if col in fn_tokens:
            per_row = fn_tokens[col]
            if any(v is not None for v in per_row):
                for i, row_idx in enumerate(idx):
                    final_sub.at[row_idx, col] = per_row[i] or "Not Applicable"
                # If filename gave a label, don't overwrite with extracted
                if col == "Characteristics[Label]":
                    continue
                continue

        vals = extr.get(col) or extr.get(base_col) or []
        vals = [v for v in vals if str(v).strip().lower() not in ("not applicable","")]

        # ── Label fallback from filename if regex found nothing ────────
        if col == "Characteristics[Label]" and not vals:
            fn_labels = set()
            for rf in raw_files:
                ru = str(rf).upper()
                if re.search(r"TMT", ru):
                    mp = re.search(r"TMT(\d+)", ru)
                    fn_labels.add(fmt_label(f"tmt{mp.group(1)}" if mp else "tmt"))
                elif re.search(r"SILAC|HEAVY|LIGHT", ru):
                    fn_labels.add("AC=MS:1002791;NT=SILAC")
                elif re.search(r"LFQ|LF_|LF\d", ru):
                    fn_labels.add("AC=MS:1002038;NT=label free sample")
            vals = list(fn_labels)

        if vals:
            assigned = assign_values_to_rows(len(idx), vals)
            for i, row_idx in enumerate(idx):
                final_sub.at[row_idx, col] = assigned[i]
        else:
            # fb = global_modes.get(col,"Not Applicable") if non_na_ratio.get(col,0.0) > 0.80 else "Not Applicable"
            if non_na_ratio.get(base_col, 0.0) > 0.80:
                fb = global_modes.get(col, "Not Applicable")
            else:
                fb = "Not Applicable"
            for row_idx in idx:
                final_sub.at[row_idx, col] = fb

final_sub = final_sub.fillna("Not Applicable")
if "Unnamed: 0" in final_sub.columns:
    final_sub = final_sub.drop(columns=["Unnamed: 0"])
for col in target_cols:
    mask = final_sub[col].astype(str).str.strip().isin(["TextSpan","nan","None","[]","","null"])
    final_sub.loc[mask, col] = "Not Applicable"

out_path = SUBMISSION_DIR / "submission.csv"
final_sub.to_csv(out_path, index=False)
print(f"{out_path.name} saved — shape {final_sub.shape}")
