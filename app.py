import os
import re
import joblib
import numpy as np
import streamlit as st
import torch

from transformers import AutoTokenizer,AutoModel
from xgboost import XGBClassifier
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from github_analyzer import (
    clone_repository,
    get_repository_files,
    get_code_files,
    read_code_file,
    read_readme,
    detect_dependencies,
    detect_tech_stack,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RepoLens AI",
    page_icon="🔍",
    layout="wide"
)

# =========================================================
# COMPACT PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Main hero / section cards */
    .hero-card {
        padding: 32px 38px !important;
        min-height: 0 !important;
        margin-bottom: 22px !important;
    }

    .hero-card h1 {
        font-size: 2.45rem !important;
        line-height: 1.15 !important;
        margin-bottom: 14px !important;
    }

    .hero-card p {
        font-size: 0.98rem !important;
        line-height: 1.55 !important;
    }

    /* Compact Streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.65rem !important;
    }

    div[data-testid="stMetric"] {
        padding: 0.55rem 0.7rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
    }

    /* Inputs and buttons */
    div[data-baseweb="input"] {
        min-height: 42px !important;
    }

    div[data-baseweb="textarea"] textarea {
        min-height: 90px !important;
    }

    .stButton > button {
        min-height: 40px !important;
        padding: 0.45rem 0.9rem !important;
        font-size: 0.88rem !important;
    }

    /* Reduce vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.8rem;
    }

    /* Expander */
    details[data-testid="stExpander"] summary {
        padding: 0.65rem 0.8rem !important;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        max-height: 360px;
    }

    /* Navigation */
    .nav-label {
        text-align: center;
        color: #8ea4c5;
        font-size: 0.78rem;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "v3_tfidf_vectorizer.pkl"
)

SELECTOR_PATH = os.path.join(
    BASE_DIR,
    "v3_tfidf_selector.pkl"
)

LANGUAGE_MAP_PATH = os.path.join(
    BASE_DIR,
    "v3_language_map.pkl"
)

XGBOOST_PATH = os.path.join(
    BASE_DIR,
    "v3_xgboost_hybrid_model.json"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# HUGGING FACE
# =========================================================

load_dotenv()

# Local .env
HF_TOKEN = os.getenv("HF_TOKEN")

# Streamlit Cloud Secrets
if not HF_TOKEN:
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except Exception:
        HF_TOKEN = None


HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-Coder-32B-Instruct"
)

# Get model from Streamlit Secrets if available
try:
    if "HF_MODEL" in st.secrets:
        HF_MODEL = st.secrets["HF_MODEL"]
except Exception:
    pass


if HF_TOKEN:
    hf_client = InferenceClient(
        token=HF_TOKEN
    )
else:
    hf_client = None


# =========================================================
# SESSION STATE
# =========================================================

default_state = {
    "repo_path": None,
    "all_files": [],
    "code_files": [],
    "technologies": [],
    "dependencies": [],
    "readme_content": "",
    "results": [],
    "codes": [],
    "filenames": [],
    "ai_fixes": {},
    "health": None,

    "comparison_a": None,
    "comparison_b": None,
    "comparison_results": None,
    "page": "Home",
    "main_repo_url": ""
}

for key, value in default_state.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base"
    )

    codebert = AutoModel.from_pretrained(
        "microsoft/codebert-base"
    )

    codebert.to(DEVICE)
    codebert.eval()

    tfidf = joblib.load(
        TFIDF_PATH
    )

    selector = joblib.load(
        SELECTOR_PATH
    )

    language_map = joblib.load(
        LANGUAGE_MAP_PATH
    )

    xgb_model = XGBClassifier()

    xgb_model.load_model(
        XGBOOST_PATH
    )

    return (
        tokenizer,
        codebert,
        tfidf,
        selector,
        language_map,
        xgb_model
    )


# =========================================================
# CODEBERT EMBEDDINGS
# =========================================================

def get_codebert_embeddings_batch(
    codes,
    tokenizer,
    model,
    batch_size=16
):

    embeddings = []

    for i in range(
        0,
        len(codes),
        batch_size
    ):

        batch_codes = codes[
            i:i + batch_size
        ]

        encoded = tokenizer(
            batch_codes,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(
                **encoded
            )

            token_embeddings = (
                outputs.last_hidden_state
            )

            attention_mask = (
                encoded["attention_mask"]
            )

            mask = (
                attention_mask
                .unsqueeze(-1)
                .expand(
                    token_embeddings.size()
                )
                .float()
            )

            summed = torch.sum(
                token_embeddings * mask,
                dim=1
            )

            summed_mask = torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )

            mean_pooled = (
                summed / summed_mask
            )

            embeddings.append(
                mean_pooled.cpu().numpy()
            )

    return np.vstack(
        embeddings
    )


# =========================================================
# STRUCTURAL FEATURES
# =========================================================

def structural_features(code):

    lines = code.splitlines()

    loc = len(lines)

    blank_lines = sum(
        1
        for line in lines
        if not line.strip()
    )

    comments = sum(
        1
        for line in lines
        if line.strip().startswith(
            ("#", "//", "/*", "*")
        )
    )

    functions = len(
        re.findall(
            r"\bdef\s+\w+|\bfunction\s+\w+|\bfunc\s+\w+",
            code
        )
    )

    classes = len(
        re.findall(
            r"\bclass\s+\w+",
            code
        )
    )

    if_count = len(
        re.findall(
            r"\bif\s*\(",
            code
        )
    )

    elif_count = len(
        re.findall(
            r"\belif\b|\belse\s+if\b",
            code
        )
    )

    loop_count = len(
        re.findall(
            r"\bfor\b|\bwhile\b",
            code
        )
    )

    try_count = len(
        re.findall(
            r"\btry\b",
            code
        )
    )

    except_count = len(
        re.findall(
            r"\bexcept\b|\bcatch\b",
            code
        )
    )

    return_count = len(
        re.findall(
            r"\breturn\b",
            code
        )
    )

    import_count = len(
        re.findall(
            r"\bimport\b|\bfrom\b.*\bimport\b|#include",
            code
        )
    )

    assignment_count = len(
        re.findall(
            r"(?<![=!<>])=(?!=)",
            code
        )
    )

    comparison_count = len(
        re.findall(
            r"==|!=|<=|>=|<|>",
            code
        )
    )

    arithmetic_count = len(
        re.findall(
            r"\+|\-|\*|/",
            code
        )
    )

    division_count = len(
        re.findall(
            r"/",
            code
        )
    )

    zero_division_pattern = len(
        re.findall(
            r"/\s*0\b",
            code
        )
    )

    null_pattern = len(
        re.findall(
            r"\bnull\b|\bNone\b|\bnil\b",
            code
        )
    )

    print_count = len(
        re.findall(
            r"\bprint\s*\(",
            code
        )
    )

    raise_count = len(
        re.findall(
            r"\braise\b|\bthrow\b",
            code
        )
    )

    token_count = len(
        re.findall(
            r"\w+|[^\w\s]",
            code
        )
    )

    non_empty_lines = [
        line
        for line in lines
        if line.strip()
    ]

    if non_empty_lines:

        avg_line_length = (
            sum(
                len(line)
                for line in non_empty_lines
            )
            / len(non_empty_lines)
        )

        max_line_length = max(
            len(line)
            for line in non_empty_lines
        )

    else:

        avg_line_length = 0
        max_line_length = 0

    max_nesting = 0

    for line in lines:

        stripped = line.lstrip()

        if stripped:

            indentation = (
                len(line)
                - len(stripped)
            )

            max_nesting = max(
                max_nesting,
                indentation // 4
            )

    complexity = (
        1
        + if_count
        + elif_count
        + loop_count
        + try_count
    )

    safe_loc = max(
        loc,
        1
    )

    return [

        loc,
        blank_lines,
        comments,
        functions,
        classes,
        if_count,
        elif_count,
        loop_count,
        try_count,
        except_count,
        return_count,
        import_count,
        assignment_count,
        comparison_count,
        arithmetic_count,
        division_count,
        zero_division_pattern,
        null_pattern,
        print_count,
        raise_count,
        token_count,
        avg_line_length,
        max_line_length,
        max_nesting,
        complexity,
        complexity / safe_loc,
        functions / safe_loc,
        comments / safe_loc,
        assignment_count / safe_loc,
        comparison_count / safe_loc
    ]


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language_from_file(filename):

    ext = os.path.splitext(
        filename
    )[1].lower()

    mapping = {

        ".py": "Python",

        ".java": "Java",

        ".js": "JavaScript",
        ".ts": "JavaScript",

        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".hpp": "C++",

        ".c": "C",
        ".h": "C",

        ".cs": "C#",

        ".php": "PHP",

        ".rb": "Ruby",

        ".go": "Go"
    }

    return mapping.get(
        ext,
        None
    )


# =========================================================
# BUG PREDICTION
# =========================================================

def predict_batch(
    codes,
    filenames,
    tokenizer,
    codebert,
    tfidf,
    selector,
    language_map,
    xgb_model
):

    if not codes:
        return []

    codebert_features = (
        get_codebert_embeddings_batch(
            codes,
            tokenizer,
            codebert
        )
    )

    tfidf_full = tfidf.transform(
        codes
    )

    tfidf_features = (
        selector
        .transform(tfidf_full)
        .toarray()
        .astype(np.float32)
    )

    structural = np.array(
        [
            structural_features(code)
            for code in codes
        ],
        dtype=np.float32
    )

    language_values = []

    for filename in filenames:

        language = (
            detect_language_from_file(
                filename
            )
        )

        language_values.append(
            language_map.get(
                language,
                -1
            )
        )

    language_features = np.array(
        language_values,
        dtype=np.float32
    ).reshape(-1, 1)

    X = np.hstack(
        [
            codebert_features,
            tfidf_features,
            structural,
            language_features
        ]
    )

    expected_features = 10799

    if X.shape[1] != expected_features:

        raise ValueError(
            f"Feature mismatch: "
            f"Expected {expected_features}, "
            f"got {X.shape[1]}"
        )

    predictions = xgb_model.predict(X)

    probabilities = (
        xgb_model.predict_proba(X)
    )

    results = []

    for (
        filename,
        prediction,
        probability
    ) in zip(
        filenames,
        predictions,
        probabilities
    ):

        confidence = float(
            np.max(probability) * 100
        )

        if int(prediction) == 0:

            label = "Bug"

        else:

            label = "No Bug"

        results.append(
            {
                "filename": filename,
                "label": label,
                "confidence": confidence
            }
        )

    return results


# =========================================================
# LINE NUMBERS
# =========================================================

def add_line_numbers(code):

    lines = code.splitlines()

    return "\n".join(
        f"{i + 1:4} | {line}"
        for i, line in enumerate(lines)
    )


# =========================================================
# AI FIX
# =========================================================

def generate_ai_fix(
    code,
    filename
):

    if hf_client is None:

        return (
            "Hugging Face token not configured. "
            "Please add HF_TOKEN to your .env file."
        )

    numbered_code = add_line_numbers(
        code
    )

    prompt = f"""
You are an expert software debugging assistant.

Analyze the following source code.

Filename:
{filename}

Code:
{numbered_code}

Provide the response in exactly these sections:

Bug Location:
Problematic Code:
Bug Explanation:
Suggested Fix:
Corrected Code:

If no obvious bug exists, clearly state that.
Do not invent a bug.
"""

    try:

        response = hf_client.chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert "
                        "code debugging assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            model=HF_MODEL,

            max_tokens=3000,

            temperature=0.2
        )

        return (
            response
            .choices[0]
            .message.content
        )

    except Exception as e:

        return f"AI Fix Error: {e}"


# =========================================================
# HEALTH SCORE
# =========================================================

def calculate_health_score(
    total_files,
    source_files,
    technologies,
    dependencies,
    readme_content,
    bug_count
):

    # Code Quality - 25
    if source_files == 0:

        code_quality = 0

    else:

        source_ratio = (
            source_files
            / max(total_files, 1)
        )

        if source_ratio >= 0.70:
            code_quality = 25

        elif source_ratio >= 0.50:
            code_quality = 22

        elif source_ratio >= 0.30:
            code_quality = 18

        elif source_ratio >= 0.15:
            code_quality = 14

        else:
            code_quality = 10

    # Bug Status - 30
    if source_files == 0:

        bug_status = 0

    else:

        bug_ratio = (
            bug_count
            / source_files
        )

        if bug_count == 0:
            bug_status = 30

        elif bug_ratio <= 0.10:
            bug_status = 26

        elif bug_ratio <= 0.25:
            bug_status = 21

        elif bug_ratio <= 0.50:
            bug_status = 15

        elif bug_ratio <= 0.75:
            bug_status = 8

        else:
            bug_status = 3

    # Documentation - 20
    if readme_content:

        readme_length = len(
            readme_content.strip()
        )

        if readme_length >= 2000:
            documentation = 20

        elif readme_length >= 1000:
            documentation = 17

        elif readme_length >= 500:
            documentation = 14

        elif readme_length >= 200:
            documentation = 10

        else:
            documentation = 7

    else:

        documentation = 0

    # Dependencies - 15
    if dependencies:

        dependency_count = len(
            dependencies
        )

        if dependency_count <= 20:
            dependency_score = 15

        elif dependency_count <= 40:
            dependency_score = 13

        elif dependency_count <= 80:
            dependency_score = 10

        else:
            dependency_score = 7

    else:

        dependency_score = 5

    # Project Structure - 10
    structure_score = 0

    if total_files > 0:
        structure_score += 3

    if source_files > 0:
        structure_score += 3

    if technologies:
        structure_score += 2

    if dependencies:
        structure_score += 1

    if readme_content:
        structure_score += 1

    total_score = (
        code_quality
        + bug_status
        + documentation
        + dependency_score
        + structure_score
    )

    total_score = max(
        0,
        min(100, total_score)
    )

    return {

        "overall": total_score,

        "code_quality": code_quality,

        "bug_status": bug_status,

        "documentation": documentation,

        "dependencies": dependency_score,

        "structure": structure_score
    }


def get_health_status(score):

    if score >= 80:

        return "Healthy", "🟢"

    elif score >= 60:

        return "Moderate", "🟡"

    elif score >= 40:

        return "Needs Improvement", "🟠"

    else:

        return "Poor", "🔴"


# =========================================================
# REPOSITORY ANALYSIS
# =========================================================

def analyze_repository(
    repo_url,
    run_bug_detection=False
):

    repo_path = clone_repository(
        repo_url
    )

    all_files = get_repository_files(
        repo_path
    )

    code_files = get_code_files(
        repo_path
    )

    technologies = detect_tech_stack(
        repo_path
    )

    dependencies = detect_dependencies(
        repo_path
    )

    readme_content = read_readme(
        repo_path
    )

    results = []

    codes = []

    filenames = []

    if (
        run_bug_detection
        and code_files
    ):

        for filename in code_files:

            try:

                code = read_code_file(
                    repo_path,
                    filename
                )

                if code:

                    codes.append(code)

                    filenames.append(
                        filename
                    )

            except Exception:

                pass

        if codes:

            (
                tokenizer,
                codebert,
                tfidf,
                selector,
                language_map,
                xgb_model
            ) = load_models()

            results = predict_batch(
                codes,
                filenames,
                tokenizer,
                codebert,
                tfidf,
                selector,
                language_map,
                xgb_model
            )

    return {

        "repo_path": repo_path,

        "all_files": all_files,

        "code_files": code_files,

        "technologies": technologies,

        "dependencies": dependencies,

        "readme_content": readme_content,

        "results": results,

        "codes": codes,

        "filenames": filenames
    }


# =========================================================
# REPOSITORY COMPARISON FUNCTION
# =========================================================

def compare_repositories(
    repo_a,
    repo_b
):

    bugs_a = sum(
        1
        for result in repo_a["results"]
        if result["label"] == "Bug"
    )

    bugs_b = sum(
        1
        for result in repo_b["results"]
        if result["label"] == "Bug"
    )

    health_a = calculate_health_score(
        total_files=len(
            repo_a["all_files"]
        ),
        source_files=len(
            repo_a["code_files"]
        ),
        technologies=repo_a[
            "technologies"
        ],
        dependencies=repo_a[
            "dependencies"
        ],
        readme_content=repo_a[
            "readme_content"
        ],
        bug_count=bugs_a
    )

    health_b = calculate_health_score(
        total_files=len(
            repo_b["all_files"]
        ),
        source_files=len(
            repo_b["code_files"]
        ),
        technologies=repo_b[
            "technologies"
        ],
        dependencies=repo_b[
            "dependencies"
        ],
        readme_content=repo_b[
            "readme_content"
        ],
        bug_count=bugs_b
    )

    return {

        "bugs_a": bugs_a,

        "bugs_b": bugs_b,

        "health_a": health_a,

        "health_b": health_b,

        "total_files_a": len(
            repo_a["all_files"]
        ),

        "total_files_b": len(
            repo_b["all_files"]
        ),

        "source_files_a": len(
            repo_a["code_files"]
        ),

        "source_files_b": len(
            repo_b["code_files"]
        )
    }


# =========================================================
# PROFESSIONAL DASHBOARD UI
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: #07111f;
        color: #e8eef7;
    }
    [data-testid="stSidebar"] {
        background: #0b1728;
        border-right: 1px solid #1d3048;
    }
    [data-testid="stSidebar"] * {
        color: #dbe7f5;
    }
    .hero {
        padding: 20px 24px;
        border: 1px solid #203650;
        border-radius: 18px;
        background: linear-gradient(135deg, #0d2037 0%, #0a1627 100%);
        margin-bottom: 24px;
    }
    .hero h1 { margin: 0 0 8px 0; font-size: 34px; }
    .hero p { margin: 0; color: #9fb2c8; font-size: 16px; }
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin: 8px 0 18px 0;
    }
    .card {
        background: #0d1b2d;
        border: 1px solid #1f3650;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .metric-card {
        background: #0d1b2d;
        border: 1px solid #1f3650;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
    }
    .metric-label { color: #91a6bc; font-size: 13px; }
    .metric-value { color: #f2f7fc; font-size: 27px; font-weight: 700; margin-top: 5px; }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: #142b43;
        color: #a9d1ff;
    }
    .bug-card {
        background: #24151a;
        border: 1px solid #63313b;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .ok-card {
        background: #10241d;
        border: 1px solid #285441;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .score {
        font-size: 48px;
        font-weight: 800;
        line-height: 1;
    }
    .muted { color: #91a6bc; }
    .small-note { color: #7f94aa; font-size: 12px; }
    div[data-testid="stMetric"] {
        background: #0d1b2d;
        border: 1px solid #1f3650;
        padding: 12px;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] {
        display: none;
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0 18px 0;
        margin-bottom: 8px;
    }
    .topbar-brand {
        font-size: 20px;
        font-weight: 800;
        color: #ffffff;
    }
    .topbar-brand span {
        color: #4ea1ff;
    }
    .topbar-subtitle {
        color: #7188a2;
        font-size: 13px;
    }
    .bottom-nav-title {
        margin-top: 28px;
        margin-bottom: 10px;
        text-align: center;
        color: #91a6bc;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .step-label {
        text-align: center;
        color: #667f99;
        font-size: 12px;
        margin-top: 8px;
    }

    .home-hero {
        padding: 58px 48px;
        border: 1px solid #1e4b76;
        border-radius: 20px;
        background: radial-gradient(circle at 82% 28%, rgba(76, 64, 202, .32), transparent 35%), linear-gradient(135deg, #0a1b31 0%, #0a1730 55%, #171a42 100%);
        margin-bottom: 34px;
    }
    .home-hero .eyebrow {
        color: #70b5ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        margin-bottom: 15px;
    }
    .home-hero h1 {
        margin: 0;
        font-size: 48px;
        line-height: 1.08;
        font-weight: 850;
    }
    .blue-text { color: #3d9bff; }
    .purple-text { color: #b86bff; }
    .home-hero p {
        max-width: 720px;
        color: #9cc4f4;
        font-size: 15px;
        line-height: 1.8;
        margin: 24px 0 0 0;
    }
    .start-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 14px;
    }
    div[data-testid="stTextInput"] input {
        background: #20212a;
        border: 1px solid #2a405d;
        color: #e8eef7;
        border-radius: 9px;
    }
    .footer {
        text-align: center;
        color: #60768e;
        font-size: 12px;
        margin-top: 28px;
        padding-top: 16px;
        border-top: 1px solid #17304a;
    }
    div.stButton > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# STATE HELPERS
# =========================================================

def reset_analysis_state():
    st.session_state.repo_path = None
    st.session_state.all_files = []
    st.session_state.code_files = []
    st.session_state.technologies = []
    st.session_state.dependencies = []
    st.session_state.readme_content = ""
    st.session_state.results = []
    st.session_state.codes = []
    st.session_state.filenames = []
    st.session_state.ai_fixes = {}
    st.session_state.health = None


def run_main_analysis(url):
    reset_analysis_state()
    with st.spinner("Cloning and analyzing repository..."):
        analysis = analyze_repository(url, run_bug_detection=False)

    st.session_state.repo_path = analysis["repo_path"]
    st.session_state.all_files = analysis["all_files"]
    st.session_state.code_files = analysis["code_files"]
    st.session_state.technologies = analysis["technologies"]
    st.session_state.dependencies = analysis["dependencies"]
    st.session_state.readme_content = analysis["readme_content"]
    st.session_state.results = []
    st.session_state.codes = []
    st.session_state.filenames = []
    st.session_state.ai_fixes = {}
    st.session_state.health = None
    st.session_state.page = "Overview"


# =========================================================
# TOP NAVIGATION
# =========================================================

# The app uses a website-style page flow instead of a permanent sidebar.
# Home stays available from every page.

st.markdown(
    """
    <div class="topbar">
        <div class="topbar-brand">◉ RepoLens <span>AI</span></div>
        <div class="topbar-subtitle">AI-Powered GitHub Repository Analyzer</div>
    </div>
    """,
    unsafe_allow_html=True
)

nav_col1, nav_col2, nav_col3 = st.columns([7, 1, 1])
with nav_col2:
    if st.button("⌂ Home", key="top_home", use_container_width=True):
        go_to("Home")
with nav_col3:
    if st.session_state.repo_path:
        if st.button("↻ Reset", key="top_reset", use_container_width=True):
            reset_analysis_state()
            st.session_state.page = "Home"
            st.rerun()


# =========================================================
# COMMON DATA
# =========================================================

repo_loaded = bool(st.session_state.repo_path)
all_files = st.session_state.all_files
code_files = st.session_state.code_files
technologies = st.session_state.technologies
dependencies = st.session_state.dependencies
readme_content = st.session_state.readme_content
results = st.session_state.results


def page_header(title, subtitle):
    st.markdown(
        f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True
    )


def metric_card(label, value):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True
    )


def go_to(page):
    st.session_state.page = page
    st.rerun()


PAGE_FLOW = [
    "Home",
    "Overview",
    "README",
    "Source Code",
    "🐛 Bug Detection",
    "🤖 AI Fix & Suggestions",
    "❤️ Health Score",
    "⚖️ Comparison",
]


def bottom_navigation(current_page):
    if current_page not in PAGE_FLOW or current_page == "Home":
        return

    current_index = PAGE_FLOW.index(current_page)

    st.markdown(
        "<div class='bottom-nav-title'>Repository Analysis Workflow</div>",
        unsafe_allow_html=True
    )

    cols = st.columns([1, 1, 1])

    with cols[0]:
        if current_index > 0:
            if st.button("← Previous", key=f"prev_{current_index}", use_container_width=True):
                go_to(PAGE_FLOW[current_index - 1])

    with cols[1]:
        if st.button("⌂ Home", key=f"bottom_home_{current_index}", use_container_width=True):
            go_to("Home")

    with cols[2]:
        if current_index < len(PAGE_FLOW) - 1:
            if st.button("Next →", key=f"next_{current_index}", use_container_width=True):
                go_to(PAGE_FLOW[current_index + 1])

    st.markdown(
        f"<div class='step-label'>Step {current_index} of {len(PAGE_FLOW)-1} · {current_page}</div>",
        unsafe_allow_html=True
    )

# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":
    st.markdown(
        """
        <div class="home-hero">
            <div class="eyebrow">AI-POWERED DEVELOPER TOOL</div>
            <h1>AI-Powered<br>
                <span class="blue-text">GitHub Repository</span>
                <span class="purple-text"> Analyzer</span>
            </h1>
            <p>Get detailed insights about any public GitHub repository. Find bugs, get AI-powered fixes, explore repository health and compare repositories.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='start-title'>Start Analysis</div>", unsafe_allow_html=True)

    repo_url = st.text_input(
        "GitHub Repository URL",
        value=st.session_state.main_repo_url,
        placeholder="https://github.com/owner/repository",
        key="home_repo_url"
    )
    st.session_state.main_repo_url = repo_url

    if st.button(
        "Analyze Repository →",
        type="primary",
        use_container_width=True,
        key="home_analyze"
    ):
        if not repo_url.strip():
            st.warning("Please enter a GitHub repository URL.")
        else:
            try:
                run_main_analysis(repo_url.strip())
                st.success("Repository cloned and analyzed successfully! ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Repository analysis failed: {e}")

    if repo_loaded:
        st.markdown("### Current Repository")
        cols = st.columns(4)
        with cols[0]:
            metric_card("Total Files", len(all_files))
        with cols[1]:
            metric_card("Source Files", len(code_files))
        with cols[2]:
            metric_card("Technologies", len(technologies))
        with cols[3]:
            metric_card("Dependencies", len(dependencies))

# =========================================================
# OVERVIEW
# =========================================================

elif st.session_state.page == "Overview":
    if not repo_loaded:
        page_header("📂 Repository Overview", "Analyze a repository from Home to view its overview.")
        st.info("No repository is loaded yet.")
        if st.button("Go to Home", type="primary"):
            go_to("Home")
    else:
        page_header("📊 Repository Overview", "A quick technical summary of the analyzed GitHub repository.")

        cols = st.columns(4)
        with cols[0]: metric_card("Total Files", len(all_files))
        with cols[1]: metric_card("Source Files", len(code_files))
        with cols[2]: metric_card("Technologies", len(technologies))
        with cols[3]: metric_card("Dependencies", len(dependencies))

        st.markdown("### Technology & Dependency Stack")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'><h3>🛠️ Technologies</h3>", unsafe_allow_html=True)
            if technologies:
                for tech in technologies:
                    st.markdown(f"<span class='badge'>{tech}</span>", unsafe_allow_html=True)
            else:
                st.info("No technologies detected.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'><h3>📦 Dependencies</h3>", unsafe_allow_html=True)
            if dependencies:
                for dependency in dependencies:
                    st.markdown(f"• {dependency}")
            else:
                st.info("No dependency files detected.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Analysis Workflow")
        st.write("Repository → README → Source Code → 🐛 Bug Detection → 🤖 AI Fix → ❤️ Health Score → ⚖️ Comparison")


# =========================================================
# README
# =========================================================

elif st.session_state.page == "README":
    if not repo_loaded:
        page_header("📘 README", "Readme information becomes available after repository analysis.")
        st.info("No repository is loaded yet.")
    else:
        page_header("📘 README Summary", "A concise view of the repository documentation.")
        if readme_content:
            readme_lines = readme_content.splitlines()
            title = ""
            for line in readme_lines:
                if line.strip().startswith("# "):
                    title = line.strip().replace("# ", "", 1)
                    break

            description = ""
            for line in readme_lines:
                line = line.strip()
                if (line and not line.startswith("#") and not line.startswith("!")
                        and not line.startswith("[") and len(line) > 30):
                    description = line
                    break

            if title:
                st.markdown(f"### {title}")
            if description:
                st.markdown(f"<div class='card'>{description}</div>", unsafe_allow_html=True)

            summary_points = []
            for line in readme_lines:
                line = line.strip()
                if not line:
                    continue
                lower_line = line.lower()
                if any(keyword in lower_line for keyword in [
                    "requirements", "python", "dependencies", "installation",
                    "docker", "api", "technology", "framework"
                ]):
                    clean_line = re.sub(r"^#+\s*", "", line)
                    clean_line = re.sub(r"^[-*]\s*", "", clean_line)
                    if clean_line not in summary_points:
                        summary_points.append(clean_line)

            if summary_points:
                st.markdown("### Important Information")
                for point in summary_points[:8]:
                    st.write(f"• {point}")

            with st.expander("📄 View Full README"):
                st.text(readme_content[:15000])
        else:
            st.info("README not found.")


# =========================================================
# SOURCE CODE
# =========================================================

elif st.session_state.page == "Source Code":
    if not repo_loaded:
        page_header("💻 Source Code Explorer", "Browse repository source files after analysis.")
        st.info("No repository is loaded yet.")
    else:
        page_header("💻 Source Code Explorer", "Inspect the source files collected from the repository.")
        if code_files:
            selected_file = st.selectbox("Select source file", code_files, key="source_file_select")
            try:
                selected_code = read_code_file(st.session_state.repo_path, selected_file)
                st.caption(f"File: {selected_file}")
                language = detect_language_from_file(selected_file) or "text"
                language_map_ui = {"Python": "python", "JavaScript": "javascript", "Java": "java",
                                   "C++": "cpp", "C": "c", "C#": "csharp", "PHP": "php",
                                   "Ruby": "ruby", "Go": "go"}
                st.code(selected_code, language=language_map_ui.get(language, "text"))
            except Exception as e:
                st.error(f"Unable to read file: {e}")
        else:
            st.info("No supported source code files found.")


# =========================================================
# BUG DETECTION
# =========================================================

elif st.session_state.page == "🐛 Bug Detection":
    if not repo_loaded:
        page_header("🐛 AI Bug Detection", "Run the trained CodeBERT + TF-IDF + structural feature + XGBoost pipeline.")
        st.info("Analyze a repository first from Home.")
    else:
        page_header("🐛 AI Bug Detection", "Detect potentially buggy source files using the trained hybrid model.")

        if st.button("🔎 Run AI Bug Detection", type="primary", use_container_width=True):
            try:
                with st.spinner("Running CodeBERT + TF-IDF + structural features + XGBoost..."):
                    analysis = analyze_repository(st.session_state.get("main_repo_url", ""), run_bug_detection=True)
                st.session_state.results = analysis["results"]
                st.session_state.codes = analysis["codes"]
                st.session_state.filenames = analysis["filenames"]
                st.session_state.ai_fixes = {}
                st.success("Bug detection completed! ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Bug detection failed: {e}")

        if results:
            bug_count = sum(1 for result in results if result["label"] == "Bug")
            analyzed_count = len(results)
            no_bug_count = analyzed_count - bug_count

            cols = st.columns(3)
            with cols[0]: metric_card("Files Analyzed", analyzed_count)
            with cols[1]: metric_card("Bugs Detected", bug_count)
            with cols[2]: metric_card("No Bug", no_bug_count)

            st.markdown("### File-level Results")
            for result in results:
                filename = result["filename"]
                label = result["label"]
                confidence = result.get("confidence", 0)
                if label == "Bug":
                    st.markdown(
                        f"<div class='bug-card'><b>🐛 {filename}</b><br>"
                        f"Potential bug detected · Confidence: {confidence:.1f}%</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='ok-card'><b>✅ {filename}</b><br>"
                        f"No bug detected · Confidence: {confidence:.1f}%</div>",
                        unsafe_allow_html=True
                    )

            st.info("Next step: open **🤖 AI Fix & Suggestions** to inspect detected bugs and generate fixes.")
            if st.button("🤖 Open AI Fix & Suggestions", type="primary"):
                go_to("🤖 AI Fix & Suggestions")
        else:
            st.info("No bug detection results yet. Click the button above to run the model.")


# =========================================================
# AI FIX & SUGGESTIONS
# =========================================================

elif st.session_state.page == "🤖 AI Fix & Suggestions":
    if not repo_loaded:
        page_header("🤖 AI Fix & Suggestions", "Generate explanations and corrected code for detected bugs.")
        st.info("Analyze a repository first from Home.")
    else:
        page_header("🤖 AI Fix & Suggestions", "Use the configured Hugging Face coding model to explain and suggest fixes for detected bugs.")

        if not results:
            st.warning("Run AI Bug Detection first.")
            if st.button("🐛 Go to Bug Detection", type="primary"):
                go_to("🐛 Bug Detection")
        else:
            bug_results = [r for r in results if r["label"] == "Bug"]
            if not bug_results:
                st.success("No files were classified as buggy. No AI fix is required.")
            else:
                st.markdown(f"### {len(bug_results)} detected bug(s) available for AI analysis")
                for index, result in enumerate(bug_results):
                    filename = result["filename"]
                    confidence = result.get("confidence", 0)
                    st.markdown(
                        f"<div class='bug-card'><b>🐛 {filename}</b><br>"
                        f"Model confidence: {confidence:.1f}%</div>",
                        unsafe_allow_html=True
                    )

                    if st.button("🔧 Find Bug Line & Generate Fix", key=f"ai_fix_{index}", use_container_width=True):
                        try:
                            code_index = st.session_state.filenames.index(filename)
                            code = st.session_state.codes[code_index]
                            with st.spinner("AI is analyzing the code and generating a fix..."):
                                fix = generate_ai_fix(code, filename)
                            st.session_state.ai_fixes[filename] = fix
                        except Exception as e:
                            st.error(f"AI Fix failed: {e}")

                    if filename in st.session_state.ai_fixes:
                        with st.expander(f"📋 AI Analysis — {filename}", expanded=True):
                            st.markdown(st.session_state.ai_fixes[filename])


# =========================================================
# HEALTH SCORE
# =========================================================

elif st.session_state.page == "❤️ Health Score":
    if not repo_loaded:
        page_header("❤️ Repository Health Score", "Rule-based health assessment of the analyzed repository.")
        st.info("Analyze a repository first from Home.")
    else:
        page_header("❤️ Repository Health Score", "A rule-based weighted score based on repository quality indicators.")

        bug_count_for_health = sum(1 for result in results if result["label"] == "Bug")
        health = calculate_health_score(
            total_files=len(all_files),
            source_files=len(code_files),
            technologies=technologies,
            dependencies=dependencies,
            readme_content=readme_content,
            bug_count=bug_count_for_health
        )
        st.session_state.health = health
        overall_score = health["overall"]
        status, status_icon = get_health_status(overall_score)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='score'>{overall_score}/100</div>", unsafe_allow_html=True)
            st.markdown(f"### {status_icon} {status}")
            st.markdown("<div class='muted'>Overall repository health</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("### Score Components")
            components = [
                ("Code Quality", health["code_quality"], 25),
                ("Bug Status", health["bug_status"], 30),
                ("Documentation", health["documentation"], 20),
                ("Dependencies", health["dependencies"], 15),
                ("Project Structure", health["structure"], 10),
            ]
            for name, value, maximum in components:
                st.write(f"**{name}** — {value}/{maximum}")
                st.progress(value / maximum if maximum else 0)

        with st.expander("ℹ️ How is the Health Score calculated?"):
            st.write(
                "The Repository Health Score is a rule-based indicator calculated from "
                "Code Quality (25), Bug Status (30), Documentation (20), Dependencies (15), "
                "and Project Structure (10). Total = 100 points. It is an analytical indicator, "
                "not model accuracy and not a probability of being bug-free."
            )


# =========================================================
# COMPARISON
# =========================================================

elif st.session_state.page == "⚖️ Comparison":
    page_header("⚖️ Repository Comparison", "Compare two public GitHub repositories using common structural and quality metrics.")

    col1, col2 = st.columns(2)
    with col1:
        comparison_url_a = st.text_input(
            "Repository A URL",
            placeholder="https://github.com/user/repository-a",
            key="comparison_url_a"
        )
    with col2:
        comparison_url_b = st.text_input(
            "Repository B URL",
            placeholder="https://github.com/user/repository-b",
            key="comparison_url_b"
        )

    if st.button("⚖️ Compare Repositories", type="primary", use_container_width=True):
        if not comparison_url_a.strip() or not comparison_url_b.strip():
            st.warning("Please enter both repository URLs.")
        else:
            try:
                with st.spinner("Analyzing both repositories with the bug detection pipeline..."):
                    repo_a = analyze_repository(comparison_url_a.strip(), run_bug_detection=True)
                    repo_b = analyze_repository(comparison_url_b.strip(), run_bug_detection=True)
                    comparison = compare_repositories(repo_a, repo_b)
                    st.session_state.comparison_a = repo_a
                    st.session_state.comparison_b = repo_b
                    st.session_state.comparison_results = comparison
                st.success("Repository comparison completed! ✅")
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    if st.session_state.comparison_results:
        comparison = st.session_state.comparison_results
        repo_a = st.session_state.comparison_a
        repo_b = st.session_state.comparison_b

        st.markdown("### Comparison Overview")
        data = {
            "Metric": ["Total Files", "Source Files", "Technologies", "Dependencies", "Bugs Detected", "Health Score"],
            "Repository A": [
                comparison["total_files_a"], comparison["source_files_a"], len(repo_a["technologies"]),
                len(repo_a["dependencies"]), comparison["bugs_a"], f"{comparison['health_a']['overall']}/100"
            ],
            "Repository B": [
                comparison["total_files_b"], comparison["source_files_b"], len(repo_b["technologies"]),
                len(repo_b["dependencies"]), comparison["bugs_b"], f"{comparison['health_b']['overall']}/100"
            ]
        }
        st.table(data)

        st.markdown("### 🛠️ Technology Comparison")
        tech1, tech2 = st.columns(2)
        with tech1:
            st.markdown("**Repository A**")
            if repo_a["technologies"]:
                for tech in repo_a["technologies"]: st.write(f"• {tech}")
            else: st.info("No technologies detected.")
        with tech2:
            st.markdown("**Repository B**")
            if repo_b["technologies"]:
                for tech in repo_b["technologies"]: st.write(f"• {tech}")
            else: st.info("No technologies detected.")

        st.markdown("### 📦 Dependency Comparison")
        dep1, dep2 = st.columns(2)
        with dep1:
            st.markdown("**Repository A**")
            if repo_a["dependencies"]:
                for dependency in repo_a["dependencies"]: st.write(f"• {dependency}")
            else: st.info("No dependencies detected.")
        with dep2:
            st.markdown("**Repository B**")
            if repo_b["dependencies"]:
                for dependency in repo_b["dependencies"]: st.write(f"• {dependency}")
            else: st.info("No dependencies detected.")

        st.markdown("### ❤️ Health Score Breakdown")
        health_data = {
            "Component": ["Code Quality", "Bug Status", "Documentation", "Dependencies", "Project Structure", "Overall"],
            "Repository A": [
                f"{comparison['health_a']['code_quality']}/25", f"{comparison['health_a']['bug_status']}/30",
                f"{comparison['health_a']['documentation']}/20", f"{comparison['health_a']['dependencies']}/15",
                f"{comparison['health_a']['structure']}/10", f"{comparison['health_a']['overall']}/100"
            ],
            "Repository B": [
                f"{comparison['health_b']['code_quality']}/25", f"{comparison['health_b']['bug_status']}/30",
                f"{comparison['health_b']['documentation']}/20", f"{comparison['health_b']['dependencies']}/15",
                f"{comparison['health_b']['structure']}/10", f"{comparison['health_b']['overall']}/100"
            ]
        }
        st.table(health_data)
        st.info("The comparison presents repository metrics without automatically declaring one repository better than the other.")


# =========================================================
# BOTTOM PAGE NAVIGATION
# =========================================================

bottom_navigation(st.session_state.page)

st.markdown(
    "<div class='footer'>RepoLens AI · AI-Powered GitHub Repository Analyzer</div>",
    unsafe_allow_html=True
)
