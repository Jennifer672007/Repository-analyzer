import os
import re
import joblib
import numpy as np
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModel
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

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-Coder-32B-Instruct"
)

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
    "comparison_results": None
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔍 RepoLens AI")

    st.caption(
        "AI-Powered GitHub Repository Analyzer"
    )

    st.divider()

    st.write(
        "Analyze source code, "
        "detect bugs and compare repositories."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🔍 AI-Powered GitHub Repository Analyzer"
)

st.write(
    "Analyze a public GitHub repository "
    "using AI-powered code analysis."
)


# =========================================================
# REPOSITORY ANALYZER
# =========================================================

st.subheader(
    "📂 Repository Analysis"
)

repo_url = st.text_input(
    "Enter GitHub Repository URL",
    placeholder=(
        "https://github.com/username/repository"
    )
)

analyze_button = st.button(
    "🚀 Analyze Repository",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not repo_url.strip():

        st.warning(
            "Please enter a GitHub repository URL."
        )

    else:

        # Clear previous data
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

        try:

            with st.spinner(
                "Cloning and analyzing repository..."
            ):

                analysis = analyze_repository(
                    repo_url,
                    run_bug_detection=False
                )

            st.session_state.repo_path = (
                analysis["repo_path"]
            )

            st.session_state.all_files = (
                analysis["all_files"]
            )

            st.session_state.code_files = (
                analysis["code_files"]
            )

            st.session_state.technologies = (
                analysis["technologies"]
            )

            st.session_state.dependencies = (
                analysis["dependencies"]
            )

            st.session_state.readme_content = (
                analysis["readme_content"]
            )

            st.session_state.results = []

            st.success(
                "Repository cloned successfully! ✅"
            )

        except Exception as e:

            st.error(
                f"Repository analysis failed: {e}"
            )


# =========================================================
# DISPLAY ANALYSIS
# =========================================================

if st.session_state.repo_path:

    all_files = (
        st.session_state.all_files
    )

    code_files = (
        st.session_state.code_files
    )

    technologies = (
        st.session_state.technologies
    )

    dependencies = (
        st.session_state.dependencies
    )

    readme_content = (
        st.session_state.readme_content
    )

    # =====================================================
    # REPOSITORY OVERVIEW
    # =====================================================

    st.subheader(
        "📊 Repository Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Files",
            len(all_files)
        )

    with col2:

        st.metric(
            "Source Files",
            len(code_files)
        )

    with col3:

        st.metric(
            "Technologies",
            len(technologies)
        )

    with col4:

        st.metric(
            "Dependencies",
            len(dependencies)
        )

    # =====================================================
    # TECHNOLOGIES + DEPENDENCIES
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🛠️ Technologies"
        )

        if technologies:

            for tech in technologies:

                st.write(
                    f"• {tech}"
                )

        else:

            st.info(
                "No technologies detected."
            )

    with col2:

        st.subheader(
            "📦 Dependencies"
        )

        if dependencies:

            for dependency in dependencies:

                st.write(
                    f"• {dependency}"
                )

        else:

            st.info(
                "No dependency files detected."
            )

    # =====================================================
    # README
    # =====================================================

    st.subheader(
        "📘 README Summary"
    )

    if readme_content:

        with st.expander(
            "View README"
        ):

            st.text(
                readme_content[:5000]
            )

    else:

        st.info(
            "README not found."
        )

    # =====================================================
    # SOURCE CODE
    # =====================================================

    st.subheader(
        "💻 Source Code Explorer"
    )

    if code_files:

        selected_file = st.selectbox(
            "Select source file",
            code_files
        )

        try:

            selected_code = read_code_file(
                st.session_state.repo_path,
                selected_file
            )

            st.code(
                selected_code,
                language="text"
            )

        except Exception as e:

            st.error(
                f"Unable to read file: {e}"
            )

    else:

        st.info(
            "No supported source code files found."
        )

    # =====================================================
    # AI BUG DETECTION
    # =====================================================

    st.subheader(
        "🐛 AI Bug Detection"
    )

    if code_files:

        if st.button(
            "🔎 Run AI Bug Detection",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Running CodeBERT + TF-IDF + XGBoost..."
                ):

                    analysis = analyze_repository(
                        repo_url,
                        run_bug_detection=True
                    )

                st.session_state.results = (
                    analysis["results"]
                )

                st.session_state.codes = (
                    analysis["codes"]
                )

                st.session_state.filenames = (
                    analysis["filenames"]
                )

                st.session_state.ai_fixes = {}

                st.success(
                    "Bug detection completed! ✅"
                )

            except Exception as e:

                st.error(
                    f"Bug detection failed: {e}"
                )

    # =====================================================
    # BUG RESULTS
    # =====================================================

    if st.session_state.results:

        results = (
            st.session_state.results
        )

        bug_count = sum(
            1
            for result in results
            if result["label"] == "Bug"
        )

        analyzed_count = len(
            results
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Files Analyzed",
                analyzed_count
            )

        with col2:

            st.metric(
                "Bugs Detected",
                bug_count
            )

        st.divider()

        for index, result in enumerate(
            results
        ):

            filename = result[
                "filename"
            ]

            label = result[
                "label"
            ]

            if label == "Bug":

                st.error(
                    f"🐛 {filename} — Bug detected"
                )

                if st.button(
                    "🔧 Find Bug Line & Generate Fix",
                    key=f"fix_{index}"
                ):

                    try:

                        code_index = (
                            st.session_state
                            .filenames
                            .index(filename)
                        )

                        code = (
                            st.session_state
                            .codes[code_index]
                        )

                        with st.spinner(
                            "AI is analyzing the code..."
                        ):

                            fix = generate_ai_fix(
                                code,
                                filename
                            )

                        st.session_state.ai_fixes[
                            filename
                        ] = fix

                    except Exception as e:

                        st.error(
                            f"AI Fix failed: {e}"
                        )

                if filename in (
                    st.session_state.ai_fixes
                ):

                    with st.expander(
                        f"🤖 AI Fix & Suggestion — {filename}",
                        expanded=True
                    ):

                        st.markdown(
                            st.session_state
                            .ai_fixes[filename]
                        )

            else:

                st.success(
                    f"✅ {filename} — No bug detected"
                )

    # =====================================================
    # HEALTH SCORE
    # =====================================================

    st.divider()

    st.subheader(
        "❤️ Repository Health Score"
    )

    bug_count_for_health = sum(
        1
        for result in st.session_state.results
        if result["label"] == "Bug"
    )

    health = calculate_health_score(

        total_files=len(all_files),

        source_files=len(code_files),

        technologies=technologies,

        dependencies=dependencies,

        readme_content=readme_content,

        bug_count=bug_count_for_health
    )

    st.session_state.health = health

    overall_score = health[
        "overall"
    ]

    status, status_icon = (
        get_health_status(
            overall_score
        )
    )

    st.markdown(
        f"""
        ### {status_icon} Overall Health: **{overall_score}/100**

        **Status:** {status}
        """
    )

    st.progress(
        overall_score / 100
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Code Quality",
            f"{health['code_quality']}/25"
        )

    with col2:

        st.metric(
            "Bug Status",
            f"{health['bug_status']}/30"
        )

    with col3:

        st.metric(
            "Documentation",
            f"{health['documentation']}/20"
        )

    with col4:

        st.metric(
            "Dependencies",
            f"{health['dependencies']}/15"
        )

    with col5:

        st.metric(
            "Structure",
            f"{health['structure']}/10"
        )

    with st.expander(
        "ℹ️ How is the Health Score calculated?"
    ):

        st.write(
            """
            The Repository Health Score is a
            rule-based indicator calculated from:

            • Code Quality — 25 points
            • Bug Status — 30 points
            • Documentation — 20 points
            • Dependencies — 15 points
            • Project Structure — 10 points

            Total = 100 points.

            This score is an analytical indicator.
            It is not model accuracy and it does not
            represent a probability of the repository
            being bug-free.
            """
        )

    # =====================================================
    # REPOSITORY COMPARISON - LAST
    # =====================================================

    st.divider()

    st.header(
        "🔄 Repository Comparison"
    )

    st.write(
        "Compare two public GitHub repositories "
        "based on repository structure, technologies, "
        "dependencies, detected bugs and health score."
    )

    comparison_col1, comparison_col2 = (
        st.columns(2)
    )

    with comparison_col1:

        st.subheader(
            "📁 Repository A"
        )

        comparison_url_a = st.text_input(
            "Repository A URL",
            placeholder=(
                "https://github.com/user/repository-a"
            ),
            key="comparison_url_a"
        )

    with comparison_col2:

        st.subheader(
            "📁 Repository B"
        )

        comparison_url_b = st.text_input(
            "Repository B URL",
            placeholder=(
                "https://github.com/user/repository-b"
            ),
            key="comparison_url_b"
        )

    compare_button = st.button(
        "⚖️ Compare Repositories",
        type="primary",
        use_container_width=True
    )

    if compare_button:

        if (
            not comparison_url_a.strip()
            or not comparison_url_b.strip()
        ):

            st.warning(
                "Please enter both repository URLs."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing both repositories..."
                ):

                    repo_a = analyze_repository(
                        comparison_url_a,
                        run_bug_detection=True
                    )

                    repo_b = analyze_repository(
                        comparison_url_b,
                        run_bug_detection=True
                    )

                    comparison = (
                        compare_repositories(
                            repo_a,
                            repo_b
                        )
                    )

                    st.session_state.comparison_a = (
                        repo_a
                    )

                    st.session_state.comparison_b = (
                        repo_b
                    )

                    st.session_state.comparison_results = (
                        comparison
                    )

                st.success(
                    "Repository comparison completed! ✅"
                )

            except Exception as e:

                st.error(
                    f"Comparison failed: {e}"
                )

    # =====================================================
    # COMPARISON RESULTS
    # =====================================================

    if st.session_state.comparison_results:

        comparison = (
            st.session_state.comparison_results
        )

        repo_a = (
            st.session_state.comparison_a
        )

        repo_b = (
            st.session_state.comparison_b
        )

        st.subheader(
            "📊 Comparison Results"
        )

        comparison_data = {

            "Metric": [

                "Total Files",

                "Source Files",

                "Technologies",

                "Dependencies",

                "Bugs Detected",

                "Health Score"
            ],

            "Repository A": [

                comparison[
                    "total_files_a"
                ],

                comparison[
                    "source_files_a"
                ],

                len(
                    repo_a[
                        "technologies"
                    ]
                ),

                len(
                    repo_a[
                        "dependencies"
                    ]
                ),

                comparison[
                    "bugs_a"
                ],

                f"{comparison['health_a']['overall']}/100"
            ],

            "Repository B": [

                comparison[
                    "total_files_b"
                ],

                comparison[
                    "source_files_b"
                ],

                len(
                    repo_b[
                        "technologies"
                    ]
                ),

                len(
                    repo_b[
                        "dependencies"
                    ]
                ),

                comparison[
                    "bugs_b"
                ],

                f"{comparison['health_b']['overall']}/100"
            ]
        }

        st.table(
            comparison_data
        )

        # =================================================
        # TECHNOLOGY COMPARISON
        # =================================================

        st.subheader(
            "🛠️ Technology Comparison"
        )

        tech_col1, tech_col2 = (
            st.columns(2)
        )

        with tech_col1:

            st.markdown(
                "**Repository A**"
            )

            if repo_a[
                "technologies"
            ]:

                for tech in repo_a[
                    "technologies"
                ]:

                    st.write(
                        f"• {tech}"
                    )

            else:

                st.info(
                    "No technologies detected."
                )

        with tech_col2:

            st.markdown(
                "**Repository B**"
            )

            if repo_b[
                "technologies"
            ]:

                for tech in repo_b[
                    "technologies"
                ]:

                    st.write(
                        f"• {tech}"
                    )

            else:

                st.info(
                    "No technologies detected."
                )

        # =================================================
        # DEPENDENCY COMPARISON
        # =================================================

        st.subheader(
            "📦 Dependency Comparison"
        )

        dep_col1, dep_col2 = (
            st.columns(2)
        )

        with dep_col1:

            st.markdown(
                "**Repository A**"
            )

            if repo_a[
                "dependencies"
            ]:

                for dependency in repo_a[
                    "dependencies"
                ]:

                    st.write(
                        f"• {dependency}"
                    )

            else:

                st.info(
                    "No dependencies detected."
                )

        with dep_col2:

            st.markdown(
                "**Repository B**"
            )

            if repo_b[
                "dependencies"
            ]:

                for dependency in repo_b[
                    "dependencies"
                ]:

                    st.write(
                        f"• {dependency}"
                    )

            else:

                st.info(
                    "No dependencies detected."
                )

        # =================================================
        # HEALTH COMPARISON
        # =================================================

        st.subheader(
            "❤️ Health Score Breakdown"
        )

        health_comparison = {

            "Component": [

                "Code Quality",

                "Bug Status",

                "Documentation",

                "Dependencies",

                "Project Structure",

                "Overall"
            ],

            "Repository A": [

                f"{comparison['health_a']['code_quality']}/25",

                f"{comparison['health_a']['bug_status']}/30",

                f"{comparison['health_a']['documentation']}/20",

                f"{comparison['health_a']['dependencies']}/15",

                f"{comparison['health_a']['structure']}/10",

                f"{comparison['health_a']['overall']}/100"
            ],

            "Repository B": [

                f"{comparison['health_b']['code_quality']}/25",

                f"{comparison['health_b']['bug_status']}/30",

                f"{comparison['health_b']['documentation']}/20",

                f"{comparison['health_b']['dependencies']}/15",

                f"{comparison['health_b']['structure']}/10",

                f"{comparison['health_b']['overall']}/100"
            ]
        }

        st.table(
            health_comparison
        )

        st.info(
            "The comparison presents factual "
            "repository metrics without automatically "
            "declaring one repository better than the other."
        )