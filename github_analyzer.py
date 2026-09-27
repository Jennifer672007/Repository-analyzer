import os
import shutil
import tempfile
from git import Repo
from urllib.parse import urlparse


CODE_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".php", ".rb", ".go", ".swift", ".kt", ".kts", ".rs",
}


IGNORED_DIRECTORIES = {
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "venv312", "build", "dist"
}


def validate_github_url(url):
    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in ["http", "https"]
            and parsed.netloc.lower() == "github.com"
            and len(parsed.path.strip("/").split("/")) >= 2
        )

    except Exception:
        return False


def clone_repository(github_url):
    if not validate_github_url(github_url):
        raise ValueError("Invalid GitHub repository URL.")

    temp_dir = tempfile.mkdtemp(prefix="github_repo_")

    try:
        Repo.clone_from(
            github_url,
            temp_dir,
            depth=1
        )

        return temp_dir

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Repository cloning failed: {e}"
        )


def get_repository_files(repo_path):
    files = []

    for root, dirs, filenames in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in filenames:
            full_path = os.path.join(root, filename)

            relative_path = os.path.relpath(
                full_path,
                repo_path
            )

            files.append(relative_path)

    return files


def get_code_files(repo_path):
    code_files = []

    for root, dirs, filenames in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in filenames:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in CODE_EXTENSIONS:

                full_path = os.path.join(
                    root,
                    filename
                )

                relative_path = os.path.relpath(
                    full_path,
                    repo_path
                )

                code_files.append(relative_path)

    return code_files


def read_code_file(repo_path, relative_path):

    full_path = os.path.join(
        repo_path,
        relative_path
    )

    try:

        with open(
            full_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            return file.read()

    except Exception:
        return ""


# ---------------- README ----------------

def get_readme_file(repo_path):

    readme_names = {
        "readme.md",
        "readme.txt",
        "readme"
    }

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            if filename.lower() in readme_names:

                return os.path.join(
                    root,
                    filename
                )

    return None


def read_readme(repo_path):

    readme_path = get_readme_file(repo_path)

    if not readme_path:
        return ""

    try:

        with open(
            readme_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            return file.read()

    except Exception:
        return ""


# ---------------- DEPENDENCIES ----------------

def detect_dependencies(repo_path):

    dependency_files = {
        "requirements.txt": "Python",
        "package.json": "JavaScript / Node.js",
        "pom.xml": "Java / Maven",
        "build.gradle": "Java / Gradle",
        "composer.json": "PHP",
        "Gemfile": "Ruby",
        "go.mod": "Go",
        "Cargo.toml": "Rust",
        "pyproject.toml": "Python",
    }

    detected = []

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            if filename in dependency_files:

                detected.append(
                    f"{filename} → "
                    f"{dependency_files[filename]}"
                )

    return detected


# ---------------- TECH STACK ----------------

def detect_tech_stack(repo_path):

    extensions = set()

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension:
                extensions.add(extension)

    tech_stack = []

    if ".py" in extensions:
        tech_stack.append("Python")

    if ".js" in extensions or ".ts" in extensions:
        tech_stack.append("JavaScript / TypeScript")

    if ".java" in extensions:
        tech_stack.append("Java")

    if ".cpp" in extensions or ".cc" in extensions:
        tech_stack.append("C++")

    if ".c" in extensions:
        tech_stack.append("C")

    if ".cs" in extensions:
        tech_stack.append("C#")

    if ".php" in extensions:
        tech_stack.append("PHP")

    if ".rb" in extensions:
        tech_stack.append("Ruby")

    if ".go" in extensions:
        tech_stack.append("Go")

    if ".rs" in extensions:
        tech_stack.append("Rust")

    return tech_stack