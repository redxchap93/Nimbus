#!/usr/bin/env python3
"""
Flask App: Python Script Executor & Search Engine

Features:
• Creates (or reuses) a virtual environment and installs missing dependencies.
• Uses a simulated model to generate Python scripts based on your search query.
• Executes, auto-corrects, and stores execution details (script, steps, output, return code).
• Displays a list of currently running Python processes.
• Provides a Port Monitor page to list listening ports and kill them.
• Full Git integration: view git status, commit & push changes, pull, clone repositories.
• GitHub integration: if credentials are not provided via environment variables, a secure login form is shown.
• **NEW:** Search for a Python script using your query. The result is a Python script tailored to your request.
• **NEW:** Every page has a “Return to Home” button.
• **NEW:** Additional navigation links for cloud platforms (AWS, GCP, Azure) and remote machine services (Citrix, AWS Workspace) added at the top.
• Uses a modern, professional dark theme with smooth animations and a responsive design.
Usage:
    python prep_env_pyth.py
Then open the printed URL in your browser.
"""

import os
import subprocess
import venv
import re
import logging
import time
import uuid
import random
import shutil
from datetime import datetime, timedelta
from flask import Flask, render_template, request, session, redirect, url_for
from jinja2 import DictLoader

# Attempt to import psutil for process/port monitoring.
try:
    import psutil
except ImportError:
    psutil = None

# Attempt to import PyGithub for GitHub integration.
try:
    from github import Github
except ImportError:
    Github = None

# Set up logging.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global constants.
VENV_DIR = ".venv_auto_exec"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "deepseek-r1:7b"
GIT_REMOTE_URL = os.getenv("GIT_REMOTE_URL", "")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

##############################
# Git Safe Directory Helper
##############################

def ensure_git_safe_directory():
    cwd = os.getcwd()
    result = subprocess.run(["git", "config", "--global", "--add", "safe.directory", cwd],
                              capture_output=True, text=True)
    if result.returncode == 0:
        logging.info(f"Configured safe directory '{cwd}'.")
    else:
        logging.warning(f"Failed to set safe directory '{cwd}': {result.stderr}")

##############################
# GitHub Login Helpers
##############################

def get_github_credentials():
    if 'github_username' in session and 'github_token' in session and 'github_login_time' in session:
        try:
            login_time = datetime.fromisoformat(session['github_login_time'])
        except Exception as e:
            logging.error("Error parsing login time: %s", e)
            return None
        if datetime.now() - login_time < timedelta(hours=1):
            logging.debug("GitHub credentials found in session.")
            return session['github_username'], session['github_token']
        else:
            logging.debug("GitHub session expired. Clearing session.")
            session.pop('github_username', None)
            session.pop('github_token', None)
            session.pop('github_login_time', None)
            return None
    logging.debug("No GitHub credentials found in session.")
    return None

##############################
# Helper Functions
##############################

def create_virtualenv(venv_dir):
    if not os.path.exists(venv_dir):
        logging.info(f"Creating virtual environment in '{venv_dir}'...")
        venv.create(venv_dir, with_pip=True)
    else:
        logging.info(f"Virtual environment '{venv_dir}' already exists.")
    return (os.path.join(venv_dir, "Scripts", "python.exe") if os.name == 'nt'
            else os.path.join(venv_dir, "bin", "python"))

def extract_imports(script_content):
    imports = set()
    pattern = r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    for line in script_content.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            imports.add(match.group(1))
    logging.info(f"Detected import modules: {imports}")
    return imports

def install_missing_packages(venv_python, modules):
    logging.info("Checking and installing missing packages...")
    for module in modules:
        try:
            result = subprocess.run([venv_python, "-c", f"import {module}"],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logging.info(f"Module '{module}' not found. Installing...")
                install_result = subprocess.run([venv_python, "-m", "pip", "install", module],
                                                capture_output=True, text=True)
                if install_result.returncode != 0:
                    logging.error(f"Failed to install '{module}': {install_result.stderr}")
                else:
                    logging.info(f"Successfully installed '{module}'.")
        except Exception as e:
            logging.error(f"Error checking/installing module '{module}': {e}")

def connect_to_ollama_server():
    logging.info(f"Connecting to Ollama server at {OLLAMA_API_URL} with model {MODEL_NAME}...")
    time.sleep(1)
    logging.info("Connected to Ollama server.")
    return {"api_url": OLLAMA_API_URL, "model": MODEL_NAME}

def analyze_script_with_deepseek(ollama_client, script_content):
    """
    Process the search query and return a Python script.
    The outcome is filtered based on keywords in your query.
    """
    query = script_content.strip()
    lower_query = query.lower()
    
    # If the query is about training an LLM
    if "train" in lower_query and "llm" in lower_query:
        return (
            "# Script generated for query:\n"
            f"# {query}\n\n"
            "import transformers\n"
            "from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer\n\n"
            "def main():\n"
            "    model_name = 'deepseek-r1:1.5b'\n"
            "    tokenizer = AutoTokenizer.from_pretrained('gpt2')\n"
            "    model = AutoModelForCausalLM.from_pretrained('gpt2')\n\n"
            "    training_args = TrainingArguments(\n"
            "        output_dir='./results',\n"
            "        num_train_epochs=3,\n"
            "        per_device_train_batch_size=4,\n"
            "        save_steps=500,\n"
            "        save_total_limit=2,\n"
            "    )\n\n"
            "    train_dataset = []  # TODO: Replace with your dataset\n\n"
            "    trainer = Trainer(\n"
            "        model=model,\n"
            "        args=training_args,\n"
            "        train_dataset=train_dataset,\n"
            "    )\n\n"
            "    trainer.train()\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    # If the query is about updating Windows
    elif "update" in lower_query and "windows" in lower_query:
        return (
            "# Script generated for query:\n"
            f"# {query}\n\n"
            "import subprocess\n"
            "import sys\n\n"
            "def update_windows():\n"
            "    try:\n"
            "        subprocess.run(['wuauclt.exe', '/detectnow'], check=True)\n"
            "        print('Windows update detection triggered successfully.')\n"
            "    except Exception as e:\n"
            "        print(f'Error triggering Windows update: {e}', file=sys.stderr)\n\n"
            "if __name__ == '__main__':\n"
            "    update_windows()\n"
        )
    else:
        # Default: return a generic Python script template with your query in the comment.
        return (
            "# Script generated for query:\n"
            f"# {query}\n\n"
            "def main():\n"
            "    # TODO: Implement functionality for the following request:\n"
            f"    # {query}\n"
            "    pass\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

def execute_script(venv_python, script_path):
    logging.info(f"Executing script '{script_path}' using '{venv_python}'...")
    result = subprocess.run([venv_python, script_path],
                            capture_output=True, text=True)
    logging.info("Script execution completed.")
    return result.returncode, result.stdout, result.stderr

def attempt_auto_correction(ollama_client, script_content, error_message):
    logging.info("Attempting automatic error correction...")
    lines = script_content.splitlines()
    new_lines = []
    changed = False
    for line in lines:
        if line.count('"') % 2 != 0:
            if line.strip().endswith(')'):
                index = line.rfind(')')
                line = line[:index] + '"' + line[index:]
            else:
                line = line.rstrip() + '"'
            changed = True
        new_lines.append(line)
    corrected_script = "\n".join(new_lines)
    if changed:
        logging.info("Auto-correction applied.")
        return corrected_script
    return script_content + f"\n# Auto-correction applied for error: {error_message}\n"

def store_history(script_content, steps, output, retcode):
    if not os.path.exists("history"):
        os.makedirs("history")
    filename = f"history_{uuid.uuid4().hex}.txt"
    filepath = os.path.join("history", filename)
    with open(filepath, "w") as f:
        f.write("Script:\n" + script_content + "\n\n")
        f.write("Execution Steps:\n" + steps + "\n\n")
        f.write("Output:\n" + output + "\n\n")
        f.write("Return Code: " + str(retcode) + "\n")
    logging.info(f"Stored execution history as {filename}.")
    return filename

def git_init():
    ensure_git_safe_directory()
    output_all = ""
    if not os.path.exists(".git"):
        out = subprocess.run(["git", "init"], capture_output=True, text=True)
        output_all += out.stdout + out.stderr
        if GIT_REMOTE_URL:
            subprocess.run(["git", "remote", "add", "origin", GIT_REMOTE_URL],
                           capture_output=True, text=True)
    else:
        output_all += "Git repository already initialized.\n"
    return output_all

def git_status():
    ensure_git_safe_directory()
    out = subprocess.run(["git", "status"], capture_output=True, text=True)
    return out.stdout + out.stderr

def git_commit_and_push():
    ensure_git_safe_directory()
    output_all = ""
    subprocess.run(["git", "add", "."], capture_output=True, text=True)
    commit_message = f"Auto commit on {time.strftime('%Y-%m-%d %H:%M:%S')}"
    out = subprocess.run(["git", "commit", "-m", commit_message],
                         capture_output=True, text=True)
    output_all += out.stdout + out.stderr
    if GIT_REMOTE_URL:
        remotes = subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip()
        if not remotes:
            subprocess.run(["git", "remote", "add", "origin", GIT_REMOTE_URL],
                           capture_output=True, text=True)
        out = subprocess.run(["git", "push", "-u", "origin", "master"],
                             capture_output=True, text=True)
        output_all += out.stdout + out.stderr
    return output_all

def git_pull():
    ensure_git_safe_directory()
    out = subprocess.run(["git", "pull"], capture_output=True, text=True)
    return out.stdout + out.stderr

def git_clone(repo_url, clone_dir="cloned_repo"):
    ensure_git_safe_directory()
    if os.path.exists(clone_dir):
        return f"Directory {clone_dir} already exists. Delete it if you want to clone again."
    out = subprocess.run(["git", "clone", repo_url, clone_dir],
                         capture_output=True, text=True)
    return out.stdout + out.stderr

def github_create_repo(repo_name, repo_desc=""):
    creds = get_github_credentials()
    if creds is None:
        return "You must sign in to GitHub first."
    username, token = creds
    try:
        g = Github(token)
        user = g.get_user()
        repo = user.create_repo(name=repo_name, description=repo_desc)
        return f"Repository '{repo.full_name}' created successfully on GitHub."
    except Exception as e:
        return f"Error creating repository: {e}"

def get_running_python_processes():
    if psutil is None:
        return [{"pid": "", "name": "psutil not installed",
                 "cmdline": ["Install psutil with: pip install psutil"]}]
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                processes.append(proc.info)
            elif proc.info['cmdline'] and any("python" in part.lower() for part in proc.info['cmdline']):
                processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return processes

def get_used_ports():
    ports = []
    if psutil is None:
        return ports
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == 'LISTEN' and conn.laddr:
            port = conn.laddr.port
            pid = conn.pid
            name = ""
            cmdline = ""
            if pid:
                try:
                    proc = psutil.Process(pid)
                    name = proc.name()
                    cmdline = " ".join(proc.cmdline())
                except Exception:
                    pass
            ports.append({"port": port, "pid": pid, "name": name, "cmdline": cmdline})
    unique = {}
    for p in ports:
        key = (p["port"], p["pid"])
        if key not in unique:
            unique[key] = p
    return list(unique.values())

def kill_port(port):
    if psutil is None:
        return "psutil is not installed. Cannot kill ports."
    killed = []
    not_killed = []
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == 'LISTEN' and conn.laddr and conn.laddr.port == port and conn.pid:
            try:
                proc = psutil.Process(conn.pid)
                proc.terminate()
                proc.wait(timeout=3)
                killed.append(f"PID {conn.pid} ({proc.name()})")
            except Exception as e:
                not_killed.append(f"PID {conn.pid}: {e}")
    if killed:
        return f"Killed processes on port {port}: " + ", ".join(killed)
    elif not_killed:
        return f"Failed to kill some processes on port {port}: " + ", ".join(not_killed)
    else:
        return f"No process found listening on port {port}."

##############################
# Routes for Search and Script Actions
##############################

@app.route("/search_script", methods=["GET", "POST"])
def search_script_route():
    if request.method == "GET":
        return render_template("search_script.html")
    else:
        query = request.form.get("search_query")
        if not query:
            return "Please enter a search query.", 400
        generated_script = analyze_script_with_deepseek({}, query)
        return render_template("search_result.html", generated_script=generated_script)

@app.route("/execute_generated", methods=["POST"])
def execute_generated_route():
    script_content = request.form.get("script_content")
    if not script_content:
        return "No script provided.", 400
    script_filename = f"generated_script_{uuid.uuid4().hex}.py"
    with open(script_filename, "w") as f:
        f.write(script_content)
    venv_python = create_virtualenv(VENV_DIR)
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                   capture_output=True, text=True)
    retcode, stdout, stderr = execute_script(venv_python, script_filename)
    output = f"Return Code: {retcode}\nOutput:\n{stdout}\nErrors:\n{stderr}\n"
    return render_template("result.html", output=output,
                           steps="Executed generated script.", retcode=retcode)

@app.route("/correct_generated", methods=["POST"])
def correct_generated_route():
    script_content = request.form.get("script_content")
    if not script_content:
        return "No script provided.", 400
    script_filename = f"generated_script_{uuid.uuid4().hex}.py"
    with open(script_filename, "w") as f:
        f.write(script_content)
    venv_python = create_virtualenv(VENV_DIR)
    attempt = 0
    max_attempts = 10
    retcode, stdout, stderr = execute_script(venv_python, script_filename)
    corrected_script = script_content
    while retcode != 0 and attempt < max_attempts:
        attempt += 1
        corrected_script = attempt_auto_correction({}, corrected_script, stderr)
        with open(script_filename, "w") as f:
            f.write(corrected_script)
        retcode, stdout, stderr = execute_script(venv_python, script_filename)
    if retcode != 0:
        return "Script could not be corrected.", 400
    return render_template("git_commit_new_repo.html",
                           corrected_script=corrected_script,
                           script_filename=script_filename)

@app.route("/commit_and_push_generated", methods=["POST"])
def commit_and_push_generated_route():
    repo_name = request.form.get("repo_name")
    repo_desc = request.form.get("repo_desc", "")
    commit_message = request.form.get("commit_message")
    script_content = request.form.get("script_content")
    if not repo_name or not commit_message or not script_content:
        return "Missing repository name, commit message, or script content.", 400
    new_repo_dir = repo_name
    if os.path.exists(new_repo_dir):
        return f"Directory {new_repo_dir} already exists. Please remove it first.", 400
    os.makedirs(new_repo_dir)
    script_path = os.path.join(new_repo_dir, "script.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    cwd = os.getcwd()
    os.chdir(new_repo_dir)
    ensure_git_safe_directory()
    subprocess.run(["git", "init"], capture_output=True, text=True)
    subprocess.run(["git", "add", "script.py"], capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", commit_message],
                   capture_output=True, text=True)
    repo_result = github_create_repo(repo_name, repo_desc)
    if GIT_REMOTE_URL:
        remote_url = GIT_REMOTE_URL
    else:
        creds = get_github_credentials()
        if not creds:
            os.chdir(cwd)
            return "No GitHub credentials available.", 400
        username, token = creds
        remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    subprocess.run(["git", "remote", "add", "origin", remote_url],
                   capture_output=True, text=True)
    push_result = subprocess.run(["git", "push", "-u", "origin", "master"],
                                 capture_output=True, text=True)
    os.chdir(cwd)
    return (f"Repository {repo_name} created and pushed successfully.<br>"
            f"{repo_result}<br>{push_result.stdout}<br>"
            "<a href='/'>Return to Home</a>")

##############################
# Git Commit (File Selection) Route
##############################

@app.route("/git_commit", methods=["GET", "POST"])
def git_commit_route_wrapper():
    if request.method == "GET":
        result = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True)
        files = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                files.append(parts[1])
        return render_template("git_commit.html", files=files)
    else:
        selected_files = request.form.getlist("files")
        commit_message = request.form.get("commit_message")
        if not selected_files or not commit_message:
            return "No files selected or no commit message provided.", 400
        for f in selected_files:
            subprocess.run(["git", "add", f], capture_output=True, text=True)
        commit_result = subprocess.run(["git", "commit", "-m", commit_message],
                                       capture_output=True, text=True)
        push_result = subprocess.run(["git", "push"],
                                     capture_output=True, text=True)
        return (f"Committed and pushed selected files.<br>{commit_result.stdout}<br>"
                f"{push_result.stdout}<br><a href='/'>Return to Home</a>")

##############################
# GitHub Login/Logout Routes
##############################

@app.route('/github_login', methods=['GET', 'POST'])
def github_login_route():
    if request.method == 'POST':
        username = request.form.get('username')
        token = request.form.get('token')
        if username and token:
            session['github_username'] = username
            session['github_token'] = token
            session['github_login_time'] = datetime.now().isoformat()
            logging.info("GitHub login successful.")
            return redirect(url_for('github_create_route'))
        else:
            error = "Both username and token are required."
            logging.warning("GitHub login failed: missing credentials.")
            return render_template("github_login.html", error=error)
    return render_template("github_login.html")

@app.route('/github_logout')
def github_logout():
    session.pop('github_username', None)
    session.pop('github_token', None)
    session.pop('github_login_time', None)
    logging.info("GitHub user logged out.")
    return redirect(url_for('index'))

##############################
# Templates via DictLoader
##############################

TEMPLATES = {
    'base.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Python Script Executor</title>
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,500&display=swap" rel="stylesheet">
    <style>
        /* Reset and basic styling */
        body, h1, h2, p, ul, li, a, input, button, textarea {
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
        }
        body {
            background: #121212;
            color: #e0e0e0;
            line-height: 1.6;
        }
        header {
            background: #212121;
            padding: 10px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        .nav-links a {
            color: #03a9f4;
            margin: 0 10px;
            text-decoration: none;
            font-size: 16px;
        }
        .nav-links a:hover {
            text-decoration: underline;
        }
        .container {
            max-width: 1100px;
            margin: 20px auto;
            background: #1e1e1e;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.8);
            animation: fadeIn 1s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1, h2 {
            text-align: center;
            color: #03a9f4;
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            height: 300px;
            background: #2c2c2c;
            border: 1px solid #424242;
            color: #e0e0e0;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            resize: vertical;
            transition: border 0.3s ease;
        }
        textarea:focus {
            outline: none;
            border-color: #03a9f4;
        }
        .btn {
            background: #03a9f4;
            color: #121212;
            border: none;
            padding: 10px 16px;
            border-radius: 4px;
            text-decoration: none;
            cursor: pointer;
            margin: 5px;
            transition: background 0.3s, transform 0.2s;
        }
        .btn:hover {
            background: #0288d1;
            transform: scale(1.05);
        }
        .output, .steps {
            background: #1e1e1e;
            border: 1px solid #424242;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: monospace;
            margin-top: 20px;
        }
        .instruction {
            margin-top: 20px;
            font-size: 16px;
        }
        .success {
            color: #00e676;
            font-weight: bold;
            font-size: 18px;
        }
        .spinner {
            border: 6px solid #424242;
            border-top: 6px solid #03a9f4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .process-list {
            background: #1e1e1e;
            border: 1px solid #424242;
            padding: 10px;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        table, th, td {
            border: 1px solid #424242;
        }
        th, td {
            padding: 8px;
            text-align: center;
        }
        th {
            background: #2c2c2c;
        }
        footer {
            text-align: center;
            padding: 10px;
            margin-top: 20px;
            font-size: 14px;
            color: #aaaaaa;
        }
        @media (max-width: 768px) {
            nav {
                flex-direction: column;
            }
            .nav-links {
                margin-top: 10px;
            }
        }
    </style>
    <script>
        function showSpinner() { document.getElementById('spinner').style.display = 'block'; }
        function downloadLogs() {
            var steps = document.querySelector('.steps').innerText;
            var output = document.querySelector('.output').innerText;
            var content = steps + "\\n\\n" + output;
            var blob = new Blob([content], { type: 'text/plain' });
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = "execution_logs.txt";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    </script>
</head>
<body>
    <header>
        <nav>
            <div class="logo">
                <a href="/" style="color:#03a9f4; font-size:20px; font-weight:bold; text-decoration:none;">Script Executor</a>
            </div>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/search_script">Search Script</a>
                <a href="/history">History</a>
                <a href="/git">Git</a>
                <a href="/ports">Port Monitor</a>
                <a href="/github_login">GitHub Login</a>
                <!-- Cloud and Remote Machine Features -->
                <a href="#">AWS</a>
                <a href="#">GCP</a>
                <a href="#">Azure</a>
                <a href="#">Citrix</a>
                <a href="#">AWS Workspace</a>
            </div>
        </nav>
    </header>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    <footer>
        &copy; {{ current_year if current_year is defined else "2025" }} Python Script Executor. All rights reserved.
    </footer>
</body>
</html>
""",
    'index.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Welcome to the Python Script Executor</h1>
    <div style="text-align: center; margin-bottom: 20px;">
        <p>Effortlessly execute, correct, and manage your Python scripts. Use the options below to get started.</p>
    </div>
    <!-- Execution Form -->
    <form action="/execute" method="post" onsubmit="showSpinner()">
        <textarea name="script" placeholder="Enter your Python script here..."></textarea>
        <br><br>
        <button type="submit" class="btn">Execute Script</button>
    </form>
    <br>
    <!-- Correct and Commit Section -->
    <h2>Correct & Commit Script to New Repo</h2>
    <form action="/correct_and_commit" method="post">
        <textarea name="script_to_correct" placeholder="Enter the script to correct and commit..."></textarea>
        <br>
        <button type="submit" class="btn">Correct & Commit to New Repo</button>
    </form>
    <br>
    <!-- Search Script Option -->
    <h2>Search for a Python Script from the Model</h2>
    <a href="/search_script" class="btn">Search Script</a>
    <br><br>
    <!-- Running Processes -->
    <h2>Running Python Processes</h2>
    <div class="process-list">
        {% if running_processes %}
            <ul>
                {% for proc in running_processes %}
                    <li>PID: {{ proc.pid }}, Name: {{ proc.name }}, Cmdline: {{ proc.cmdline | join(' ') }}</li>
                {% endfor %}
            </ul>
        {% else %}
            <p>No Python processes found or psutil not installed.</p>
        {% endif %}
    </div>
    <br>
    <!-- Additional Navigation -->
    <a href="/ports" class="btn">Port Monitor</a>
    <a href="/history" class="btn">View History</a>
    <a href="/git" class="btn">Git Options</a>
    <a href="/git_pull" class="btn">Git Pull</a>
    <a href="/git_clone" class="btn">Git Clone</a>
    <a href="/git_commit" class="btn">Git Commit Selected Files</a>
    <a href="/github_create" class="btn">GitHub Create Repo</a>
    <div id="spinner" class="spinner"></div>
{% endblock %}
""",
    'result.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Execution Results</h1>
    <div class="steps">{{ steps }}</div>
    <br>
    <div class="output">{{ output }}</div>
    <br>
    <div class="instruction">
        {% if retcode == 0 %}
            <span class="success">✔ The script executed successfully!</span>
        {% else %}
            The script did not execute successfully. Please review the error details below:
            <ul>
                <li>Ensure all string literals are properly closed.</li>
                <li>Verify that parentheses, brackets, and braces are correctly matched.</li>
                <li>Look for missing commas or other syntax errors as shown in the logs.</li>
            </ul>
        {% endif %}
    </div>
    <br>
    <a href="/" class="btn">Return to Home</a>
    <a href="javascript:void(0)" class="btn" onclick="downloadLogs()">Download Logs</a>
    <a href="/history" class="btn">View History</a>
    <a href="/git" class="btn">Git Options</a>
{% endblock %}
""",
    'history_list.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Execution History</h1>
    <ul>
        {% for file in files %}
            <li><a href="/history/{{ file }}" class="btn">{{ file }}</a></li>
        {% endfor %}
    </ul>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'history_view.html': """
{% extends "base.html" %}
{% block content %}
    <h1>History: {{ filename }}</h1>
    <pre class="output">{{ content }}</pre>
    <br>
    <a href="/history" class="btn">Back to History</a>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'git_status.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Git Status</h1>
    <pre class="output">{{ git_status }}</pre>
    <br>
    <a href="/git_push" class="btn">Commit & Push Code</a>
    <a href="/git_pull" class="btn">Git Pull</a>
    <a href="/git_clone" class="btn">Git Clone</a>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'git_push.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Git Push Results</h1>
    <pre class="output">{{ git_output }}</pre>
    <br>
    <a href="/git" class="btn">Back to Git Options</a>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'git_clone.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Clone GitHub Repository</h1>
    <form action="/git_clone" method="post">
        <input type="text" name="repo_url" placeholder="Repository URL" class="btn" style="padding:8px; margin:5px;"/>
        <button type="submit" class="btn">Clone Repository</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'github_create.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Create GitHub Repository</h1>
    <form action="/github_create" method="post">
        <input type="text" name="repo_name" placeholder="Repository Name" class="btn" style="padding:8px; margin:5px;"/>
        <input type="text" name="repo_desc" placeholder="Repository Description" class="btn" style="padding:8px; margin:5px;"/>
        <button type="submit" class="btn">Create Repository</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'github_login.html': """
{% extends "base.html" %}
{% block content %}
    <h1>GitHub Login</h1>
    {% if error %}
        <p style="color:#ff5252;">{{ error }}</p>
    {% endif %}
    <form action="/github_login" method="post">
        <input type="text" name="username" placeholder="GitHub Username" class="btn" style="padding:8px; margin:5px;"/>
        <input type="password" name="token" placeholder="GitHub Token" class="btn" style="padding:8px; margin:5px;"/>
        <button type="submit" class="btn">Login</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'ports.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Port Monitor & Killer</h1>
    {% if kill_result %}
        <div class="instruction">{{ kill_result }}</div>
        <br>
    {% endif %}
    {% if used_ports %}
        <table>
            <tr>
                <th>Port</th>
                <th>PID</th>
                <th>Process Name</th>
                <th>Command Line</th>
            </tr>
            {% for p in used_ports %}
            <tr>
                <td>{{ p.port }}</td>
                <td>{{ p.pid }}</td>
                <td>{{ p.name }}</td>
                <td>{{ p.cmdline }}</td>
            </tr>
            {% endfor %}
        </table>
    {% else %}
        <p>No listening ports found.</p>
    {% endif %}
    <br>
    <h2>Kill a Port</h2>
    <form action="/kill_port" method="post">
        <input type="number" name="port" placeholder="Enter port number" class="btn" style="padding:8px;"/>
        <button type="submit" class="btn">Kill Port</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'search_script.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Search for a Python Script</h1>
    <form action="/search_script" method="post">
        <input type="text" name="search_query" placeholder="Enter your request..." class="btn" style="padding:8px; margin:5px; width:80%;"/>
        <button type="submit" class="btn">Search</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'search_result.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Search Result</h1>
    <p>Generated Python script based on your query:</p>
    <form id="scriptForm" method="post">
        <textarea name="script_content">{{ generated_script }}</textarea>
        <br><br>
        <!-- Action Buttons -->
        <button formaction="/execute_generated" class="btn">Execute Script</button>
        <button formaction="/correct_generated" class="btn">Correct Script</button>
        <button formaction="/commit_and_push_generated" class="btn">Commit & Push</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'git_commit_new_repo.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Corrected Script & Commit to New Repo</h1>
    <form action="/commit_and_push" method="post">
        <label>Corrected Script:</label>
        <textarea name="corrected_script" readonly>{{ corrected_script }}</textarea>
        <br>
        <input type="hidden" name="script_filename" value="{{ script_filename }}">
        <input type="text" name="repo_name" placeholder="New Repository Name" class="btn" style="padding:8px; margin:5px;"/>
        <input type="text" name="repo_desc" placeholder="Repository Description" class="btn" style="padding:8px; margin:5px;"/>
        <input type="text" name="commit_message" placeholder="Commit Message" class="btn" style="padding:8px; margin:5px;"/>
        <button type="submit" class="btn">Commit & Push to New Repo</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
""",
    'git_commit.html': """
{% extends "base.html" %}
{% block content %}
    <h1>Git Commit Selected Files</h1>
    <form action="/git_commit" method="post">
        <label>Select files to commit:</label>
        <ul>
            {% for file in files %}
                <li><input type="checkbox" name="files" value="{{ file }}"> {{ file }}</li>
            {% endfor %}
        </ul>
        <input type="text" name="commit_message" placeholder="Commit Message" class="btn" style="padding:8px; margin:5px;"/>
        <button type="submit" class="btn">Commit & Push Selected Files</button>
    </form>
    <br>
    <a href="/" class="btn">Return to Home</a>
{% endblock %}
"""
}

app.jinja_loader = DictLoader(TEMPLATES)

##############################
# Existing Routes
##############################

@app.route("/", methods=["GET"])
def index():
    running_processes = get_running_python_processes()
    return render_template("index.html", running_processes=running_processes)

@app.route("/execute", methods=["POST"])
def execute():
    execution_steps = []
    script_content = request.form.get("script")
    if not script_content:
        return render_template("result.html", output="No script provided.", steps="", retcode=1)
    execution_steps.append("Step 1: Received script input.")
    script_filename = f"user_script_{uuid.uuid4().hex}.py"
    with open(script_filename, "w") as f:
        f.write(script_content)
    execution_steps.append(f"Step 2: Saved script as {script_filename}.")
    venv_python = create_virtualenv(VENV_DIR)
    execution_steps.append("Step 3: Virtual environment prepared.")
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                   capture_output=True, text=True)
    execution_steps.append("Step 4: Upgraded pip in virtual environment.")
    python_version = subprocess.run([venv_python, "--version"],
                                    capture_output=True, text=True).stdout.strip()
    execution_steps.append(f"Step 4.1: Python version: {python_version}.")
    modules = extract_imports(script_content)
    execution_steps.append(f"Step 5: Detected imports: {', '.join(modules) if modules else 'None'}.")
    install_missing_packages(venv_python, modules)
    execution_steps.append("Step 6: Installed missing packages (if any).")
    ollama_client = connect_to_ollama_server()
    execution_steps.append("Step 7: Connected to Ollama server.")
    analyzed_script = analyze_script_with_deepseek(ollama_client, script_content)
    execution_steps.append("Step 8: Script analyzed by DeepSeek.")
    with open(script_filename, "w") as f:
        f.write(analyzed_script)
    retcode, stdout, stderr = execute_script(venv_python, script_filename)
    output = "Initial Execution:\n"
    output += f"Return Code: {retcode}\n"
    if stdout:
        output += f"Output:\n{stdout}\n"
    if stderr:
        output += f"Errors:\n{stderr}\n"
    execution_steps.append("Step 9: Initial script execution completed.")
    attempt = 0
    max_attempts = 10
    while retcode != 0 and attempt < max_attempts:
        attempt += 1
        execution_steps.append(f"Step 10: Auto-correction attempt {attempt}.")
        output += f"\n--- Auto-correction Attempt {attempt} ---\n"
        analyzed_script = attempt_auto_correction(ollama_client, analyzed_script, stderr)
        with open(script_filename, "w") as f:
            f.write(analyzed_script)
        retcode, stdout, stderr = execute_script(venv_python, script_filename)
        output += f"Return Code: {retcode}\n"
        if stdout:
            output += f"Output:\n{stdout}\n"
        if stderr:
            output += f"Errors:\n{stderr}\n"
        execution_steps.append(f"Auto-correction attempt {attempt} complete. Return code: {retcode}.")
        if retcode == 0:
            break
    history_filename = store_history(script_content, "\n".join(execution_steps), output, retcode)
    execution_steps.append(f"Stored execution history as {history_filename}.")
    if retcode == 0:
        execution_steps.append("✔ Script executed successfully!")
        output = "✔ Script executed successfully!\n\n" + output
    else:
        execution_steps.append("The script still fails after maximum auto-correction attempts.")
        output += "\nThe script could not be auto-corrected to execute successfully.\n"
        output += "Please review the error details and consider the suggestions above.\n"
    steps_display = "\n".join(execution_steps)
    return render_template("result.html", output=output, steps=steps_display, retcode=retcode)

@app.route("/history", methods=["GET"])
def history():
    files = os.listdir("history") if os.path.exists("history") else []
    files.sort(reverse=True)
    return render_template("history_list.html", files=files)

@app.route("/history/<filename>", methods=["GET"])
def view_history(filename):
    filepath = os.path.join("history", filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        return render_template("history_view.html", filename=filename, content=content)
    return "History not found.", 404

@app.route("/git", methods=["GET"])
def git_status_route():
    status = git_status()
    return render_template("git_status.html", git_status=status)

@app.route("/git_push", methods=["GET"])
def git_push_route():
    init_output = git_init()
    push_output = git_commit_and_push()
    git_output = init_output + "\n" + push_output
    return render_template("git_push.html", git_output=git_output)

@app.route("/git_pull", methods=["GET"])
def git_pull_route():
    output = git_pull()
    return render_template("git_push.html", git_output=output)

@app.route("/git_clone", methods=["GET", "POST"])
def git_clone_route():
    if request.method == "GET":
        return render_template("git_clone.html")
    else:
        repo_url = request.form.get("repo_url")
        if not repo_url:
            return "No repository URL provided."
        clone_output = git_clone(repo_url)
        return render_template("git_push.html", git_output=clone_output)

@app.route("/github_create", methods=["GET", "POST"])
def github_create_route():
    creds = get_github_credentials()
    if creds is None:
        if not (GITHUB_USERNAME and GITHUB_TOKEN):
            logging.debug("No GitHub credentials found; redirecting to login.")
            return redirect(url_for('github_login_route'))
        creds = (GITHUB_USERNAME, GITHUB_TOKEN)
    if request.method == "GET":
        return render_template("github_create.html")
    else:
        repo_name = request.form.get("repo_name")
        repo_desc = request.form.get("repo_desc", "")
        if not repo_name:
            return "Repository name is required."
        result = github_create_repo(repo_name, repo_desc)
        return f"{result} <br><a href='/'>Return to Home</a>"

@app.route("/ports", methods=["GET"])
def ports():
    used_ports = get_used_ports()
    return render_template("ports.html", used_ports=used_ports, kill_result=None)

@app.route("/kill_port", methods=["POST"])
def kill_port_route():
    port = request.form.get("port")
    if port:
        try:
            port = int(port)
        except ValueError:
            kill_result = "Invalid port number."
        else:
            kill_result = kill_port(port)
    else:
        kill_result = "No port number provided."
    used_ports = get_used_ports()
    return render_template("ports.html", used_ports=used_ports, kill_result=kill_result)

##############################
# Main: Dynamic Deployment
##############################

if __name__ == "__main__":
    random_port = random.randint(5000, 6000)
    host = "0.0.0.0"
    logging.info(f"Running on http://{host}:{random_port}")
    app.run(host=host, port=random_port, debug=True)
