"""Monitor QuantMind pipeline using Claude + GitHub."""
from github import Github
import os
import json

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "ghp_your_token_here")
REPO_NAME = "yourusername/quant_system"

try:
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(REPO_NAME)
    print(f"Connected to: {repo.full_name}")
    
    runs = repo.get_workflow("quantmind.yml").get_runs()
    if runs.totalCount > 0:
        latest = runs[0]
        print(f"Latest run: {latest.conclusion}")
        print(f"Status: {latest.status}")
        print(f"Time: {latest.updated_at}")
    else:
        print("No workflow runs yet.")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure:")
    print("  - GITHUB_TOKEN env var is set")
    print("  - Repo name is correct")
    print("  - You have PyGithub installed: pip install PyGithub")
