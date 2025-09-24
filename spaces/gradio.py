import os
from huggingface_hub import HfApi, upload_folder, create_repo

token = os.environ["HF_TOKEN"]
space_id = os.environ["HF_SPACE_ID"]
# garante que exista como Space Gradio
api = HfApi()
try:
    api.create_repo(
        repo_id=space_id,
        token=token,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False
    )
except Exception as e:
    print("create_repo warn:", e)

# Sobe a pasta 'spaces' como conteúdo do Space
upload_folder(
    repo_id=space_id,
    repo_type="space",
    folder_path="spaces",
    token=token,
    commit_message="CI: update Space app"
)