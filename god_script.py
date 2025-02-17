import os

def create_project_structure():
    base_path = "twitter_bookmark_manager"

    structure = {
        "core": [
            "__init__.py",
            "auth.py",
            "data_ingestion.py",
            "ai_categorization.py",
            "deduplication.py",
            "rag.py"
        ],
        "web": {
            "static": ["styles.css", "script.js"],
            "templates": ["index.html", "chat.html"],
            "files": ["server.py"]
        },
        "database": ["db.py", "models.py"],
        "models": ["mistral-7b-instruct-v0.1.Q4_K_M.gguf"],
        "media": [],
        "config": ["constants.py", "secrets.json"],
        "tests": [],
        "root": ["requirements.txt", "main.py"]
    }

    def create_files(path, files):
        for file in files:
            with open(os.path.join(path, file), "w") as f:
                if file.endswith(".py"):
                    f.write("# Placeholder for " + file)

    os.makedirs(base_path, exist_ok=True)

    # Create core files
    core_path = os.path.join(base_path, "core")
    os.makedirs(core_path, exist_ok=True)
    create_files(core_path, structure["core"])

    # Create web files
    web_path = os.path.join(base_path, "web")
    os.makedirs(web_path, exist_ok=True)
    for sub_dir, files in structure["web"].items():
        if sub_dir == "files":
            create_files(web_path, files)
        else:
            sub_path = os.path.join(web_path, sub_dir)
            os.makedirs(sub_path, exist_ok=True)
            create_files(sub_path, files)

    # Create database files
    db_path = os.path.join(base_path, "database")
    os.makedirs(db_path, exist_ok=True)
    create_files(db_path, structure["database"])

    # Create models directory
    models_path = os.path.join(base_path, "models")
    os.makedirs(models_path, exist_ok=True)
    create_files(models_path, structure["models"])

    # Create media directory
    media_path = os.path.join(base_path, "media")
    os.makedirs(media_path, exist_ok=True)

    # Create config files
    config_path = os.path.join(base_path, "config")
    os.makedirs(config_path, exist_ok=True)
    create_files(config_path, structure["config"])

    # Create tests directory
    tests_path = os.path.join(base_path, "tests")
    os.makedirs(tests_path, exist_ok=True)

    # Create root files
    create_files(base_path, structure["root"])

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")
