import os

def extract_python_files_with_content(repo_path, output_file="output.txt"):
    lines = []
    lines.append("Folder structure:")
    lines.append("###")
    lines.append("")

    for root, dirs, files in os.walk(repo_path):
        for name in files:
            if name.endswith(".py"):
                file_path = os.path.join(root, name)
                rel_path = os.path.relpath(file_path, repo_path)

                lines.append(f"{rel_path}:")
                lines.append("'''")

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines.append(content)

                lines.append("'''")
                lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    repo_path = "src/"  # Change this to your repository path
    extract_python_files_with_content(repo_path)
    print(f"Extracted Python files and their content to output.txt")
if __name__ == "__main__":
    main()