import os
import jupytext

# Directory containing Python files
directory = "src"
notebook_content = ""

# Loop through all Python files in the directory and merge their content
for filename in os.listdir(directory):
    if os.path.isdir(filename):
        if filename.endswith(".py"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                notebook_content += (
                    f"# {filename}\n"  # Add the file name as a markdown cell
                )
                notebook_content += file.read() + "\n\n"  # Add the content of the file

# Save the merged content as a single notebook
notebook = jupytext.reads(notebook_content, fmt="py")
jupytext.write(notebook, "merged_notebook.ipynb")
