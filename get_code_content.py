import os
import re
from collections import Counter
from typing import Generator, Tuple, List
import pandas as pd


def extract_docs_type_hints_and_contents_of(code: str, separator: str) -> str:
    # Extract docstrings
    docstring_pattern = re.compile(r"\"\"\"[\s\S]*?\"\"\"")
    docstrings = docstring_pattern.findall(code)

    # Extract function and method signatures
    signature_pattern = re.compile(r"(\s*def\s+\w+\s*\([^)]*\)\s*:)")
    signatures = signature_pattern.findall(code)

    # Extract class definitions
    class_pattern = re.compile(r"(class\s+\w+\s*\([^)]*\)\s*:)")
    classes = class_pattern.findall(code)

    # Extract lines starting with "==================== CONTENTS OF:"
    contents_of_pattern = re.compile(rf"(^.*?{separator}.*?$)", re.MULTILINE)
    contents_of_lines = contents_of_pattern.findall(code)

    # Combine all extracted parts
    extracted_parts = docstrings + signatures + classes + contents_of_lines
    result = "\n".join(extracted_parts)

    return result


class CodeContentOrganizer:
    """
    A class to generate, organize, and optimize code content for LLM prompts.
    Allows optimizing of code content from python libraries by offering all code content,
    while having the ability to lower amount of content by removing less relevant functions or modules for fitting into a prompt.

    Parameters
    ----------
    root_dir : str
        The root directory containing the code files.
    exclude_dirs : list of str, optional
        Directories to exclude from processing (default is None).
    exclude_files : list of str, optional
        Files to exclude from processing (default is None).
    """

    def __init__(
        self,
        root_dir: str,
        exclude_dirs: List[str] = ["__pycache__"],
        exclude_files: List[str] = ["test"],
    ):
        self.root_dir = root_dir
        self.exclude_dirs = exclude_dirs if exclude_dirs is not None else []
        self.exclude_files = exclude_files if exclude_files is not None else []
        self.separator = "===================="
        self.content_separator = " CONTENTS OF: "

    def generate_directory_tree(self) -> Generator[str, None, None]:
        """
        Generate a directory tree structure starting from the root directory.

        Yields
        ------
        str
            A line representing a part of the directory tree.
        """

        def walk_dir(current_dir: str, prefix: str = "") -> Generator[str, None, None]:
            contents = sorted(os.listdir(current_dir))
            files = [
                f
                for f in contents
                if os.path.isfile(os.path.join(current_dir, f))
                and f.endswith(".py")
                and not any(
                    exclude in os.path.join(current_dir, f)
                    for exclude in self.exclude_files
                )
            ]
            dirs = [
                d
                for d in contents
                if os.path.isdir(os.path.join(current_dir, d))
                and not any(
                    exclude in os.path.join(current_dir, d)
                    for exclude in self.exclude_dirs
                )
            ]

            for index, file in enumerate(files):
                if index == len(files) - 1 and not dirs:
                    yield f"{prefix}└── {file}"
                else:
                    yield f"{prefix}├── {file}"

            for index, directory in enumerate(dirs):
                if index == len(dirs) - 1:
                    yield f"{prefix}└── {directory}"
                    yield from walk_dir(
                        os.path.join(current_dir, directory), prefix + "    "
                    )
                else:
                    yield f"{prefix}├── {directory}"
                    yield from walk_dir(
                        os.path.join(current_dir, directory), prefix + "│   "
                    )

        yield f"{os.path.abspath(self.root_dir)}"
        yield from walk_dir(self.root_dir)

    def generate_file_contents(self) -> Generator[Tuple[str, str], None, None]:
        """
        Generate the contents of Python files in the directory tree.

        Yields
        ------
        tuple of (str, str)
            The file path and its content.
        """
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [
                d
                for d in dirs
                if not any(
                    exclude in os.path.join(root, d) for exclude in self.exclude_dirs
                )
            ]
            for file in files:
                if file.endswith(".py") and not any(
                    exclude in os.path.join(root, file)
                    for exclude in self.exclude_files
                ):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        yield file_path, content
                    except Exception as e:
                        yield file_path, f"Could not read file {file_path}: {e}"

    def get_code_content(self) -> str:
        """
        Get the code content from the directory tree.

        Returns
        -------
        str
            The formatted code content.
        """
        content = f"{self.separator} DIRECTORY TREE STRUCTURE {self.separator}\n"
        content += "\n".join(self.generate_directory_tree())

        content += f"\n\n{self.separator} CODE CONTENT {self.separator}\n"
        file_contents = self.generate_file_contents()
        for file_path, file_content in file_contents:
            content += f"\n{self.separator}{self.content_separator}{file_path} {self.separator}\n"
            content += file_content

        return content

    def optimize_content_length(
        self, content: str, max_lines: int, top_p_start=0.5, increments=0.01
    ) -> str:
        """
        Optimize the code content length by filtering less relevant functions.

        Parameters
        ----------
        content : str
            The original code content.
        max_lines : int
            The maximum number of lines for the optimized content.

        Returns
        -------
        str
            The optimized code content.
        """
        lines = content.split("\n")
        if len(lines) <= max_lines:
            return content

        ranking = self._rank_internal_modules(content)
        top_p = top_p_start

        while len(lines) > max_lines and top_p > 0:
            top_index = int(len(ranking) * top_p)
            modules_to_remove = ranking.iloc[top_index:].index.tolist()
            content = self._remove_modules_by_name(
                content, modules_to_remove, self.separator
            )
            lines = content.split("\n")
            top_p -= increments

        return content

    def _rank_internal_modules(self, content: str) -> pd.Series:
        """
        Rank internal modules based on their usage within the package.

        Parameters
        ----------
        content : str
            The original code content.

        Returns
        -------
        pd.Series
            A Series containing the ranking of each internal module.
        """
        module_imports = re.findall(
            r"^\s*from\s+(\.\w+)\s+import|^\s*import\s+(\.\w+)",
            content,
            flags=re.MULTILINE,
        )
        flattened_imports = [
            item for sublist in module_imports for item in sublist if item
        ]
        import_counts = Counter(flattened_imports)
        return pd.Series(import_counts).sort_values(ascending=False)

    def _remove_modules_by_name(
        self, code: str, module_names: List[str], separator: str
    ) -> str:
        """
        Removes modules from the codecontent-string based on their names.

        Parameters
        ----------
        code : str
            The original code as a string.
        module_names : list of str
            The names of the modules to be removed.
        separator : str
            The separator string used in the code content.

        Returns
        -------
        str
            The code with the specified modules removed.
        """
        # Create a regex pattern to match the content blocks for the specified modules
        module_patterns = [
            re.escape(separator)
            + rf"{self.content_separator}.*"
            + re.escape(module.replace(".", os.sep))
            + r"\.py "
            + re.escape(separator)
            + r"[\s\S]*?(?="
            + re.escape(separator)
            + rf"{self.content_separator}|\Z)"
            for module in module_names
        ]
        module_pattern = "|".join(module_patterns)

        # Use re.sub to remove the matched blocks
        code = re.sub(module_pattern, "", code, flags=re.MULTILINE)

        return code

    def _rank_internal_functions(self, content: str) -> pd.Series:
        """
        Rank internal functions based on their usage within the package.

        Parameters
        ----------
        content : str
            The original code content.

        Returns
        -------
        pd.Series
            A Series containing the count of each function.
        """
        lines = content.split("\n")
        func_counts = dict()
        all_funcs = set(f"{i[4:].split('(')[0]}" for i in lines if i.startswith("def "))
        for func in all_funcs:
            c = content.count(func)
            func_counts[func] = c
        func_counts = pd.Series(func_counts).sort_values(ascending=False)
        return func_counts

    def _remove_functions_by_name(self, code: str, func_names: List[str]) -> str:
        """
        Removes functions from the codecontent-string based on their names.

        Parameters
        ----------
        code : str
            The original code as a string.
        func_names : list of str
            The names of the functions to be removed.

        Returns
        -------
        str
            The code with the specified functions removed.
        """
        # Create a regex pattern to match the function definitions, including decorators
        func_pattern = r"(^\s*@.*\n\s*)?(^\s*def\s+({})\s*\(.*?\):[\s\S]*?)(?=^\s*@|\Z|^\s*def\s)".format(
            "|".join(re.escape(func) for func in func_names)
        )

        # Use re.sub to remove the matched functions
        code = re.sub(func_pattern, "", code, flags=re.MULTILINE)

        return code


if __name__ == "__main__":
    root_dir = r".venv\Lib\site-packages\accelerate"

    organizer = CodeContentOrganizer(root_dir)
    content = organizer.get_code_content()
    optimized_content = organizer.optimize_content_length(content, 5000)
    optimized_content = extract_docs_type_hints_and_contents_of(optimized_content, organizer.separator)
    
    print(len(optimized_content.split("\n")))
