"""Execute the TDCS notebook top-to-bottom in IPython and save real cell outputs.

This runner avoids starting an extra ZeroMQ kernel process on Windows.  It is
an execution fallback for environments where Jupyter kernel startup stalls.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tdcs_separate_multiscale_experiment.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    # Do not write IPython's shared user history database from this isolated run.
    shell = InteractiveShell.instance(config=Config({"HistoryManager": {"hist_file": ":memory:"}}))
    count = 0
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        start = time.monotonic()
        count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
        cell.execution_count = count
        cell.outputs = outputs
        error = result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__,
                                                      evalue=str(error), traceback=[str(error)]))
            nbformat.write(notebook, notebook_path)
            raise RuntimeError(f"Notebook cell {cell_index} failed") from error
        print(f"Executed cell {cell_index}: {time.monotonic() - start:.1f}s", flush=True)
    notebook.metadata["execution_method"] = "top-to-bottom in-process IPython; real captured outputs"
    notebook.metadata["execution_status"] = "passed"
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    print(f"Saved {count} executed code cells: {notebook_path}", flush=True)


if __name__ == "__main__":
    main()
