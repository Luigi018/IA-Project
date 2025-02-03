# IA-Project

## Repository Content:
- **Instances**: Graphs provided by the professor for algorithm execution.
- **Report**: LaTeX files and the report in PDF format.
- **requirements.txt**: Packages required to run the code.
- **VertexCoverOptimized.py, VertexCoverSlow.py, VertexCoverLimited.py**: Code implementations developed for the project.
- **Instructions.txt**: Instructions for testing the code.

## Basic Commands

Install the required packages using the following command:

    pip install -r requirements.txt

Or in the absence of the requirements.txt, install the packages with the following command:

    pip install numpy networkx matplotlib

Test the first code:

    python3 VertexCoverSlow.py Instances/vc_20_60_01.txt

Test the second code:

    python3 VertexCoverOptimized.py Instances/vc_20_60_01.txt

Test the third code:

    python3 VertexCoverLimited.py Instances/vc_20_60_01.txt

Code output:
- Best solution;
- Best cost;
- Total iterations;
- Algorithm Execution Time.

These values allow you to compare the behavior of the two scripts and their execution speed.

The differences between the first and second implementations are detailed in the report.  
For more information, refer to **"Progetto_IA.pdf"** inside the **Report** folder.
