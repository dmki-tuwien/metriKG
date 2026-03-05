import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About")

st.markdown(
"""

### About MetriKG

MetriKG is a web-based tool for computing and exploring structural metrics of knowledge graphs. 
The system supports both local RDF files and SPARQL endpoints, enabling users to calculate and 
analyze a range of graph metrics in a structured and reproducible way.

[MetriKG on GitHub](https://github.com/dmki-tuwien/metriKG)

### Developer and Academic Context

MetriKG was developed by **Hasan Günes** as part of a bachelor's thesis at **TU Wien 
(Technical University of Vienna)**.

The project was conducted **on behalf of  Univ.Prof. Dr.-Ing. Katja Hose** and **under the supervision of 
Dr. Milos Jovanovik**.

The goal of the project is to provide a practical tool for computing and analyzing structural 
metrics of knowledge graphs, with a particular focus on reproducibility and the analysis of 
evolving knowledge graphs.

### Functionalities

The tool provides two functionalities: **Metric Computation** and **Metric History Visualization**.

#### Metric Computation

The **Metric Computation** module allows you to calculate selected knowledge graph metrics either from a **local RDF file** or from a **SPARQL endpoint**.

**How to use it:**
1. Choose the data source: **Local RDF-File** (upload an RDF file) or **SPARQL Endpoint** (enter the endpoint URL; optionally a default graph IRI).
2. Select one or more metric categories using the checkboxes.
3. Click **Calculate** to run the computation. The results are shown in the **Metric Values** table, including the metric value, the data source, and a timestamp. You can download the resultrs in the CSV format or save the values to the brwoser-memory.

The computation is executed in a reproducible way and is designed to support repeated runs, enabling comparisons across different graph versions, files, or endpoints.

#### Metric History Visualization

The **Metric History Visualization** module allows users to analyze how metric values change over time.

**How to use it:**
1. Upload a **CSV file** containing previously computed metric values. The file must include the columns *Metric*, *Value*, *Source*, and *Run At*.
2. Select the metric you want to visualize from the dropdown menu.
3. The tool will generate a **time-based line chart** showing how the selected metric evolved across different runs.

Each point in the chart represents a recorded metric value at a specific timestamp. This functionality helps users track structural changes in knowledge graphs and compare results across different datasets, versions, or data sources.

""")