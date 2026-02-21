# MetriKG

**MetriKG** is a web-based tool for calculating structural and semantic metrics on RDF-based Knowledge Graphs.  
It was developed as part of a Bachelor's thesis at **TU Wien (Vienna University of Technology)**.

---

## Requirements

- **Python 3.10+**
- **pip** (Python package manager)
- Optional but recommended: a **virtual environment** (`venv`) to isolate dependencies.

---

## Installation

1. **Clone or download the repository**
   ```
   git clone https://github.com/dmki-tuwien/metriKG.git
   cd metriKG
   ```

2. **Recommended: Create and activate a virtual environment**
    
    - Windows:

    ```bash
    python -m venv venv
    venv\Scripts\activate   
    ```

    - Linux / macOS:
    
    ```bash
    python -m venv venv
    source venv/bin/activate    
    ```

3. **Install all required packages**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Start the web interface using Streamlit:

    streamlit run app_v2.py

`app_v1.py` is just the first version of the tool, which does not contain the visualization component.

By default, the tool will be accessible at: [http://localhost:8501](http://localhost:8501)

To **stop** the application, press **Ctrl + C** in the terminal.

---

## Usage

### Metric Computation Component

1. Upload an **RDF file** or provide a **SPARQL endpoint URL** (optionally specify a **Default Graph**).

2. Select the desired **metrics** to compute.

3. Click **Calculate** to start the evaluation.

4. The **results** will be displayed as a **table** below the Calculate button.

### Metric History Visualization

1. Upload a **CSV file** conatining timestamped metric values. The plot will appear automatically then.

2. Select the desired **metric** to be visualized.