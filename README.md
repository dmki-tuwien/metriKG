# MetriKG

**MetriKG** is a web-based tool for calculating structural metrics on RDF-based Knowledge Graphs.  
It was developed as part of a Bachelor's thesis at **TU Wien (Vienna University of Technology)**.

---

## Requirements

- **Python 3.10+**
- **pip** (Python package manager)
- Optional but recommended: a **virtual environment** (`venv`) to isolate dependencies.

---

## Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/dmki-tuwien/metriKG.git
   cd metriKG
   ```

2. **Recommended: Create and activate a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate     # Linux / macOS
    venv\Scripts\activate        # Windows PowerShell
    ```

3. **Install all required packages**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Start the web interface using Streamlit:

    ```bash
    streamlit run app.py
    ```

By default, the tool will be accessible at: [http://localhost:8501](http://localhost:8501)

To **stop** the application, press **Ctrl + C** in the terminal.

---

## Usage

1. Upload an **RDF file** or provide a **SPARQL endpoint URL** (optionally specify a **Default Graph**).

2. Select the desired **metrics** to compute.

3. Click **Calculate** to start the evaluation.

4. The **results** will be displayed as a **table** below the Calculate button.