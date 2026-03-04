"""
Main entry point of the Streamlit web application.

Handles the entire user interface for:
- Uploading RDF files or connecting to SPARQL endpoints
- Selecting metrics and executing the metric computation notebooks
- Displaying results in a structured table

Internally manages session state, notebook execution via Papermill, and dynamic UI updates.
"""

import streamlit as st
import pandas as pd
import papermill as pm
import scrapbook as sb
import tempfile, os
from pathlib import Path
from datetime import datetime
import sys, subprocess
import jupyter_client
from urllib.parse import urlparse, urlsplit

import streamlit.components.v1 as components
import time

import altair as alt

# Initializing session state variables for controlling the app flow
if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False

if "first_run_made" not in st.session_state:
    st.session_state.first_run_made = False

if "full_table" not in st.session_state:
    st.session_state.full_table = False

if "calculating" not in st.session_state:
    st.session_state.calculating = False

def clean_iri(s: str) -> str:
    """
    Cleans an IRI string by removing unwanted characters like quotes and brackets.
    """

    if s is None:
        return ""
    # trim, delete Smart-Quotes/Quotes/Brackets
    s = s.strip().strip('\'"“”‘’<>')
    return s

def is_valid_iri(iri: str) -> bool:
    """
    Validates if a given string is a valid IRI.
    """
    if not iri or " " in iri:
        return False
    p = urlparse(iri)

    forbidden = set('<>"{}|\\^`')
    if any(ch in forbidden for ch in iri):
        return False
    return True

def clean_url(s: str) -> str:
    """
    Cleans a URL string by removing unwanted characters around the edges.
    """
    if s is None:
        return ""
    # nur Rand bereinigen (Wrapper/Spaces), innen nichts verändern
    return s.strip().strip('\'"“”‘’<>')

def is_valid_endpoint_url(url: str) -> bool:
    """
    Validates if the given URL is a valid SPARQL endpoint URL.
    """
    if not url or any(ch.isspace() for ch in url):
        return False
    
    forbidden = set('<>"{}|\\^`')

    if any(ch in forbidden for ch in url):
        return False
    p = urlsplit(url)
    return p.scheme in ("http", "https") and bool(p.netloc) and not p.fragment

# Defining the list of all metric rows
ALL_METRIC_ROWS = [
    # Paths/Depth
    "Number of Paths", "Absolute Depth", "Average Depth", "Maximum Depth",
    # Ontology Tangledness
    "Ontology Tangledness",
    # Degree
    "Degree Variance",
    # Primitives
    "Number of Entities", "Number of Properties", "Number of Classes", "Number of Instances", "Number of Object Properties",
    # Depth of Inheritance Tree
    "Depth of Inheritance Tree", 
    # T-Box
    "Property Class Ratio", "Class Property Ratio",
    "Inheritance Richness", "Attribute Richness",
    # A-Box
    "Average Population", "Average Class Connectivity",
    # Cohesion
    "Cohesion",
]

# Initializing the results DataFrame in the session state
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame({
        "Metric": ALL_METRIC_ROWS,
        "Value": [None] * len(ALL_METRIC_ROWS),
        "Source": [None] * len(ALL_METRIC_ROWS),     
        "Run At": [None] * len(ALL_METRIC_ROWS),  
    })

# Mapping of categories to their metrics
CATEGORY_TO_METRICS = {
    "paths_depth" : ["Number of Paths", "Absolute Depth", "Average Depth", "Maximum Depth"],
    "ont_tangledness" : ["Ontology Tangledness"],
    "degree_variance": ["Degree Variance"],
    "primitives": ["Number of Entities", "Number of Properties", "Number of Classes", "Number of Instances", "Number of Object Properties"],
    "depth_of_inheritance_tree" : ["Depth of Inheritance Tree"],
    "tbox": ["Property Class Ratio", "Class Property Ratio", "Inheritance Richness", "Attribute Richness"],
    "abox": ["Average Class Connectivity", "Average Population"],
    "cohesion": ["Cohesion"],
}

# updating dataframe with new results
def update_results_df(part_df: pd.DataFrame, source_label: str, source_value: str):
    """
    Updates the results DataFrame with the new metric values and source information.
    """
    # part_df must have columns ["Metric","Value"] 

     #getting date and time for Run At value
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    # sets the metric as index 
    base = st.session_state.results_df.set_index("Metric")
    # just takes the value for every metric and ignores other columns
    upd = part_df.set_index("Metric")[["Value"]]

    # updating values
    base.loc[upd.index, "Value"] = upd["Value"]
    base.loc[upd.index, "Source"] = source_value
    base.loc[upd.index, "Run At"] = now

    st.session_state.results_df = base.reset_index()

def ensure_kernel(kernel_name="metrikg-venv", display=None):
    """
    Ensures that the specified Jupyter kernel is available. If not, it registers it.
    """
    display = display or f"Python 3 ({kernel_name})"
    try:
        # check if kernel exists
        jupyter_client.kernelspec.KernelSpecManager().get_kernel_spec(kernel_name)
    except Exception:
        # register a kernel, which uses this venv
        subprocess.check_call([sys.executable, "-m", "ipykernel", "install", "--user",
                               "--name", kernel_name, "--display-name", display])
    return kernel_name

# Ensure that the kernel "metrikg-venv" is available
KERNEL = ensure_kernel() 

# Streamlit page configuration
st.set_page_config(page_title="Metric Computation for Evolving Knowledge Graphs", page_icon="📊", layout="wide")

# Paths to files for notebooks
BASE = Path(__file__).resolve().parent
NB_FILE = BASE / "nb_file.ipynb"
NB_EP   = BASE / "nb_endpoint.ipynb"

# mapping: checkbox -> set of metrics
METRICS = {
    "paths_depth": "Number of Paths, Absolute/Average/Maximum Depth",
    "ont_tangledness": "Ontology Tangledness",
    "degree_variance": "Degree Variance",
    "primitives": "Primitives (Entities, Properties, Classes, Instances, Object Properties)",
    "depth_of_inheritance_tree": "Depth of Inheritance Tree",
    "tbox": "T-Box (Property/Class-Ratio, Class/Property-Ratio, Inheritance/Attribute Richness)",
    "abox": "A-Box (Average Population, Average Class Connectivity)",
    "cohesion": "Cohesion",
}

# mapping: checkbox -> tooltip text (local version)
METRIC_HELP_LOCAL = {
    "paths_depth":  "In the context of an RDF graph, a path is a sequence of nodes connected by edges. Nodes correspond to the subjects and objects of RDF triples, while the predicate "
                    " of a triple represents the directed edge from the subject to the object.\n\n"

                    " A \"root\" is defined as any node that does not appear as an object in any triple, ensuring that traversals start at the top-level nodes of the graph structure. "
                    "The traversal is performed using a depth-first search (DFS) algorithm.\n\n"

                    "**Number of Paths** ([Reference](https://ieeexplore.ieee.org/document/4031647)):  \n &nbsp;&nbsp;&nbsp;&nbsp;The total count of all unique paths starting from a "
                    "root node and ending at a leaf node (a node with no outgoing edges or a literal).\n\n"
                    
                    "**Absolute Depth** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/isaf.1360)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The sum of the lengths (in edges) of all identified paths.\n\n"
                    
                    "**Average Depth** ([Reference](https://link.springer.com/chapter/10.1007/11762256_13)):  \n &nbsp;&nbsp;&nbsp;&nbsp;  The average length of a path (Absolute Depth / Number of Paths)\n\n"
                    
                    "**Maximum Depth** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/isaf.1360)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The length of the longest path found in the graph",

    "ont_tangledness":  "**Ontology Tangledness** ([Reference](https://www.scitepress.org/Link.aspx?doi=10.5220/0004541400480057)):  \n"
                        "&nbsp;&nbsp;&nbsp;&nbsp;The ratio of total classes (criteria for a class: see **Number of Classes**) to classes"
                        " with more than 1 superclass (`rdfs:subClassOf`).\n\n",
                        
    "degree_variance":  "**Degree Variance** ([Reference](https://link.springer.com/article/10.1134/S1064230711010072)):  \n"
                        " &nbsp;&nbsp;&nbsp;&nbsp; This metric measures how much the connectivity of nodes in an RDF graph deviates from the average.\n\n"
                        " The degree of a node is defined as the total number of incoming and outgoing edges."
                        " The mean degree μ is calculated as (2 x number of edges) / (number of nodes), since each edge contributes to two node degrees."
                        " The Degree Variance is then computed as the average squared deviation of each node's degree from this mean:  \n"
                        " Var(d) = Σ((dᵢ - μ)²) / (nG - 1), where dᵢ is the degree of node *i* and nG the number of nodes.\n\n"
                        " A higher variance indicates a more uneven edge distribution (some nodes are highly connected while others are isolated), "
                        "whereas a lower variance reflects a more balanced and uniform graph structure.\n\n",
    
    "primitives":   "**Number of Entities** ([Reference](https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-018-0188-7)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The total count of unique resources (URIs and BNodes) that appear as a subject or a non-literal object in any triple.\n\n"
                    
                    "**Number of Properties** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/smr.341)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The sum of two counts:  \n"
                    " &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.  Properties explicitly declared as `owl:ObjectProperty`, `owl:DatatypeProperty`, or `RDF.Property` (T-Box).  \n "
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.  All unique predicates used in the graph (A-Box). \n\n"
                    
                    "**Number of Classes** ([Reference](https://ieeexplore.ieee.org/document/4031647)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The count of classes. The criteria for identifiying a class:  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.  Any resource that is the object of an `rdf:type` triple.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.  Both the subject and object of an `rdfs:subClassOf` / `owl:equivalentClass` triple.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. The subject of a triple declaring an `owl:Restriction`.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.  The subject of complex class definitions using `owl:unionOf`, `owl:intersectionOf`, `owl:complementOf` or `owl:oneOf`.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5. The subject of an `owl:hasValue` restriction.  \n\n"
                    
                    "**Number of Instances** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp;" 
                    "Count of all individual instances in the graph.  Specifically, this is the number of unique subjects in triples of the form `(s, rdf:type, o)` "
                    " where `s` is not itself a class. In other words, classes (`rdf:type` objects) are excluded from the count to ensure only individuals are counted. \n\n"
                    
                    "**Number of Object Properties** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/isaf.1360)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The sum of two counts:  \n"
                    " &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Properties explicitly declared as `owl:ObjectProperty` (T-Box).  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. All unique predicates in the rdf triples that have a non-literal as an object (A-Box). ",

    "depth_of_inheritance_tree": "**Depth of Inheritance Tree** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/smr.341)):  \n &nbsp;&nbsp;&nbsp;&nbsp; "   
                                 "This metric measures the maximum depth (= number of edges in longest path) of the class hierarchy based on `rdfs:subClassOf` relationships.\n\n"
                                 "The calculation starts by identifying all root classes (classes that are not a subclass of any other class). From each root, the hierarchy is traversed downwards using a depth-first search (DFS) algorithm to find the longest path to a leaf class (a class with no subclasses).  \n\n",

    "tbox": "**Property/Class-Ratio** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/smr.341)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The ratio of explicitly declared T-Box properties (`owl:ObjectProperty`, `owl:DatatypeProperty`, `RDF.Property`) to the total number of classes (criteria for a class: see **Number of Classes**). \n\n"
            
            "**Class/Property-Ratio** ([Reference](https://www.scitepress.org/Link.aspx?doi=10.5220/0004541400480057)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The inverse of the Property Class Ratio, showing the ratio of classes to properties. \n\n"
            
            "**Inheritance Richness** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The ratio of the number of subclass relationships (`rdfs:subClassOf`) to the total number of classes (criteria for a class: see **Number of Classes**).  \n\n"
            
            "**Attribute Richness** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The ratio of explicitly declared datatype properties (`owl:DatatypeProperty`) to the total number of classes. (criteria for a class: see **Number of Classes**)",
    
    "abox": "**Average Population** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp;The average number of instances per class in the ontology.  \n It is calculated by dividing the total number of instances (see **Number of Instances**) by the total number of classes (criteria for a class: see **Number of Classes**).\n\n"
            
            "**Average Class Connectivity** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The average number of relationships that instances of a class have with instances of other classes.  \n" 
            " Relationships between resources (ignoring `rdf:type` statements and links to Literals) are being analyzed. For each valid relationship, it checks the classes of the two connected resources. If they belong to different classes, " 
            "both classes receive a \"connectivity point\". The final metric is the average number of these points across all classes, which shows how strongly the classes in the ontology are interconnected.",

    "cohesion": "**Cohesion** ([Reference](https://link.springer.com/chapter/10.1007/978-3-642-11829-6_19)):  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp; This metric measures the number of disconnected subgraphs (connected components) within the RDF graph. \n"
            " Each component represents a group of nodes that are mutually reachable through RDF triples (either as subject or object).\n\n"
            " The algorithm starts with an arbitrary unvisited node and explores all nodes reachable from it by following both incoming and outgoing edges."
            " When no new nodes can be reached, that group of nodes forms one component."
            " The process repeats with the next unvisited node until all nodes have been explored.\n\n"
            " The cohesion value equals the total number of such disconnected components — the fewer components, the fewer the cohesion value and the higher the graph's cohesion.\n\n",
}

# mapping: checkbox -> tooltip text (endpoint version)
METRIC_HELP_ENDPOINT = {
    "paths_depth":  "In the context of an RDF graph, a path is a sequence of nodes connected by edges. Nodes correspond to the subjects and objects of RDF triples, while the predicate "
                    " of a triple represents the directed edge from the subject to the object.\n\n"

                    " A \"root\" is defined as any node that does not appear as an object in any triple, ensuring that traversals start at the top-level nodes of the graph structure. "
                    "The traversal is performed using a depth-first search (DFS) algorithm.\n\n"

                    "To calculate these metrics, several SPARQL queries are used to extract the graph structure before local traversal:  \n\n"
                       
                    "&nbsp;&nbsp;&nbsp;&nbsp;• Literals are retrieved once using `FILTER isLiteral(?literal)` and stored locally. "
                    "This avoids repeated queries for large literal values, which could cause endpoint errors. Literals are treated as terminal nodes during traversal.  \n\n"
                    
                    "&nbsp;&nbsp;&nbsp;&nbsp;• Blank Nodes (BNodes) and their neighbors are collected in a single query. "
                    "That query identifies all triples where a blank node occurs and optionally includes both its incoming and outgoing links. "
                    "Because blank nodes have no persistent identifiers (e.g., `_b1` can represent different entities in separate queries), "
                    "their neighbor relationships are cached locally in two sets: one for cases where the bnode acts as a subject and one for cases where it appears as an object. "
                    "Within this same query, it is also detected whether a blank node functions as a root node — namely, when it has outgoing neighbors "
                    "but no incoming ones.\n\n"
                    
                    "&nbsp;&nbsp;&nbsp;&nbsp;• Root Nodes (including BNodes) are thus determined in two ways:  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Directly within the BNode neighbor query as mentioned above.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Through a separate query selecting all non-blank subjects that never appear as an object.  \n\n"
                    
                    "This approach reconstructs the structural hierarchy of a remote RDF graph reliably, even when blank nodes or large literal values "
                    "make direct endpoint traversal difficult.\n\n"

                    "**Number of Paths** ([Reference](https://ieeexplore.ieee.org/document/4031647)):  \n &nbsp;&nbsp;&nbsp;&nbsp;The total count of all unique paths starting from a "
                    "root node and ending at a leaf node (a node with no outgoing edges or a literal).\n\n"
                    
                    "**Absolute Depth** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/isaf.1360)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The sum of the lengths (in edges) of all identified paths.\n\n"
                    
                    "**Average Depth** ([Reference](https://link.springer.com/chapter/10.1007/11762256_13)):  \n &nbsp;&nbsp;&nbsp;&nbsp;  The average length of a path (Absolute Depth / Number of Paths)\n\n"
                    
                    "**Maximum Depth** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/isaf.1360)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The length of the longest path found in the graph",

    "ont_tangledness":  "**Ontology Tangledness** ([Reference](https://www.scitepress.org/Link.aspx?doi=10.5220/0004541400480057)):  \n"
                        "&nbsp;&nbsp;&nbsp;&nbsp;The ratio of total classes (criteria for a class: see **Number of Classes**) to classes"
                        " with more than 1 superclass (`rdfs:subClassOf`).\n\n",

    "degree_variance":  "**Degree Variance** ([Reference](https://link.springer.com/article/10.1134/S1064230711010072)):  \n"
                        " &nbsp;&nbsp;&nbsp;&nbsp; This metric measures how much the connectivity of nodes in an RDF graph deviates from the average.\n\n"
                        " The degree of a node is defined as the total number of incoming and outgoing edges."
                        " The mean degree μ is calculated as (2 x number of edges) / (number of nodes), since each edge contributes to two node degrees."
                        " The Degree Variance is then computed as the average squared deviation of each node's degree from this mean:  \n"
                        " Var(d) = Σ((dᵢ - μ)²) / (nG - 1), where dᵢ is the degree of node *i* and nG the number of nodes.\n\n"
                        " A higher variance indicates a more uneven edge distribution (some nodes are highly connected while others are isolated), "
                        "whereas a lower variance reflects a more balanced and uniform graph structure.\n\n",
    
    "primitives":   "**Number of Entities** ([Reference](https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-018-0188-7)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The total count of unique resources (URIs and BNodes) that appear as a subject or a non-literal object in any triple.\n\n"
                    
                    "**Number of Properties** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/smr.341)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The sum of two counts:  \n"
                    " &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.  Properties explicitly declared as `owl:ObjectProperty`, `owl:DatatypeProperty`, or `RDF.Property` (T-Box).  \n "
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.  All unique predicates used in the graph (A-Box). \n\n"
                    
                    "**Number of Classes** ([Reference](https://ieeexplore.ieee.org/document/4031647)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The count of classes. The criteria for identifiying a class:  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.  Any resource that is the object of an `rdf:type` triple.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.  Both the subject and object of an `rdfs:subClassOf` / `owl:equivalentClass` triple.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.  The subject of a triple declaring an `owl:Restriction`.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.  The subject of complex class definitions using `owl:unionOf`, `owl:intersectionOf`, `owl:complementOf` or `owl:oneOf`.  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.  The subject of an `owl:hasValue` restriction.  \n\n"
                    
                    "**Number of Instances** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp;" 
                    "Count of all individual instances in the graph.  Specifically, this is the number of unique subjects in triples of the form `(s, rdf:type, o)` "
                    " where `s` is not itself a class. In other words, classes (`rdf:type` objects) are excluded from the count to ensure only individuals are counted. \n\n"
                    
                    "**Number of Object Properties** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/isaf.1360)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The sum of two counts:  \n"
                    " &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Properties explicitly declared as `owl:ObjectProperty` (T-Box).  \n"
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. All unique predicates in the rdf triples that have a non-literal as an object (A-Box). ",

    "depth_of_inheritance_tree": "**Depth of Inheritance Tree** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/smr.341)):  \n &nbsp;&nbsp;&nbsp;&nbsp; "   
                                 "This metric measures the maximum depth (= number of edges in longest path) of the class hierarchy based on `rdfs:subClassOf` relationships.\n\n"
                                 "The calculation starts by identifying all root classes (classes that are not a subclass of any other class)."
                                " From each root, the hierarchy is traversed downwards using a depth-first search (DFS) algorithm to find the longest path to a leaf class (a class with no subclasses).  \n\n"
                                "To determine this hierarchy remotely, the function first reconstructs the class structure using SPARQL queries before performing a local traversal:  \n\n"
                              
                              "&nbsp;&nbsp;&nbsp;&nbsp;• Blank Nodes (BNodes):  \n"
                              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A single query retrieves all blank nodes involved in `rdfs:subClassOf` relations. "
                              "For each bnode, both incoming and outgoing links are collected, "
                              "ensuring that all subclass/superclass connections are captured despite the lack of stable identifiers. "
                              "Within this same query, each bnode is also checked for whether it acts as a root node (i.e., a class without a superclass). "
                              "This allows the detection of bnode roots directly during retrieval.  \n\n"
                              
                              "&nbsp;&nbsp;&nbsp;&nbsp;• Named Root Classes:  \n"
                              "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In addition to blank nodes, all named classes (`owl:Class` or `rdfs:Class`) "
                              "that do not appear as a subclass in any triple are identified.  \n\n"

                              "This method ensures that both named and anonymous classes are properly considered, "
                              "accurately reconstructing the full inheritance hierarchy even when blank nodes or distributed subclass relationships "
                              "make direct traversal through the endpoint unreliable.",
    "tbox": "**Property/Class-Ratio** ([Reference](https://onlinelibrary.wiley.com/doi/10.1002/smr.341)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The ratio of explicitly declared T-Box properties (`owl:ObjectProperty`, `owl:DatatypeProperty`, `RDF.Property`) to the total number of classes (criteria for a class: see **Number of Classes**). \n\n"
            
            "**Class/Property-Ratio** ([Reference](https://www.scitepress.org/Link.aspx?doi=10.5220/0004541400480057)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The inverse of the Property Class Ratio, showing the ratio of classes to properties. \n\n"
            
            "**Inheritance Richness** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The ratio of the number of subclass relationships (`rdfs:subClassOf`) to the total number of classes (criteria for a class: see **Number of Classes**).  \n\n"
            
            "**Attribute Richness** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp; The ratio of explicitly declared datatype properties (`owl:DatatypeProperty`) to the total number of classes. (criteria for a class: see **Number of Classes**)",
    
    "abox": "**Average Population** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n &nbsp;&nbsp;&nbsp;&nbsp;The average number of instances per class in the ontology.  \n It is calculated by dividing the total number of instances (see **Number of Instances**) by the total number of classes (criteria for a class: see **Number of Classes**).\n\n"
            
            "**Average Class Connectivity** ([Reference](https://link.springer.com/chapter/10.1007/978-90-481-8847-5_5)):  \n "
            "&nbsp;&nbsp;&nbsp;&nbsp; The average number of relationships that instances of a class have with instances of other classes.\n\n" 
            "The computation of the Average Class Connectivity (and of course for the Average Population, too) is performed directly on the endpoint using a SPARQL query. "
            "For every triple `(instance → property → target)`, the query checks the classes of both resources via `rdf:type` "
            "and counts a connection whenever the two belong to different classes. "
            "All `rdf:type` statements are excluded from this analysis, as well as relationships involving literals.  \n\n"
            
            "The query counts connections in both directions — once where the subject's class links to the object's class, "
            "and once vice versa — ensuring that class-level connectivity is treated symmetrically. "
            "Each occurrence increases the connectivity score of both involved classes.  \n\n"
            
            "The final Average Class Connectivity value is obtained by dividing the total number of such inter-class connections "
            "by the total number of classes in the ontology. "
            "A higher value indicates that instances of different classes frequently interact, "
            "signifying a highly connected A-Box, while a lower value implies a more isolated or weakly linked class structure."
            , 
    "cohesion": "**Cohesion** ([Reference](https://link.springer.com/chapter/10.1007/978-3-642-11829-6_19)):  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp; This metric measures the number of disconnected subgraphs (connected components) within the RDF graph. \n"
            " Each component represents a group of nodes that are mutually reachable through RDF triples (either as subject or object).\n\n"
            " The algorithm starts with an arbitrary unvisited node and explores all nodes reachable from it by following both incoming and outgoing edges."
            " When no new nodes can be reached, that group of nodes forms one component."
            " The process repeats with the next unvisited node until all nodes have been explored.\n\n"
            "The cohesion value is the total number of such components. "
            "A single connected component (value = 1) indicates a highly cohesive graph, "
            "while multiple components signify fragmentation or disconnected clusters within the data.  \n\n"
            "To compute this metric remotely, the algorithm reconstructs the graph structure step by step using SPARQL queries and local caching:  \n\n"
                    
            "&nbsp;&nbsp;&nbsp;&nbsp;• Node and Neighbor Retrieval:  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;All RDF nodes (subjects and objects) are first retrieved in a single query, "
            "which also collects their immediate neighbors. This ensures that both directions of RDF relationships are considered "
            "and that all node types (URIs, Blank Nodes, Literals) are included in the node set.  \n\n"
            
            "&nbsp;&nbsp;&nbsp;&nbsp;• Blank Node and Literal Handling:  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Because Blank Nodes are not globally identifiable and Literals can be complex or large, "
            "their relationships are extracted once and stored locally when retrieving the nodes with their neighbors from the first query. Two local structures are maintained: "
            "Bnodes and their neighbors and literals with their neighbors. "
            "These cached mappings are reused when exploring the graph, avoiding repeated endpoint queries.  \n\n"
                    
            "This endpoint-based approach allows accurate structural analysis even when blank nodes or large literal values "
            "make direct traversal through the SPARQL interface unreliable."
}


def metric_keys_to_run(selected_key: str) -> list[str]:
    
    if selected_key == "ALL":
        # Return all metrics except "ALL" if the user selects "ALL"
        return [k for k in METRICS.keys() if k != "ALL"]
    # Return the selected key as the only metric
    return [selected_key]

# Streamlit page title
st.title("Metric Exploration for Evolving Knowledge Graphs")

left, right = st.columns([5, 5])  

with left:

    st.header("Metric Computation")

    #st.subheader("Metric Calculation")
    # User interface for selecting the source of the data (either uploading a file or providing a SPARQL endpoint URL)
    source = st.radio("Source", ["Local RDF-File", "SPARQL Endpoint"], horizontal=True)

    if source == "Local RDF-File":
        # If the user selects "Local RDF-File", they are prompted to upload an RDF file
        uploaded = st.file_uploader("RDF-File")
    else:
        # If the user selects "SPARQL Endpoint", a warning is shown about potential issues with SPARQL endpoints
        st.warning("""Note: Be aware that some SPARQL Endpoints do not return the full results 
                    and have Timeouts which may result in incorrect or unavailable Metrics.""", icon="⚠️")
        
        # Text inputs for entering the SPARQL endpoint URL and an optional default graph IRI
        endpoint_url  = st.text_input("Endpoint URL", placeholder="https://example.org/sparql")
        default_graph = st.text_input("Default Graph IRI (optional)")

    # Streamlit title for Metrics section
    st.subheader("Metrics")


    # dropdown menu (METRICS[k] means it should show the value/description of k, not the key k)
    # metric_key = st.selectbox("Metric", list(METRICS.keys()), format_func=lambda k: METRICS[k])

    # Session state for metric selections
    for key in METRICS.keys():
        if key not in st.session_state:
            st.session_state[key] = False

    # User interface for selecting metrics (Select all/none buttons)
    left_2, right_2, _ = st.columns([2, 2, 5])  
    with left_2:
        if st.button("Select all"):
            for key in METRICS.keys():
                st.session_state[key] = True
    with right_2:
        if st.button("Select none"):
            for key in METRICS.keys():
                st.session_state[key] = False

    # Checkbox for each metric
    selected_metrics = {}
    for key, description in METRICS.items():
        
        if source == "Local RDF-File":
            # Display checkbox for the metric if the source is a local RDF file, with help text from `METRIC_HELP_LOCAL`
            selected_metrics[key] = st.checkbox(
                description, 
                key=key, 
                help=METRIC_HELP_LOCAL.get(key, "")
            )
        else:
            # Display checkbox for the metric if the source is a SPARQL endpoint, with help text from `METRIC_HELP_ENDPOINT`
            selected_metrics[key] = st.checkbox(
                description, 
                key=key, 
                help=METRIC_HELP_ENDPOINT.get(key, "")
            )

    # button to start calculation
    run = st.button("Calculate", type="primary")

    def run_nb(nb_path: Path, params: dict):
        """
        Executes the specified notebook with the given parameters.
        """
        
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            executed_nb = tmp / "run.ipynb"
            out_csv = tmp / "metrics.csv"
            params["output_csv"] = str(out_csv)


            # if local file (creates source path in paramaters)
            if "uploaded" in params:
                # path for tmp file
                # params["uploaded"].name = name of uploaded file
                in_path = tmp / params["uploaded"].name
                # copies content of uploaded file to tmp file
                in_path.write_bytes(params["uploaded"].getbuffer())
                params["source_path"] = str(in_path)
                del params["uploaded"]
            
            # Execute the notebook with the parameters
            pm.execute_notebook(
                input_path=str(nb_path),
                output_path=str(executed_nb),
                parameters=params,
                cwd=str(BASE), # working directory
                kernel_name="metrikg-venv",
                progress_bar=True,
            )

            df = pd.read_csv(out_csv) if out_csv.exists() else None
            
            # should not be the case
            if df is None:
                nb = sb.read_notebook(str(executed_nb))
                if "metrics_table" in nb.scraps:
                    df = nb.scraps["metrics_table"].data

            return df#, executed_bytes

    # Stores checkbox status for each metric
    keys_to_run = [k for k, checked in selected_metrics.items() if checked]

    # Creating space for subheader, download-button and enlarge/reduce-button
    left_2, middle, right_2 = st.columns([4, 2, 1])  

    # This checks if the calculation should run (or if already in progress)
    if run:
        # Setting session state flags for calculation
        st.session_state.calculating = True
        st.session_state.first_run_made = True

    # Setting the page title in Streamlit
    with left_2:
        if st.session_state.first_run_made == True:
            #st.title("Knowledge Graph Metrics")
            st.subheader("Metric Values")

    # This checks if the calculation should run (or if already in progress)
    if run:
        st.session_state.calculating = True
        st.session_state.first_run_made = True
        
        try:
            # Check if at least one metric is selected
            if not keys_to_run:
                st.warning("Please select at least one metric.")

            if source == "Local RDF-File":
                # Case when the source is a local RDF file
                if not uploaded:
                    # Show an error if the file is not uploaded
                    st.error("Missing uploaded File.")
                else:
                    with st.status("Calculating metrics …", expanded=False) as status:
                        # Loop through the selected metrics and calculate them
                        for i, k in enumerate(keys_to_run, start=1):

                            # shows progress
                            status.update(label=f"[{i}/{len(keys_to_run)}] {METRICS[k]}")
                            st.session_state.last_status = f"[{i}/{len(keys_to_run)}] {METRICS[k]}"

                            try:
                                # Run the notebook to calculate the current metric
                                df = run_nb(NB_FILE, {"metric_key": k, "uploaded": uploaded})
                                # Set the source value as the uploaded file name
                                source_value = uploaded.name 

                            except Exception as e:
                                # Raise an error if there is an issue with the metric calculation
                                error_message = f"Error when calculating metric({METRICS[k]}): \n\n{e}"
                                raise RuntimeError(error_message)  # Raise the simplified error
                                
                            if df is None or df.empty:
                                raise RuntimeError(f"{METRICS[k]}: No result")

                            # Update the results DataFrame with the calculated values
                            update_results_df(df, source_label="file", source_value=source_value)

                        # Notify that calculation is done
                        st.success("Calculation done.")
                        st.session_state.full_table = False

                    # Filter the metrics based on user selection
                    keys_to_show = []

                    for k, checked in selected_metrics.items():
                        if checked:
                            if k in CATEGORY_TO_METRICS:
                                # Add all metrics from the selected category (if checkbox consists of many metrics)
                                keys_to_show.extend(CATEGORY_TO_METRICS[k])
                            else:
                                # Add the individual metric (if checkbox consists of one metric)
                                keys_to_show.append(k)

                    # Filter the results DataFrame to show only selected metrics
                    df_to_show = st.session_state.results_df[
                        st.session_state.results_df["Metric"].isin(keys_to_show)
                    ].copy()

                    # Save the filtered results DataFrame to the session state
                    st.session_state["df_to_show"] = df_to_show

            else:
                # Case when the source is a SPARQL endpoint
                # Clean the inputs
                endpoint_url = clean_url(endpoint_url)
                default_graph = clean_iri(default_graph)

                if not endpoint_url:
                    # Error if the endpoint URL is not provided
                    st.error("Missing Endpoint URL.")

                elif not is_valid_endpoint_url(endpoint_url):
                    # Error if the URL is invalid
                    st.error("Invalid Endpoint URL")

                elif default_graph and not is_valid_iri(default_graph):
                    # Error if the default graph IRI is invalid
                    st.error("Invalid Default Graph IRI")

                else:
                                
                    with st.status("Calculating metrics …", expanded=False) as status:
                        # Loop through the selected metrics and calculate them
                        for i, k in enumerate(keys_to_run, start=1):
                            status.update(label=f"[{i}/{len(keys_to_run)}] {METRICS[k]}")
                            
                            try:
                                # Run the notebook to calculate the current metric for the SPARQL endpoint
                                df = run_nb(NB_EP, {
                                    "metric_key": k,
                                    "endpoint_url": endpoint_url,
                                    "default_graph": default_graph
                                })
                                # Set the source value based on the endpoint and default graph
                                if default_graph :
                                    source_value = "EP: " + endpoint_url + "  DG: " + default_graph
                                else:
                                    source_value = "EP: " + endpoint_url
                            except Exception as e:
                                # Raise an error if there is an issue with the metric calculation
                                error_message = f"ENDPOINT ERROR ({METRICS[k]}): \n{e}"
                                raise RuntimeError(error_message)  

                            if df is None or df.empty:
                                raise RuntimeError(f"{METRICS[k]}: Empty Result")
                            # Update the results DataFrame with the calculated values
                            update_results_df(df, source_label="endpoint", source_value=source_value)

                        # Notify that calculation is done
                        st.success("Calculation done.")
                        st.session_state.full_table = False

                    # Filter the metrics based on user selection
                    keys_to_show = []

                    for k, checked in selected_metrics.items():
                        if checked:
                            
                            if k in CATEGORY_TO_METRICS:
                                # Add all metrics from the selected category (if checkbox consists of many metrics)
                                keys_to_show.extend(CATEGORY_TO_METRICS[k])
                            else:
                                # Add the individual metric (if checkbox consists of one metric)
                                keys_to_show.append(k)

                    # Filter the results DataFrame to show only selected metrics
                    df_to_show = st.session_state.results_df[
                        st.session_state.results_df["Metric"].isin(keys_to_show)
                    ].copy()

                    # Save the filtered results DataFrame to the session state
                    st.session_state["df_to_show"] = df_to_show

            st.session_state.calculating = False

        except Exception as e:
            # If an error occurs during calculation, show an error message
            st.error(f"Metric Computation threw an error: {str(e)}")


    # Displaying the results if calculation is complete
    if ("df_to_show" in st.session_state) and st.session_state.calculating == False:
        # Create columns for the table display
        left_2, middle = st.columns([1,8])

        # Set table height based on whether the full table is displayed
        table_height = 700 if st.session_state.full_table else "auto"
        #table_height = 720
        #table_height = "auto"

        # Display the results DataFrame as a table
        st.dataframe(
            st.session_state["df_to_show"],
            width="stretch",
            height=table_height,
            hide_index=True # Hide the row index for a cleaner view
        )

        snapshot_json = st.session_state["df_to_show"].to_json(orient="records")

        # for saving results in browser-memory
        if st.button("Save Metric Values to Browser-Memory", type="primary"):
            unique = time.time()
        #     # invisible element to be rendered
            # components.html(
            #     f"""
            #     <script>
            #         const entry = {{
            #             timestamp: new Date().toISOString(),
            #             data: {snapshot_json}
            #         }};
            #         let history = JSON.parse(localStorage.getItem("metrikg_history") || "[]");
            #         history.push(entry);
            #         localStorage.setItem("metrikg_history", JSON.stringify(history));
            #         alert("Results saved in Browser-Memory.");
            #     </script>
            #     """,
            #     height=0,   # invbisible
            # )

            components.html(
                f"""
                <script>
                    console.log("save_snapshot_{unique}");
                    const entry = {{
                        timestamp: new Date().toISOString(),
                        data: {snapshot_json}
                    }};
                    let history = JSON.parse(localStorage.getItem("metrikg_history") || "[]");
                    history.push(entry);
                    localStorage.setItem("metrikg_history", JSON.stringify(history));
                    alert("Metric Values saved to Browser-Memory.");
                </script>
                """,
                height=0
            )

with right:
    st.header("Metric History Visualization")

    uploaded_file = st.file_uploader(
        "CSV File containing metric data",
        type=["csv"]
    )

    if uploaded_file is not None:
        # read CSV file 
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

        # check columns
        expected_cols = ["Metric", "Value", "Source", "Run At"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV file: {missing}")
            st.stop()

        # parse timestamp and value
        df["Run At"] = pd.to_datetime(
            df["Run At"],
            dayfirst=True,      # für 17.11.2025 15:58:51
            errors="coerce"
        )

        # for debugging
        # st.write("Zeilen mit ungültigem Datum:")
        # st.write(df[pd.to_datetime(df["Run At"], dayfirst=True, errors="coerce").isna()])

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Run At", "Value"])

        # get metrics
        metrics = sorted(df["Metric"].unique())

        # no selectbox if just one metric in file
        if len(metrics) == 1:
            selected_metric = metrics[0]
            st.write(f"Found only metric: **{selected_metric}**")
        else:
            selected_metric = st.selectbox("Select metric", metrics)

        metric_df = df[df["Metric"] == selected_metric].copy()
        metric_df = metric_df.sort_values("Run At")

        # String-Spalte für die Achsenlabels (nur dort gibt es auch Werte)
        #metric_df["Run At Label"] = metric_df["Run At"].dt.strftime("%d.%m.%Y %H:%M:%S")
        ### metric_df["Run At Label"] = metric_df["Run At"].dt.strftime("%d.%m.%Y")

        st.subheader(f"Visualization of: {selected_metric}")

        # Tage mit Messungen für die Tick-Positionen (Mitternacht-normalisiert)
        tick_days = (
            metric_df["Run At"]
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        chart = (
            alt.Chart(metric_df)
            .mark_line(point=True)
            .encode(
                # diskrete Achse: ein Tick pro vorhandenem Messzeitpunkt
                x=alt.X(
                    ###"Run At Label:N",
                    "Run At:T",
                    title="Timestamp",
                    axis=alt.Axis(
                        format="%d.%m.%Y", # nur Datum anzeigen
                        values=list(tick_days),
                        labelFontSize=16,
                        titleFontSize=16,
                        labelAngle=-45,
                        labelOverlap=False,
                        ) 
                ),
                y=alt.Y(
                    "Value:Q", 
                    title="Value",
                    axis=alt.Axis(
                        labelFontSize=16,
                        titleFontSize=16,
                        ) 
                    ),
                tooltip=[
                    ### alt.Tooltip("Run At:T", title="Timestamp"),
                    alt.Tooltip(
                        "Run At:T",
                        title="Timestamp",
                        format="%d.%m.%Y %H:%M:%S",
                        #format="%d.%m.%Y",
                    ),

                    alt.Tooltip("Value:Q", title="Value"),
                    alt.Tooltip("Source:N", title="Source")
                ]
            )
            .properties(width="container", height=400)
        )

        st.altair_chart(chart, use_container_width=True)
