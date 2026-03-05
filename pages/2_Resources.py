import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Resources", layout="wide")

st.title("Resources")

st.markdown("""
## Sample KGs for Metric Computation

The following publicly available RDF datasets can be used to experiment with the metric computation functionality of MetriKG.

Please note that for larger files, the computation of certain metrics may take a significant amount of time.
""")

st.download_button(
    "Download: W3C — Tim Berners-Lee FOAF profile ( ~8KB RDF/XML File)",
    open("resources/foaf_tim_berners_lee.rdf","rb").read(),
    file_name="foaf_tim_berners_lee.rdf",
    mime="application/rdf+xml",
)

st.download_button(
    "Download: FOAF Vocabulary Specification (FOAF ontology) ( ~43KB RDF/XML File)",
    open("resources/foaf_ontology.rdf","rb").read(),
    file_name="foaf_ontology.rdf",
    mime="application/rdf+xml",
)

st.download_button(
    "Download:  W3C OWL Guide — Wine ontology example dataset ( ~76KB RDF/XML File)",
    open("resources/wine_ontology.rdf","rb").read(),
    file_name="wine_ontology.rdf",
    mime="application/rdf+xml",
)

st.download_button(
    "Download: DBpedia — resource “Vienna” ( ~2MB Turtle File )",
    open("resources/dbpedia_vienna.ttl","rb").read(),
    file_name="dbpedia_vienna.ttl",
    mime="text/turtle"
)

st.markdown("""
## Sample CSV file for Metric History Visualization

The following example CSV file containing metric values for several KG versions can be used to explore the metric history visualization functionality of MetriKG.       
""")

st.download_button(
    "Download: LOV - schema ( ~6KB CSV File )",
    open("resources/lov_schema_metric_history.csv","rb").read(),
    file_name="lov_schema_metric_history.csv",
    mime="text/csv",
)

