"""
Utility module responsible for loading and accessing RDF data.
"""

from rdflib import Graph
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

def load_graph_from_file(file_or_path) -> Graph:
    """
        Loads RDF data from a given path.
    """
    g = Graph()

    """# case (b): UploadedFile (hat 'getvalue' und 'name')
    if hasattr(file_or_path, "getvalue") and hasattr(file_or_path, "name"):
        suffix = Path(file_or_path.name).suffix.lower()
        #fmt = EXT2FMT.get(suffix)
        try:
            g.parse(data=file_or_path.getvalue())
            #g.parse(data=file_or_path.getvalue(), format=fmt)
        except Exception as e:
            raise RuntimeError(f"Error when parsing uploaded file '{file_or_path.name}': {e}")
        return g"""


    file_path = Path(file_or_path) # Converting file_or_path to a Path object
    suffix = file_path.suffix.lower()
    #fmt = EXT2FMT.get(suffix)
    try:
        #g.parse(file_path.as_posix(), format=fmt)
        g.parse(file_path.as_posix()) # Parsing the RDF data from the file path (no format specified)

    except Exception as e:
        # Raising an error if parsing fails.
        raise RuntimeError(f"Error when parsing file '{file_or_path}': {e}")
    
    return g

def get_sparql_from_endpoint(endpoint_url: str, default_graph: str = None):
    """
        Creates and returns a SPARQLWrapper object connected to a specified endpoint.
        Optionally adds a default graph to the query if provided.
        
        Arguments:
        - endpoint_url (str): The URL of the SPARQL endpoint.
        - default_graph (str, optional): A default graph to use in the SPARQL query (optional).
        
        Returns:
        - sparql (SPARQLWrapper): A SPARQLWrapper object configured to interact with the endpoint.
    """
    # Creating a SPARQLWrapper instance to interact with the endpoint.
    sparql = SPARQLWrapper(endpoint_url)
    
    # Adding the default graph to the SPARQL query if provided.
    if default_graph:
        sparql.addDefaultGraph(default_graph)
    
    # Setting the return format of the query results to JSON.
    sparql.setReturnFormat(JSON)
    
    return sparql