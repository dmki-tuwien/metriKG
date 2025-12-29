"""
Contains the actual implementations of metric computations for RDF graphs.
Each function calculates one or more Knowledge Graph metrics (e.g., Depth, Degree Variance, Cohesion).
Used by the Jupyter notebooks to compute metric values and export them as CSV tables.
"""


import pandas as pd
from rdflib import Graph, Literal, URIRef, BNode, RDF, OWL, RDFS
from SPARQLWrapper import SPARQLWrapper, JSON
from graph_loader import get_sparql_from_endpoint

############## LOCAL FILE FUNCTIONS ##############

def _get_classes_local(g: Graph):
    """
    Identifies and collects all unique classes from an RDFLib Graph.

    This function traverses the graph and identifies classes based on various
    RDF, RDFS, and OWL constructs that define or imply class membership.
    It aggregates all findings into a single set to ensure uniqueness.

    The criteria for identifying a class are:
    1.  Any resource that is the object of an `rdf:type` triple.
    2.  Both the subject and object of an `rdfs:subClassOf` triple.
    3.  Both the subject and object of an `owl:equivalentClass` triple.
    4.  The subject of a triple declaring an `owl:Restriction`.
    5.  The subject of complex class definitions using `owl:unionOf`, 
        `owl:intersectionOf`, `owl:complementOf`, or `owl:oneOf`.
    6.  The subject of an `owl:hasValue` restriction.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).

    Returns:
        set: A set containing the class resources found in the graph.
    """
    classes = set()

    # 1. all objects of rdf:type
    for s, p, o in g.triples((None, RDF.type, None)):
        classes.add(o)

    # 2. subclasses (left side)
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        classes.add(s)
    # 3. superclasses
        classes.add(o)

    # 4. owl:equivalentClass
    for s, p, o in g.triples((None, OWL.equivalentClass, None)):
        classes.add(s)
        classes.add(o)

    # 5. owl:Restriction
    for s, p, o in g.triples((None, RDF.type, OWL.Restriction)):
        classes.add(s)

    # 6. complex classes (unionOf, intersectionOf, complementOf, oneOf)
    for prop in [OWL.unionOf, OWL.intersectionOf, OWL.complementOf, OWL.oneOf]:
        for s, p, o in g.triples((None, prop, None)):
            classes.add(s)

    # 7. owl:hasValue
    for s, p, o in g.triples((None, OWL.hasValue, None)):
        classes.add(s)

    return classes

def _get_num_instances_local(g: Graph) -> int:
    """
    Calculates the number of instance resources (ABox individuals) 
    in a local RDF graph.

    This function identifies all resources that appear as the subject 
    in an `rdf:type` triple. To ensure that only true individuals (ABox 
    resources) are counted, the function excludes all resources that also appear 
    as objects of an `rdf:type` triple, since those represent classes (TBox elements).

    The resulting number reflects the total count of RDF resources 
    that are explicitly used as instances but not as classes within the graph.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).

    Returns:
        int: The number of unique instances (individuals) in the graph.
    """
    num_instances = 0

    instances = {s for s in g.subjects(RDF.type, None)}

    #classes = {o for o in g.objects(None, RDF.type)}

    classes = _get_classes_local(g)

    # we just want the individuals
    individuals = instances - classes 

    num_instances = len(individuals)

    return num_instances

def paths_depth_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates path and depth-related metrics for a given RDF graph.

    In the context of an RDF graph, a path is a sequence of nodes connected by edges.
    Nodes correspond to the subjects and objects of RDF triples, while the predicate
    of a triple represents the directed edge from the subject to the object.

    This function traverses the graph to compute four metrics:
    - **Number of Paths**: The total count of all unique paths starting from a
      root node and ending at a leaf node (a node with no outgoing edges or a literal).
    - **Absolute Depth**: The sum of the lengths (in edges) of all identified paths.
    - **Average Depth**: The average length of a path (Absolute Depth / Number of Paths).
    - **Maximum Depth**: The length of the longest path found in the graph.

    A "root" is defined as any node that does not appear as an object in any triple,
    ensuring that traversals start at the top-level nodes of the graph structure.
    The traversal is performed using a depth-first search (DFS) algorithm.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): The number of decimal places for rounding the
                                    average depth. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame with the calculated metrics: "Number of Paths",
                      "Absolute Depth", "Average Depth", and "Maximum Depth".
    """

    num_paths = 0
    abs_depth = 0
    max_depth = 0

    # cache for storing neighbors of nodes
    neighbors_cache = {}

    def get_neighbors(node):
        """
        Retrieves the neighbors of a given node from the graph.

        This function finds all objects connected to the given `node` (as a subject)
        in the RDF graph. It uses a cache (`neighbors_cache`) to store and retrieve
        results for previously seen nodes, improving performance by avoiding
        redundant graph lookups. If the node is an RDF `Literal`, it is considered
        to have no neighbors and an empty list is returned.

        Args:
            node: The RDF node for which to find neighbors.

        Returns:
            list: A list of neighbor nodes. Returns an empty list if the node is a
                  Literal or has no neighbors.
        """
        if isinstance(node, Literal):
            return []
        
        if node in neighbors_cache:
            return neighbors_cache[node]
        
        neighbors = list(g.objects(subject=node))
        neighbors_cache[node] = neighbors

        return neighbors

    def dfs(node, path):
        """
        Performs a depth-first search (DFS) to find all paths from a node.

        This recursive function explores paths starting from the given `node`.
        It avoids cycles by checking if a node has already been visited in the
        current `path`. When a path terminates (i.e., a node with no outgoing
        neighbors is reached), it updates the global metrics: `num_paths`,
        `abs_depth`, and `max_depth`.

        Args:
            node: The current node to visit in the DFS traversal.
            path (list): The list of nodes representing the current path from the
                         root to the parent of the current `node`.
        """
        nonlocal num_paths, abs_depth, max_depth  

        if node in path:
            return

        path.append(node)
        neighbors = get_neighbors(node)

        if not neighbors:
            # path length = number of edges in a path = number of nodes in path - 1
            path_length = len(path) - 1
            num_paths += 1
            abs_depth += path_length

            if path_length > max_depth:
                max_depth = path_length

        else:
            for n in neighbors:
                dfs(n, path)

        # remove node from the path to find next path
        path.pop()

    # classes = _get_classes_local(g)
    
    # # owl_classes  = set(g.subjects(RDF.type, OWL.Class))
    # # rdfs_classes = set(g.subjects(RDF.type, RDFS.Class))
    # # classes = owl_classes | rdfs_classes    
    
    # roots = [c for c in classes if not any(True for _ in g.objects(c, RDFS.subClassOf))]

    all_nodes = set(g.subjects()) | set(g.objects()) 
    object_nodes = set(g.objects()) 
    roots = all_nodes - object_nodes

    for r in roots:
        dfs(r, [])

    if num_paths > 0:
        avg_depth = round(abs_depth / num_paths, dec_places) 
    else:
        avg_depth = 0.0

    return pd.DataFrame([
        {"Metric": "Number of Paths", "Value": int(num_paths)},
        {"Metric": "Absolute Depth", "Value": int(abs_depth)},
        {"Metric": "Average Depth", "Value": float(avg_depth)},
        {"Metric": "Maximum Depth", "Value": int(max_depth)},
    ])

def ont_tangledness_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the Ontology Tangledness metric for a given RDF graph.

    Ontology Tangledness is defined as the ratio of the total number of classes to the number of classes
    that have more than one superclass ( = multiple `rdfs:subClassOf` relationships). If there are no tangled 
    classes, the metric is 0.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): Number of decimal places to round the result to. Defaults to 2.
    Returns:
        pd.DataFrame: A DataFrame containing the metric name ("Ontology Tangledness") and its value.
    """

    # getting distinct classes
    classes = _get_classes_local(g)

    num_classes = len(classes)

    # fpr counting superclasses of nodes
    superclass_counts = {}

    for class_name, _, _ in g.triples((None, RDFS.subClassOf, None)):
        superclass_counts[class_name] = superclass_counts.get(class_name, 0) + 1

    # all classes with more than one superclass
    tangled_classes = {class_name for class_name, count in superclass_counts.items() if count > 1}

    t = len(tangled_classes)

    if t > 0:
        # source 37 says num_classes / t
        ont_tangledness = round(num_classes / t, dec_places) 
    else:
        ont_tangledness = 0.0

    return pd.DataFrame([
        {"Metric": "Ontology Tangledness", "Value": float(ont_tangledness)},
    ])

def degree_variance_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the degree variance of a given RDF graph.

    The degree variance is computed as the variance of the degree distribution,
    where the degree of a node is the sum of its incoming and outgoing edges. It quantifies 
    how unevenly the connections (edges) in a graph are distributed among its nodes.
    The result is returned as a pandas DataFrame containing the metric name and its value.

    First, the mean degree μ is computed as (2 x number of edges) / (number of nodes),
    since each edge contributes to two node degrees. The variance is then obtained by
    averaging the squared deviations of all node degrees from this mean:
        Var(d) = Σ((dᵢ - μ)²) / (nG - 1)
    where dᵢ is the degree of node i and nG the number of nodes.

    A higher degree variance indicates that some nodes are much more connected than others 
    (less uniform structure), while a lower variance means the graph's connectivity is more balanced.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): Number of decimal places to round the variance to. Default is 2.
    Returns:
        pd.DataFrame: A DataFrame with one row containing the metric name ("Degree Variance") and its value.
    """
    # nG...number of nodes in gaph
    # nE...number of edges in graph

    # Calculating nE
    nE = len(g)

    all_nodes = set(g.subjects()) | set(g.objects())

    # Calculating nG
    nG = len(all_nodes)

    # for storing number of in-/outgoing edges per node
    degree_counts = {}

    for s, _, o in g:
        # Outgoing edge: s -> o
        degree_counts[s] = degree_counts.get(s, 0) + 1
        # Incoming edge: o <- s
        degree_counts[o] = degree_counts.get(o, 0) + 1

    #sum_of_degrees = sum(degree_counts.values())

    if nG > 1:
        mean_degree = (2 * nE) / nG
        squared_diffs = [(deg_v - mean_degree) ** 2 for _,deg_v in degree_counts.items()]
        degree_variance = round(sum(squared_diffs) / (nG-1), 2)
    else:
        degree_variance = 0.0 

    return pd.DataFrame([
        {"Metric": "Degree Variance", "Value": float(degree_variance)},
    ])

def primitives_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates a set of primitive metrics (number of entities/properties/classes/instances/object properties) for a given RDF graph.

    The metrics are calculated as follows:
    - **Number of Entities**: The total count of unique resources (URIs and BNodes) that appear
      as a subject or a non-literal object in any triple. 
    - **Number of Properties**: The sum of two counts:
      1.  Properties explicitly declared as `owl:ObjectProperty`, `owl:DatatypeProperty`, or `owl:AnnotationProperty` (T-Box).
      2.  All unique predicates used in the graph (A-Box). 
    - **Number of Classes**: The count of unique classes identified by the `_get_classes_local` helper function.
    - **Number of Instances**:  Count of all individual instances in the graph.  Specifically, this is the number of unique 
      subjects in triples of the form `(s, rdf:type, o)` where `s` is not itself a class. In other words, classes (`rdf:type` objects) 
      are excluded from the count to ensure only individuals are counted.
    - **Number of Object Properties**: The sum of two counts:
      1.  Properties explicitly declared as `owl:ObjectProperty` (T-Box).
      2.  All unique predicates in the rdf triples that have a non-literal as an object (A-Box).

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): The number of decimal places for rounding. This parameter is currently
                                   not used in this function but is kept for API consistency. Defaults to 2.

    Returns:
        pd.DataFrame: A pandas DataFrame with two columns, "Metric" and "Value", containing the names
                      and calculated values of the primitive metrics.
    """
    ### Entities ###

    entities = set()

    for s, p, o in g:
        entities.add(s)  
                    
        if not isinstance(o, Literal): 
            entities.add(o)

    num_entities = len(entities)

    ### Classes ###
    classes = _get_classes_local(g)

    num_classes = len(classes)

    ### Properties ###

    property_types = [OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty]

    # Properties in T-Box
    properties_t = set()

    for t in property_types:
        for s, p, o in g.triples((None, RDF.type, t)):
            properties_t.add(s)

    num_properties_t = len(properties_t)

    # Properties in A-Box
    properties_a = set(g.predicates())

    num_properties_a = len(properties_a)

    num_properties = num_properties_t + num_properties_a

    ### Instances ###

    num_instances = _get_num_instances_local(g)

    ### Object Properties ###

    # Object Properties in T-Box
    object_properties_t = set(g.subjects(RDF.type, OWL.ObjectProperty))

    num_obj_properties_t = len(object_properties_t)

    # object Properties in A-Box
    object_properties_a = set()

    for s, p, o in g:
        if not isinstance(o, Literal):  #if object is not a literal
            object_properties_a.add(p)

    num_obj_properties_a = len(object_properties_a)

    num_obj_properties = num_obj_properties_t + num_obj_properties_a

    return pd.DataFrame([
    {"Metric": "Number of Entities",          "Value": int(num_entities)},
    {"Metric": "Number of Properties",        "Value": int(num_properties)},
    {"Metric": "Number of Classes",           "Value": int(num_classes)},
    {"Metric": "Number of Instances",         "Value": int(num_instances)},
    {"Metric": "Number of Object Properties", "Value": int(num_obj_properties)},
])

def depth_of_inh_tree_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the maximum depth of the inheritance tree in an RDF graph.

    This function identifies all inheritance hierarchies by finding root classes
    (classes with no superclass) and then traverses each hierarchy downwards
    using `rdfs:subClassOf` relationships. The metric represents the length of the
    longest path from a root class to a leaf class in any of the inheritance trees.

    The traversal is performed using a depth-first search (DFS) algorithm.

    Args:
        g (Graph): An RDF graph object to be analyzed.
        dec_places (int, optional): This parameter is kept for API consistency but is
                                    not used in this function. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the metric "Depth of Inheritance Tree"
                      and its calculated value (the maximum depth).
    """

    max_depth_inh_tree = 0

    # for storing subclasses of nodes
    subclasses_cache = {}

    def find_roots_without_superclass(g: Graph):
        """
        Finds root classes in the inheritance hierarchy of an RDF graph.

        A root class is defined as a class that is explicitly declared as an
        `owl:Class` or `rdfs:Class` but is not a subclass of any other class
        (i.e., it does not appear as the subject of an `rdfs:subClassOf` triple).

        Args:
            g (Graph): An RDF graph object.

        Returns:
            set: A set of URIRefs representing the root classes of the
                    inheritance hierarchies.
        """
        # 1) all objects which are explicitly named classes
        owl_classes  = set(g.subjects(RDF.type, OWL.Class))
        rdfs_classes = set(g.subjects(RDF.type, RDFS.Class))
        classes = owl_classes | rdfs_classes

        # 2) FILTER NOT EXISTS { ?root rdfs:subClassOf ?anyClass }
        roots = {c for c in classes if not any(g.triples((c, RDFS.subClassOf, None)))}

        return roots
    
    def get_subclasses(node):
        """
        Returns all direct subclasses of a given node (class) in the RDF graph.

        This helper function queries the RDF graph for all subjects of `rdfs:subClassOf`
        triples where the given `node` is the object. The result is cached for efficiency.
        Used to traverse the inheritance tree when searching for all subclass paths.

        Args:
            node: The RDF node (class) for which to find direct subclasses.

        Returns:
            set: A set of nodes representing the direct subclasses of the given node.
        """
        if node in subclasses_cache:
            return subclasses_cache[node]
        
        subclasses = set(g.subjects(RDFS.subClassOf, node))
        subclasses_cache[node] = subclasses

        return subclasses

    def dfs(path, node):
        """
        Performs a depth-first search (DFS) to find all inheritance paths from a class.

        This recursive function explores inheritance paths starting from the given `node`.
        It avoids cycles by checking if a class has already been visited in the
        current `path`. When a path terminates (i.e., a leaf class with no subclasses
        is reached), it updates the nonlocal `max_depth_inh_tree` metric.

        Args:
            path (list): The list of nodes representing the current inheritance path.
            node: The current class to visit in the DFS traversal.
        """

        nonlocal max_depth_inh_tree

        # for avoiding cycles
        if node in path:
            return
        
        path.append(node)
        
        neighbors = get_subclasses(node) 

        if not neighbors:
            max_depth_inh_tree = max(max_depth_inh_tree, len(path) - 1)
        else:
            for neighbor in neighbors:
                dfs(path, neighbor)

        path.pop()
    
    tree_roots = find_roots_without_superclass(g)

    for root in tree_roots:
        dfs([], root)

    return pd.DataFrame([
        {"Metric": "Depth of Inheritance Tree", "Value": int(max_depth_inh_tree)},
    ])

def tbox_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates several T-Box (Terminological Box) metrics for a given RDF graph 
    (Property Class Ratio, Class Property Ratio, Inheritance Richness, Attribute Richness).

    The metrics are calculated as follows:
    - **Property Class Ratio**: The ratio of explicitly declared T-Box properties (`owl:ObjectProperty`, 
      `owl:DatatypeProperty`, `owl:AnnotationProperty`) to the total number of classes.
      This metric indicates the average number of properties per class.
    - **Class Property Ratio**: The inverse of the Property Class Ratio, showing the ratio of classes to properties.
    - **Inheritance Richness**: The ratio of the number of subclass relationships (`rdfs:subClassOf`) to the
      total number of classes. 
    - **Attribute Richness**: The ratio of explicitly declared datatype properties (`owl:DatatypeProperty`) 
      to the total number of classes.  

    The function handles division-by-zero cases by returning 0 for a ratio if the denominator is zero.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): The number of decimal places to round the results to. Defaults to 2.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the names and calculated values of the T-Box metrics.
    """
    classes = _get_classes_local(g)

    num_classes = len(classes)

    subclass_triples = list(g.triples((None, RDFS.subClassOf, None)))
    num_subclasses = len(subclass_triples)

    property_types = [OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty]

    # Properties in T-Box
    properties_t = set()

    for t in property_types:
        for s, _, _ in g.triples((None, RDF.type, t)):
            properties_t.add(s)

    # this function calculates metrics regarding T-Box --> we are only interested in T-Box properties
    num_properties = len(properties_t)

    # Datatype Properties in T-Box
    datatype_properties_t = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    
    # this function calculates metrics regarding T-Box --> we are only interested in T-Box properties
    num_datatype_properties = len(datatype_properties_t)

    # Property Class Ratio - Inheritance Richness - Attribute Richness 
    if num_classes > 0:
        prop_class_ratio = round(num_properties / num_classes, dec_places) 
        inheritance_richness = round(num_subclasses / num_classes, dec_places) 
        attr_richness = round(num_datatype_properties / num_classes, dec_places)

    else:
    # source 172 - page: assumes that classes must exist for properties to exist (Number of Properties, Number of CLasses > 1)
    # I assume: no classes -> ratio = 0
        prop_class_ratio = 0.0
        inheritance_richness = 0.0
        attr_richness = 0.0

    # Class Property Ratio
    if num_properties > 0:
        class_prop_ratio = round(num_classes / num_properties, dec_places) 
    
    else:
    # metric is not defined for num_properties = 0
    # I assume: no properties -> ratio = 0
        class_prop_ratio = 0

    return pd.DataFrame([
        {"Metric": "Property Class Ratio", "Value": float(prop_class_ratio)},
        {"Metric": "Class Property Ratio", "Value": float(class_prop_ratio)},
        {"Metric": "Inheritance Richness", "Value": float(inheritance_richness)},
        {"Metric": "Attribute Richness", "Value": float(attr_richness)},
    ])

def abox_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates some A-Box (Assertional Box) metrics related to the instances in an RDF graph.
    (Average Class Connectivity, Average Population). 

    The metrics are calculated as follows:
    - **Average Class Connectivity**: Measures the average number of relationships that instances of a class
      have with instances of other classes. The function analyzes relationships between resources (ignoring
      rdf:type statements and links to Literals). For each valid relationship, it checks the classes
      of the two connected resources. If they belong to different classes, both classes receive a "connectivity point". 
      The final metric is the average number of these points across all classes, which shows how strongly 
      the classes in the ontology are interconnected.
    - **Average Population**: The average number of instances per class in the ontology. It is calculated
      by dividing the total number of instances by the total number of classes.

    The function handles division-by-zero cases by returning 0 if no classes are present.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): The number of decimal places to round the results to. Defaults to 2.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the names and calculated values of the A-Box metrics.
    """
    # Average Class Connectivity
    # Connectivity of a class is defined as the total number of relationships instances of 
    # the class have with instances of other classes (source 227 - page 10)

    classes = _get_classes_local(g)

    num_classes = len(classes)

    connectivity = {}  

    for s, p, o in g:
        # we are not interested in rdf:type relations
        # literals don't belong to classes
        if p == RDF.type or isinstance(o, Literal):
            continue
        
        # classes the subject belongs to
        s_classes = set(g.objects(s, RDF.type))
        # classes the object belongs to
        o_classes = set(g.objects(o, RDF.type))

        if not s_classes or not o_classes:
            continue  # we are just interested in the instance-to-instance relations

        # s is connected to o --> every class s is belonging to is connected to every class o is belonging to
        for class_of_subject in s_classes:
            for class_of_object in o_classes:
                if class_of_subject != class_of_object:
                    connectivity[class_of_subject] = connectivity.get(class_of_subject, 0) + 1
                    connectivity[class_of_object] = connectivity.get(class_of_object, 0) + 1

    sum_connectivities = sum(connectivity.values())

    num_instances = _get_num_instances_local(g)

    # Average Class Connectivity - Average Population
    if num_classes > 0:
        avg_class_connectivity = round(sum_connectivities / num_classes, dec_places)
        avg_population = round(num_instances / num_classes, dec_places)

    else:
        # metric is not defined for num_classes = 0
        avg_class_connectivity = 0.0
        avg_population = 0.0

    return pd.DataFrame([
        {"Metric": "Average Class Connectivity", "Value": float(avg_class_connectivity)},
        {"Metric": "Average Population", "Value": float(avg_population)},
    ])

def cohesion_local(g: Graph, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the cohesion metric of an RDF graph by identifying its connected components.

    This function determines how many disconnected subgraphs (connected components)
    exist within the RDF graph. Each component represents a set of nodes that are
    mutually reachable through RDF triples, either as subject or object.

    The algorithm works iteratively as follows:
    1. Select an arbitrary unvisited node as a starting point.
    2. From this node, find all directly and indirectly reachable nodes by repeatedly 
       exploring their neighbors (both incoming and outgoing edges).
    3. When no new nodes can be reached, mark the discovered set of nodes as one component.
    4. If unvisited nodes still remain, repeat the process with another unvisited node.
    
    The total number of discovered components is returned as the cohesion metric.

    Args:
        g (Graph): An RDF graph object containing triples (subject, predicate, object).
        dec_places (int, optional): The number of decimal places to round the results to. Defaults to 2.

    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the metric "Cohesion" and its calculated
                      value, representing the number of disconnected subgraphs (components)
                      in the RDF graph.
    """
    neighbors_cache = {}

    def get_new_neighbors(node):
        """
        Retrieves the neighbors of a given node from the graph.

        This function finds all subjects and objects connected to the given `node`
        in the RDF graph. It uses a cache (`neighbors_cache`) to store and retrieve
        results for previously seen nodes, improving performance by avoiding
        redundant graph lookups. 
        
        The function only returns neighbors that are part of `all_nodes` and have not
        yet been visited during traversal. It also includes consistency checks to ensure
        that all discovered neighbors belong to the expected set of graph nodes. If a 
        neighbor is discovered that is not contained in `all_nodes`, a `ValueError`is
        raised. This indicates a possible inconsistency in the RDF graph (e.g., missing
        triples or incomplete data extraction).

        Args:
            node: The RDF node for which to find neighbors.

        Returns:
            set: A set of unvisited neighbor nodes. Returns an empty set if no
                 neighbors are found.

        Raises:
            ValueError: If a discovered neighbor is not part of `all_nodes`.  
                        This may indicate that the RDF graph is incomplete or inconsistent.
        """
        nonlocal g
        nonlocal all_nodes
        nonlocal visited

        # TODO: ich denke das nachschauen im cache is unnötig -> wenn cache eintrag existiert -> return set() !
        if node in neighbors_cache:
            return neighbors_cache[node]

        # neighbors as a set is okay because we just want to get all neighbors which can be reached from node
        # --> we do not have to store howe many times a neighbor can be visited from a node --> set is enough
        neighbors = set()

        # ingoing: ?s ?p node
        for neighbor in g.subjects(object=node):
            if ((neighbor in all_nodes) and (neighbor not in visited)):
                neighbors.add(neighbor)

            elif neighbor not in visited:
                #debug_print("##### NEIGHBOR NOT IN ALL_NODES: " + format_literal_for_sparql(neighbor_val, neighbor_type, neighbor_lang, neighbor_dtype) + " #####")
                raise ValueError(
                            f"Neighbor {neighbor} (from outgoing edge) not found in all_nodes. "
                            f"This may indicate an incomplete or inconsistent RDF graph."
                        ) 


        # outgoing: node ?p ?o  (only possible if node is not literal)
        if not isinstance(node, Literal):
            for neighbor in g.objects(subject=node):
                if ((neighbor in all_nodes) and (neighbor not in visited)):
                    neighbors.add(neighbor)
                    #debug_print("found unvisited neighbor: " + str(neighbor))
                elif neighbor not in visited:
                    raise ValueError(
                            f"Neighbor {neighbor} (from outgoing edge) not found in all_nodes. "
                            f"This may indicate an incomplete or inconsistent RDF graph."
                        ) 

        neighbors_cache[node] = neighbors

        return neighbors

    all_nodes = set(g.subjects()).union(set(g.objects()))

    # describes number of independent components of graph (like subgraphs which are connected internally)
    components = 0

    # describes visited / discovered nodes
    visited = set()

    # stores each component with its nodes
    components_nodes = []

    ## algorithm: searches for components of graph
    while visited != all_nodes:

        # Choose first unvisited node
        # iter creates iterator for set - next gives next element in set
        start_node = next(iter(all_nodes - visited))

        frontier = {start_node}
        visited.add(start_node)

        # starting new component
        current_component = set([start_node])

        while frontier:
            
            new_frontier = set()

            for node in frontier:
                
                new_visited_neighbors = set()
                
                # getting just all completely new neighbors
                new_visited_neighbors = get_new_neighbors(node)

                # neighbors just consists of new neighbors 
                visited |= new_visited_neighbors

                current_component |= new_visited_neighbors

                if visited == all_nodes:
                    break

                new_frontier |= new_visited_neighbors
            
            if visited == all_nodes:
                break

            frontier = new_frontier
            
        components += 1
        components_nodes.append(current_component)

    
    return pd.DataFrame([
        {"Metric": "Cohesion", "Value": int(components)},
    ])

############## ENDPOINT FUNCTIONS ##############


# def _get_sparql_from_endpoint(endpoint_url: str, default_graph: str = None):
#     """
#     Initializes and configures a SPARQLWrapper instance for a given endpoint.

#     This helper function creates a SPARQLWrapper object, optionally sets a default
#     graph for the queries, and sets the default return format to JSON.

#     Args:
#         endpoint_url (str): The URL of the SPARQL endpoint to connect to.
#         default_graph (str, optional): The URI of the default graph to be used
#                                        for queries. If None, no default graph
#                                        is set. Defaults to None.

#     Returns:
#         SPARQLWrapper: A configured SPARQLWrapper instance ready to be used for
#                        executing queries.

#     Raises:
#         ConnectionError: If the SPARQLWrapper instance cannot be initialized, e.g.,
#                          due to an invalid endpoint URL, network issue, or
#                          misconfiguration.
#     """
    
#     try:
#         sparql = SPARQLWrapper(endpoint_url)
        
#         if default_graph:
#             sparql.addDefaultGraph(default_graph)
        
#         sparql.setReturnFormat(JSON)
        
#         return sparql
    
#     except Exception as e:
#         raise ConnectionError(f"Failed to initialize SPARQL endpoint: {e}")

def _send_query(query: str, sparql: SPARQLWrapper, retfmt=JSON):
    """
    Executes a SPARQL query against a configured endpoint and returns the result.

    This function takes a SPARQL query, sets it on the provided SPARQLWrapper instance,
    executes the query, and converts the result to the specified format. It includes
    error handling for empty results and other exceptions that may occur during the
    query execution.

    Args:
        query (str): The SPARQL query string to be executed.
        sparql (SPARQLWrapper): A configured SPARQLWrapper instance.
        retfmt (str, optional): The desired return format for the query results.
                                Defaults to JSON.

    Returns:
        dict or any: The converted result of the SPARQL query, typically a dictionary
                     when the return format is JSON.
                                
    Raises:
        ValueError: If the SPARQL endpoint returns an empty result.
        Exception: Propagates any exception that occurs during the query execution,
                   such as connection errors.
    """
    try:
        sparql.setReturnFormat(retfmt)
        sparql.setQuery(query)
        res = sparql.query().convert()

        # if res is empty
        if not res:  
            raise ValueError(f"Endpoint returns empty result")

        return res
    
    except Exception as e:
        #raise RuntimeError(f"Endpoint-Error: {e}\nQuery:\n{query}") from e
        #error_message = f"Endpoint-Error: {str(e)}. Query:\n{query}"
        raise e
        #raise RuntimeError(error_message) 


def _get_num_classes_ep(sparql: SPARQLWrapper) -> int:
    """
    Queries a SPARQL endpoint to count the total number of unique classes.

    This function uses a SPARQL query to identify and count all
    distinct classes based on some criteria. A class can be:

    1.  Any resource that is the object of +an `rdf:type` triple.
    2.  Both the subject and object of an `rdfs:subClassOf` triple.
    3.  Both the subject and object of an `owl:equivalentClass` triple.
    4.  The subject of a triple declaring an `owl:Restriction`.
    5.  The subject of complex class definitions using `owl:unionOf`, 
        `owl:intersectionOf`, `owl:complementOf`, or `owl:oneOf`.
    6.  The subject of an `owl:hasValue` restriction.

    Args:
        sparql (SPARQLWrapper): A configured SPARQLWrapper instance 

    Returns:
        int: The total number of unique classes found in the endpoint.
    """
    
    # Defintion of Class: 
    # source: 213, page: 5 - source 250, page 3
    # TNOC (total number of classes/concepts) = classes, subclasses, superclasses, anonymous classes
    # anonymous classes = equivalent/restriction/unionOf/intersectionOf/complementOf/oneOf/hasValue classes
    query_classes = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT (COUNT(DISTINCT ?class) AS ?num_classes)
        WHERE {
            {
                # 1. explicitly/implicitly used RDF classes
                # explicitly: ?class a owl:Class . or ?class a rdfs:Class .
                # implicitly: ?any rdf:type ?class . (includes also explicitly used classes)

                ?any rdf:type ?class .
            }
        UNION
            {
                # 2. subclasses
                ?class rdfs:subClassOf ?any .
            }
        UNION
            {
                # 3. superclasses
                ?any rdfs:subClassOf ?class .
            }
        UNION
            {
                # 4. classes used with owl:equivalentClass
                { ?class owl:equivalentClass ?any . }
                UNION
                { ?any owl:equivalentClass ?class . }
            }
        UNION
            {
                # 5. OWL restriction classes
                ?class a owl:Restriction .
            }
        UNION
            {
                # 6. complex classes with using unionOf, intersectionOf etc.
                ?class owl:unionOf|owl:intersectionOf|owl:complementOf|owl:oneOf ?list .
            }
        UNION
            {
                # 7. OWL hasValue restrictions
                ?class owl:hasValue ?val .
            }
        }  
    """

    results = _send_query(query_classes, sparql, JSON)

    num_classes = 0

    for binding in results["results"]["bindings"]:
        num_classes = int(binding["num_classes"]["value"])

    return num_classes

def _get_classes_ep(sparql: SPARQLWrapper):
    """
    Queries a SPARQL endpoint to count the total number of unique classes.

    This function uses a SPARQL query to identify and count all
    distinct classes based on some criteria. A class can be:

    1.  Any resource that is the object of +an `rdf:type` triple.
    2.  Both the subject and object of an `rdfs:subClassOf` triple.
    3.  Both the subject and object of an `owl:equivalentClass` triple.
    4.  The subject of a triple declaring an `owl:Restriction`.
    5.  The subject of complex class definitions using `owl:unionOf`, 
        `owl:intersectionOf`, `owl:complementOf`, or `owl:oneOf`.
    6.  The subject of an `owl:hasValue` restriction.

    Args:
        sparql (SPARQLWrapper): A configured SPARQLWrapper instance 

    Returns:
        int: The total number of unique classes found in the endpoint.
    """
    
    # Defintion of Class: 
    # source: 213, page: 5 - source 250, page 3
    # TNOC (total number of classes/concepts) = classes, subclasses, superclasses, anonymous classes
    # anonymous classes = equivalent/restriction/unionOf/intersectionOf/complementOf/oneOf/hasValue classes
    query_classes = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT DISTINCT ?class
        WHERE {
            {
                # 1. explicitly/implicitly used RDF classes
                # explicitly: ?class a owl:Class . or ?class a rdfs:Class .
                # implicitly: ?any rdf:type ?class . (includes also explicitly used classes)

                ?any rdf:type ?class .
            }
        UNION
            {
                # 2. subclasses
                ?class rdfs:subClassOf ?any .
            }
        UNION
            {
                # 3. superclasses
                ?any rdfs:subClassOf ?class .
            }
        UNION
            {
                # 4. classes used with owl:equivalentClass
                { ?class owl:equivalentClass ?any . }
                UNION
                { ?any owl:equivalentClass ?class . }
            }
        UNION
            {
                # 5. OWL restriction classes
                ?class a owl:Restriction .
            }
        UNION
            {
                # 6. complex classes with using unionOf, intersectionOf etc.
                ?class owl:unionOf|owl:intersectionOf|owl:complementOf|owl:oneOf ?list .
            }
        UNION
            {
                # 7. OWL hasValue restrictions
                ?class owl:hasValue ?val .
            }
        }  
    """

    results = _send_query(query_classes, sparql, JSON)

    # num_classes = 0

    classes = set()

    for binding in results["results"]["bindings"]:
        # num_classes = int(binding["num_classes"]["value"])
        classes.add(binding["class"]["value"])

    # return num_classes
    return classes

def _get_num_instances_ep(sparql: SPARQLWrapper) -> int:
    """
    Retrieves and counts the number of instance resources (ABox individuals) 
    in an RDF graph accessible through a SPARQL endpoint.

    This function queries the endpoint for all triples of the form 
    `?s rdf:type ?type` to identify resources that are explicitly declared 
    as instances of some class. It then distinguishes between classes 
    (objects of `rdf:type` statements) and instances (subjects of such statements).

    To ensure that only true ABox individuals are counted (and not classes that 
    also appear as instances), the function subtracts the set of classes from 
    the set of all subjects.

    The result corresponds to the total number of entities that appear as 
    instances but not as classes within the RDF graph.

    Args:
        sparql (SPARQLWrapper):  A configured SPARQLWrapper instance.

    Returns:
        int: The number of unique instances (individuals) in the graph. 
    """
    query_inst = """ 
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?s ?type
        WHERE {
            ?s rdf:type ?type . 
        }
    """

    results = _send_query(query_inst, sparql, JSON)
    
    num_instances = 0

    instances = set()
    classes = set()

    for binding in results["results"]["bindings"]:
        rdf_class = binding["type"]["value"]
        instance = binding["s"]["value"]

        classes.add(rdf_class)
        instances.add(instance)

    indidivuals = instances - classes

    num_instances = len(indidivuals)

    return num_instances

def _get_num_instances_ep_2(sparql: SPARQLWrapper) -> int:
    """
    Retrieves and counts the number of instance resources (ABox individuals) 
    in an RDF graph accessible through a SPARQL endpoint.

    This function queries the endpoint for all triples of the form 
    `?s rdf:type ?type` to identify resources that are explicitly declared 
    as instances of some class. It then distinguishes between classes 
    (objects of `rdf:type` statements) and instances (subjects of such statements).

    To ensure that only true ABox individuals are counted (and not classes that 
    also appear as instances), the function subtracts the set of classes from 
    the set of all subjects.

    The result corresponds to the total number of entities that appear as 
    instances but not as classes within the RDF graph.

    Args:
        sparql (SPARQLWrapper):  A configured SPARQLWrapper instance.

    Returns:
        int: The number of unique instances (individuals) in the graph. 
    """
    query_inst = """ 
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?s
        WHERE {
            ?s rdf:type ?type . 
        }
    """

    results = _send_query(query_inst, sparql, JSON)
    
    num_instances = 0

    instances = set()
    # classes = set()

    for binding in results["results"]["bindings"]:
        # rdf_class = binding["type"]["value"]
        instance = binding["s"]["value"]

        # classes.add(rdf_class)
        instances.add(instance)

    classes = _get_classes_ep(sparql)

    indidivuals = instances - classes

    num_instances = len(indidivuals)

    return num_instances

def _get_num_properties_ep(sparql: SPARQLWrapper, calc_t: bool = True , calc_a: bool = True) -> tuple[int, int]:
    """
    Queries a SPARQL endpoint to count T-Box and A-Box properties.

    This function can calculate two types of property counts:
    1.  T-Box Properties (`calc_t=True`): Counts properties that are explicitly
        declared as `owl:ObjectProperty`, `owl:DatatypeProperty`, or
        `owl:AnnotationProperty`.
    2.  A-Box Properties (`calc_a=True`): Counts all unique predicates that
        are used in any triple in the graph.

    Args:
        sparql (SPARQLWrapper): A configured SPARQLWrapper instance.
        calc_t (bool, optional): Whether to calculate T-Box properties.
                                 Defaults to True.
        calc_a (bool, optional): Whether to calculate A-Box properties.
                                 Defaults to True.

    Returns:
        tuple[int, int]: A tuple containing the count of T-Box properties
                         and A-Box properties `(num_properties_t, num_properties_a)`.
    """
    
    num_properties_t = 0
    num_properties_a = 0

    if calc_t:
        # number of properties in T-Box
        # source says: property = explicitly defined property
        query_properties_t = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT (COUNT(DISTINCT ?property) AS ?propertyCount)
            WHERE {
                VALUES ?type { owl:ObjectProperty owl:DatatypeProperty owl:AnnotationProperty }
                ?property rdf:type ?type .
            }
        """

        results = _send_query(query_properties_t, sparql, JSON)

        #num_properties_t = 0

        for binding in results["results"]["bindings"]:
            num_properties_t = int(binding["propertyCount"]["value"])

    if calc_a:
        # number of properties in A-Box
        # source says: property = the unique ?p in ?s ?p ?o
        query_properties_a = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT (COUNT(DISTINCT ?p) AS ?propertyCount)
            WHERE {
                ?s ?p ?o .
            }
        """

        results = _send_query(query_properties_a, sparql, JSON)

        #num_properties_a = 0

        for binding in results["results"]["bindings"]:
            num_properties_a = int(binding["propertyCount"]["value"])

    return (num_properties_t, num_properties_a)

def paths_depth_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """    
    Calculates path- and depth-related metrics for an RDF graph accessible via a SPARQL endpoint.

    In the context of an RDF graph, a path is defined as a sequence of nodes connected by directed 
    edges (triples). Nodes correspond to the subjects and objects of RDF triples, while predicates 
    represent the edges between them. This function determines structural properties of the RDF graph 
    by traversing all reachable nodes starting from the identified root nodes.

    This function traverses the graph to compute four metrics:
    - **Number of Paths**: The total count of all unique paths starting from a
      root node and ending at a leaf node (a node with no outgoing edges or a literal).
    - **Absolute Depth**: The sum of the lengths (in edges) of all identified paths.
    - **Average Depth**: The average length of a path (Absolute Depth / Number of Paths).
    - **Maximum Depth**: The length of the longest path found in the graph.

    Implementation Details:

    1. Connection Setup:
        Establishes a connection to the SPARQL endpoint using the provided `endpoint_url` and 
        optional `default_graph`.

    2. Literal Extraction:
        Identifies all literal nodes (`isLiteral(?literal)`) in the graph and stores them in a set 
        for later exclusion during traversal (literals are treated as terminal nodes). Literals have 
        to be stored in advance, because searching for some complex / big literals by using queries
        ends up in Errors.

    3. Blank Node (BNode) Handling:
        Since blank nodes have no globally unique identifiers and may differ across queries, 
        they are handled using two precomputed sets:
        - `bnode_subject_neighbors`: stores triples where a blank node appears as a subject.  
        - `bnode_object_neighbors`: stores triples where a blank node appears as an object.  
        These structures are later used to resolve neighbor relationships consistently during traversal.

    4. Root Node Identification:
        Root nodes are defined as all nodes that never appear as an object in any triple 
        (`FILTER NOT EXISTS { ?s ?p2 ?root }`). Additionally, blank nodes that have object-neighbors 
        but no subject-neighbors are considered roots as well. This way of special treatment for blank
        nodes is used because they are not uniquely identified.

    5. Depth-First Search:
        For each identified root node, the algorithm calls the internal `dfs` function to recursively 
        explore all paths reachable from that root.  
        Each completed path contributes to the global metrics that represent the structural depth 
        characteristics of the graph.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph to use for queries. Defaults to None.
        dec_places (int, optional): The number of decimal places to round the result to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the names and calculated values of the metrics.
    """

    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    query_literals = """
        SELECT DISTINCT ?literal
        WHERE {
            ?s ?p ?literal .
            FILTER(isLiteral(?literal))
        }
    """

    results = _send_query(query_literals, sparql, JSON)

    literals = set()

    for res in results["results"]["bindings"]:
        literal_value = res["literal"]["value"]
        literal_type = res["literal"]["type"]
        lang_tag = res["literal"].get("xml:lang") 
        datatype = res["literal"].get("datatype") 
        if lang_tag:
            literal_value = f"{literal_value}@{lang_tag}"
        elif datatype:
            literal_value = f"{literal_value}^^{datatype}"

        literals.add((literal_value, literal_type, lang_tag, datatype))

    # idea: storing left (=bnode as object) and right (=bnode as subject) neighbors for every bnode in a set
    # when getting neighbors of nodes, bnode neighbors of the nodes will be searched by looking up in the set
    # this way is chosen because bnodes do not have a unique id and are numbered ascending in every query
    # so f.e. b1 in one query does not have to be the same b1 in another query
    query_bnodes_neighbors = """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>

        SELECT DISTINCT ?bnode ?inNeighbor ?inP ?outNeighbor ?outP
        WHERE {

            # 1) all triples, which include a bnode
            {
                SELECT DISTINCT ?bnode WHERE {
                { ?inS  ?inPtmp  ?bnode . FILTER(isBlank(?bnode)) }
                UNION
                { ?bnode ?outPtmp ?outO . FILTER(isBlank(?bnode)) }
                }
            }

            # 2) Optional: all triples, where the bnode is an object
            OPTIONAL { 
                ?inNeighbor ?inP ?bnode . 
            }

            # 3) Optional: all triples, where the bnode is a subject
            OPTIONAL { 
                ?bnode ?outP ?outNeighbor .  
            }

        }
    """

    results = _send_query(query_bnodes_neighbors, sparql, JSON)

    # stores bnodes and their neighbors where bnodes are subjects
    bnode_subject_neighbors = set()
    # stores bnodes and their neighbors where bnodes are objects
    bnode_object_neighbors = set()
    # stores all root nodes in graph
    root_nodes = set()

    for binding in results["results"]["bindings"]:

        bnode_obj = binding["bnode"]
        bnode_value = bnode_obj["value"]
        bnode_type = bnode_obj["type"]
        # this values will be empty, but they are also stored for completeness
        bnode_lang = bnode_obj.get("xml:lang") 
        bnode_dtype = bnode_obj.get("datatype")
        bnode = (bnode_value, bnode_type, bnode_lang, bnode_dtype)
        
        # inNeighbor if it exists
        in_ngh_obj = binding.get("inNeighbor")

        if in_ngh_obj:

            in_ngh_obj = binding["inNeighbor"]
            in_ngh_value = in_ngh_obj["value"]
            in_ngh_type = in_ngh_obj["type"]
            in_ngh_lang = in_ngh_obj.get("xml:lang") 
            in_ngh_dtype = in_ngh_obj.get("datatype")
            in_neighbor = (in_ngh_value, in_ngh_type, in_ngh_lang, in_ngh_dtype)

            #storing predicate too because it can be the case that one node has the same neighbor twice but with 2 different predicates
            inP = binding["inP"]["value"]

            bnode_object_neighbors.add((bnode, in_neighbor, inP))

        # outNeighbor if it exists
        out_ngh_obj = binding.get("outNeighbor")
        
        if out_ngh_obj:

            out_ngh_value = out_ngh_obj["value"]
            out_ngh_type = out_ngh_obj["type"]
            out_ngh_lang = out_ngh_obj.get("xml:lang")
            out_ngh_dtype = out_ngh_obj.get("datatype")
            out_neighbor = (out_ngh_value, out_ngh_type, out_ngh_lang, out_ngh_dtype)
            
            #storing predicate too because it can be the case that one node has the same neighbor twice but with 2 different predicates
            outP = binding["outP"]["value"]
            
            bnode_subject_neighbors.add((bnode, out_neighbor, outP))

    # if a bnode has object-neighbors, but no subject.neighbors, then this bnode is a root!
    root_nodes = {b for (b,_,_) in bnode_subject_neighbors} - {b for (b,_,_) in bnode_object_neighbors}

    # for getting non-Bnode roots
    query_roots = """
        SELECT DISTINCT ?root
        WHERE {
            ?root ?p ?o .
            FILTER NOT EXISTS {
                ?s ?p2 ?root .
            }
            FILTER(!isBlank(?root))
        }
    """

    results = _send_query(query_roots, sparql, JSON)

    for res in results["results"]["bindings"]:
        root_obj = res["root"]
        root_value = root_obj["value"]
        root_type = root_obj["type"]
        root_lang = root_obj.get("xml:lang") 
        root_dtype = root_obj.get("datatype")

        root = (root_value, root_type, root_lang, root_dtype)

        root_nodes.add(root)

    num_paths = 0
    abs_depth = 0
    max_depth = 0

    # for storing neighbors of nodes
    neighbors_cache = {}

    def _get_neighbors(node):
        """
        Retrieves the neighboring nodes of a given RDF node via SPARQL queries.

        This function identifies all directly connected nodes (neighbors) of the given `node`
        in a remote RDF graph accessed through a SPARQL endpoint. It first checks a local 
        cache (`neighbors_cache`) to avoid redundant lookups for previously processed nodes.

        For non-blank nodes (`node_type != "bnode"`), the function performs a SPARQL query 
        that retrieves all objects directly connected to the node via any predicate, excluding 
        blank nodes. Literals and Typed-Literals are formatted with language or datatype annotations 
        (e.g., `"text@en"` or `"42^^xsd:int"`). Additionally, blank node connections that 
        reference the current node as an object are appended from a precomputed list where bnodes
        with their subject neighbors are stored (`bnode_object_neighbors`).

        For blank nodes (`node_type == "bnode"`), the function instead searches for all objects 
        that are connected to the given blank node as a subject, using the list 
        (`bnode_subject_neighbors`).

        All retrieved neighbors are cached in `neighbors_cache` for reuse in later calls.

        Args:
            node (tuple): A tuple representing the RDF node 
                        (node_value, node_type, lang_tag, datatype).

        Returns:
            list: A list of tuples representing the neighboring nodes. Each tuple contains 
                (value, value_type, lang_tag, datatype). Returns an empty list if no 
                neighbors are found.
        """

        # Check if node is in cache
        if node in neighbors_cache:
            return neighbors_cache[node]
 
        node_value,node_type,_,_ = node 
        
        if node_type != "bnode":
            # just searching for neighbors which are not of type bnode. the bnode neighbors will be handled afterwards
            query = f"""
                SELECT DISTINCT ?p ?next 
                WHERE {{
                    <{str(node_value)}> ?p ?next .
                    FILTER(!isBlank(?next))
                }}
            """

            results = _send_query(query, sparql, JSON)
        
            neighbors = []
            for binding in results["results"]["bindings"]:
                next_obj = binding["next"]
                value = next_obj["value"]
                value_type = next_obj["type"]  
                lang_tag = next_obj.get("xml:lang") 
                datatype = next_obj.get("datatype") 
                
                if value_type == "literal" or value_type == "typed-literal":
                    lang_tag = next_obj.get("xml:lang") 
                    datatype = next_obj.get("datatype") 
                    
                    # directly storing tag information in the value
                    if lang_tag:  # add lagnuage tag
                        value = f"{value}@{lang_tag}"
                    elif datatype:  # add datatype tag
                        value = f"{value}^^{datatype}"

                #neighbors.append((value, value_type))
                neighbors.append((value, value_type, lang_tag, datatype))

            # checking for bnode neighbors
            for bnode, neighbor,_ in bnode_object_neighbors:
                if neighbor == node:
                    neighbors.append(bnode)

            neighbors_cache[node] = neighbors

            return neighbors
        
        elif node_type == 'bnode':
            
            neighbors = []
            
            # directly looking for neighbors in bnode neighbor set
            for bnode, neighbor,_ in bnode_subject_neighbors:
                if bnode == node:
                    neighbors.append(neighbor)

            neighbors_cache[node] = neighbors

            return neighbors

    def dfs(path, node):
        """
        Performs a depth-first search (DFS) to find all paths from a node.

        This recursive function explores paths starting from the given `node`.
        It avoids cycles by checking if a node has already been visited in the
        current `path`. When a path terminates (i.e., a node with no outgoing
        neighbors is reached), it updates the global metrics: `num_paths`,
        `abs_depth`, and `max_depth`.

        Literals and Typed Literals are treated as terminal nodes (they cannot have 
        outgoing neighbors). When such nodes are encountered, the path is counted 
        and its depth is added to the global statistics.

        Args:
            node: The current node to visit in the DFS traversal.
            path (list): The list of nodes representing the current path from the
                         root to the parent of the current `node`.
        """
        nonlocal num_paths, abs_depth, max_depth

        # skip node if it already is in path (avoiding cycles)
        if node in path:
            return
        
        # add current node to path
        path.append(node)

        _, node_type, _, _ = node
        
        # if node is a (typed) literal, than it cannot have any neighbors
        neighbors = list(_get_neighbors(node)) if (node_type != "literal" and node_type != "typed-literal" and node not in literals) else []

        if not neighbors:
            num_paths += 1
            abs_depth += (len(path) - 1)
            max_depth = max(max_depth, len(path) - 1)

        else:
            for neighbor in neighbors:
                # if node is literal -> it does not have any neighbors -> path is finished
                if neighbor in literals:
                    path.append(neighbor)
                    #paths.append(list(path))
                    num_paths += 1
                    abs_depth += len(path) - 1
                    max_depth = max(max_depth, len(path) - 1)
                    # remove node from the path to find next path
                    path.pop()
                else:
                    dfs(path, neighbor)

        # remove node from the path to find next path
        path.pop()

    for r in root_nodes:
        dfs([], r)

    avg_depth = round(abs_depth / num_paths, dec_places) if num_paths > 0 else 0.0
    return pd.DataFrame([
        {"Metric": "Number of Paths", "Value": int(num_paths)},
        {"Metric": "Absolute Depth", "Value": int(abs_depth)},
        {"Metric": "Average Depth", "Value": float(avg_depth)},
        {"Metric": "Maximum Depth", "Value": int(max_depth)},
    ])

def ont_tangledness_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the Ontology Tangledness metric for a graph via a SPARQL endpoint.

    Ontology Tangledness is defined as the ratio of the total number of classes to the tangled classes.
    A class is considered "tangled" if it has more than one
    direct superclass ( = multiple `rdfs:subClassOf` relationships).

    The function first determines the total number of classes by calling
    `_get_num_classes_ep`. It then executes a second query to count the number
    of classes with more than one superclass. The final metric is the ratio
    of these two values. If there are no tangled classes, the metric is 0.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph to use for queries. Defaults to None.
        dec_places (int, optional): The number of decimal places to round the result to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the metric name ("Ontology Tangledness")
                      and its calculated value.
    """

    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    # num_classes = _get_num_classes_ep(sparql)

    classes = _get_classes_ep(sparql)

    num_classes = len(classes)

    # Select number of classes with more than one superclass 
    query_var2 = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(?class) AS ?tangledCount)
        WHERE {
            SELECT ?class (COUNT(?super) AS ?numSupers)
            WHERE {
                ?class rdfs:subClassOf ?super .
            }
            GROUP BY ?class
            HAVING (COUNT(?super) > 1)
        }
    """

    results = _send_query(query_var2, sparql, JSON)

    t = 0

    for binding in results["results"]["bindings"]:
        t = int(binding["tangledCount"]["value"])


    if t > 0:
        # source 37 says num_classes / t
        # source 73 says denominator and numerator should be switched -> t / num_classes
        ont_tangledness = round(num_classes / t, dec_places) 
    else:
        ont_tangledness = 0.0

    return pd.DataFrame([
        {"Metric": "Ontology Tangledness", "Value": float(ont_tangledness)},
    ])

def degree_variance_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the degree variance of a graph via a SPARQL endpoint.

    The degree variance is computed as the variance of the degree distribution,
    where the degree of a node is the sum of its incoming and outgoing edges.
    This function executes several SPARQL queries to get the total number of
    edges (nE), the total number of nodes (nG), and the degree of each node.
    The result is returned as a pandas DataFrame.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph to use for queries.
                                              Defaults to None.
        dec_places (int, optional): Number of decimal places to round the variance to.
                                    Default is 2.

    Returns:
        pd.DataFrame: A DataFrame with one row containing the metric name ("Degree Variance")
                      and its calculated value.
    """
    
    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    # nG...number of nodes in gaph
    # nE...number of edges in graph

    # Calculating nE
    query_nE = """
        SELECT (COUNT(*) AS ?tripleCount)
        WHERE {
            ?s ?p ?o .
        }
    """

    results = _send_query(query_nE, sparql, JSON)

    nE = 0

    for binding in results["results"]["bindings"]:
        nE = int(binding["tripleCount"]["value"]) 

    # if nE == 0: metric is 0
    if nE == 0:
        degree_variance = 0.0 

        return pd.DataFrame([
            {"Metric": "Degree Variance", "Value": float(degree_variance)},
        ])

    # Calculating nG
    query_nG = """
        SELECT (COUNT(DISTINCT ?node) AS ?nodeCount)
        WHERE {
            {
                SELECT ?node 
                WHERE {
                    { ?node ?p1 ?o }       
                    UNION
                    { ?s ?p2 ?node }       
                }
            }
        }
    """

    results = _send_query(query_nG, sparql, JSON)

    nG = 0

    for binding in results["results"]["bindings"]:
        nG = int(binding["nodeCount"]["value"])

    # if nG <= 1: metric is 0
    if nG <= 1:
        degree_variance = 0.0 

        return pd.DataFrame([
            {"Metric": "Degree Variance", "Value": float(degree_variance)},
        ])

    # Calculating degree for every node in graph
    query_degrees = """
        SELECT ?node (COUNT(?any) AS ?degree)
        WHERE {
            { ?node ?p1 ?any }     
            UNION
            { ?any ?p2 ?node }      
        }
        GROUP BY ?node
    """

    results = _send_query(query_degrees, sparql, JSON)

    # for storing degree of each node
    degrees = []

    for binding in results["results"]["bindings"]:
        node = binding["node"]["value"]
        degree = int(binding["degree"]["value"])
        degrees.append((node, degree))

    if nG > 1:
        mean_degree = (2 * nE) / nG
        squared_diffs = [(deg_v - mean_degree) ** 2 for _,deg_v in degrees]
        degree_variance = round(sum(squared_diffs) / (nG-1), 2)
    else:
        degree_variance = 0.0 

    return pd.DataFrame([
        {"Metric": "Degree Variance", "Value": float(degree_variance)},
    ])
        
def primitives_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates a set of primitive metrics for a graph via a SPARQL endpoint.

    The metrics are calculated by executing multiple SPARQL queries:
    - **Number of Entities**: The total count of unique resources (URIs and BNodes) that appear
      as a subject or a non-literal object in any triple.
    - **Number of Properties**: The sum of two counts from the `_get_num_properties_ep` helper function:
    - **Number of Classes**: The count of unique classes from the `_get_num_classes_ep` helper function.
    - **Number of Instances**: The total number of individuals from the `_get_num_instances_ep` helper function.
    - **Number of Object Properties**: The sum of two counts:
      1.  T-Box: Properties explicitly declared as `owl:ObjectProperty`.
      2.  A-Box: All unique predicates in triples that have a non-literal as an object.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph for queries. Defaults to None.
        dec_places (int, optional): Number of decimal places to round the values to. Default is 2.

    Returns:
        pd.DataFrame: A DataFrame with columns "Metric" and "Value", containing the names
                      and calculated values of the primitive metrics.
    """

    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    # Calculating nG for checking default graph (otherwise calculation will get en error if given default graph IRI does not exist in endpoint) 
    query_nG = """
        SELECT (COUNT(DISTINCT ?node) AS ?nodeCount)
        WHERE {
            {
                SELECT ?node 
                WHERE {
                    { ?node ?p1 ?o }       
                    UNION
                    { ?s ?p2 ?node }       
                }
            }
        }
    """

    results = _send_query(query_nG, sparql, JSON)

    nG = 0

    for binding in results["results"]["bindings"]:
        nG = int(binding["nodeCount"]["value"])

    if nG == 0:
        return pd.DataFrame([
            {"Metric": "Number of Entities",          "Value": 0},
            {"Metric": "Number of Properties",        "Value": 0},
            {"Metric": "Number of Classes",           "Value": 0},
            {"Metric": "Number of Instances",         "Value": 0},
            {"Metric": "Number of Object Properties", "Value": 0},
        ])

    # Number of distinct entities
    query_entitities = """
    SELECT (COUNT(DISTINCT ?entity) AS ?entityCount)
    WHERE {
        {
            SELECT DISTINCT ?entity
            WHERE {
                ?entity ?p ?o .
            }
        }
        UNION
        {
            SELECT DISTINCT ?entity 
            WHERE {
                ?s ?p ?entity .
                FILTER(!isLiteral(?entity))
            }
        }
    }
    """

    results = _send_query(query_entitities, sparql, JSON)

    num_entities = 0

    for binding in results["results"]["bindings"]:
        num_entities = int(binding["entityCount"]["value"])

    num_instances = _get_num_instances_ep(sparql)

    # num_classes = _get_num_classes_ep(sparql)

    classes = _get_classes_ep(sparql)

    num_classes = len(classes)

    (num_properties_t, num_properties_a) = _get_num_properties_ep(sparql)

    num_properties = num_properties_t + num_properties_a

    # Number of object properties in T-Box
    # Non-Inheritance -> excluding inheritance properties like rdfs:subPropertyOf or rdfs:subClassOf
    query_object_properties_t = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT (COUNT(DISTINCT ?property) AS ?numObjectProperties)
        WHERE {
            ?property rdf:type owl:ObjectProperty .
        }
    """

    results = _send_query(query_object_properties_t, sparql, JSON)

    num_obj_properties_t = 0

    for binding in results["results"]["bindings"]:
        num_obj_properties_t = int(binding["numObjectProperties"]["value"])

    # Number of object properties in A-Box
    # Non-Inheritance -> excluding inheritance properties like rdfs:subPropertyOf or rdfs:subClassOf
    query_object_properties_a = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT (COUNT(DISTINCT ?property) AS ?numObjectProperties)
        WHERE {
            ?s ?property ?o 
            Filter(!isLiteral(?o))
        }
    """

    num_obj_properties_a = 0

    results = _send_query(query_object_properties_a, sparql, JSON)

    for binding in results["results"]["bindings"]:
        num_obj_properties_a = int(binding["numObjectProperties"]["value"])

    num_obj_properties = num_obj_properties_t + num_obj_properties_a

    return pd.DataFrame([
        {"Metric": "Number of Entities",          "Value": int(num_entities)},
        {"Metric": "Number of Properties",        "Value": int(num_properties)},
        {"Metric": "Number of Classes",           "Value": int(num_classes)},
        {"Metric": "Number of Instances",         "Value": int(num_instances)},
        {"Metric": "Number of Object Properties", "Value": int(num_obj_properties)},
    ])

def depth_of_inh_tree_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the maximum depth of the inheritance tree in an RDF graph accessible via a SPARQL endpoint.

    This function identifies all class hierarchies by locating root classes
    (classes that have no superclass) and then traversing each hierarchy
    downward using `rdfs:subClassOf` relationships retrieved through SPARQL queries.  
    The metric represents the length (= number of edges) of the longest inheritance path 
    between a root class and a leaf class (a class with no subclasses) across the entire graph.

    The traversal is performed using a depth-first search (DFS) algorithm,
    similar in logic to the local graph version of this metric, but adapted for
    remote graphs queried through SPARQL.

    Implementation Details:

    1. Connection Setup:
        Establishes a connection to the SPARQL endpoint using the provided `endpoint_url` and 
        optional `default_graph`.

    2. Retrieve Blank Node Relationships:
        Executes a SPARQL query to identify all blank nodes (`bnodes`) that
        participate in subclass relationships. Each bnode is classified as either
        a superclass or a subclass, and a flag (`isRoot`) indicates whether it is a
        root node (i.e., a class without a superclass). The results are stored in the 
        sets `bnode_subclassOf`, `bnode_superclassOf` and `tree_roots`. This way is chosen
        because BNodes are not globally identified. So, all triples regarding BNodes
        have to be got in one query.

    3. Retrieve non-BNode Root Classes:
        Queries all named classes (`owl:Class` or `rdfs:Class`) that do not have
        a superclass (`FILTER NOT EXISTS { ?root rdfs:subClassOf ?anyClass . }`),
        excluding blank nodes. These are added to the overall set of root nodes.

    4. Depth-First Traversal (DFS):
        Starting from each identified root class, a DFS is executed to recursively
        explore subclass relationships. Cycles are prevented by checking if a node
        already appears in the current traversal path.  
        Each time a leaf class (a node with no subclasses) is reached, the current path
        length is compared to the global `max_depth_inh_tree` value, which stores the
        longest path discovered so far.

    After all root nodes have been explored, the function returns the maximum depth of the 
    inheritance hierarchy.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph to use for queries. Defaults to None.
        dec_places (int, optional): The number of decimal places to round the result to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the names and calculated value of the metric.
    """
    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    max_depth_inh_tree = 0

    # for storing sublasses of nodes    
    subclasses_cache = {} 

    # idea: 
    # 1. getting bnodes and their sub- and superclasses + the information if a bnode is a root
    # 2. getting all nodes which are root nodes without looking at bnodes
    # 3. searching for subclasses of a node:
    #   3.1 if node = uri: first looking at subclasses != bnodes, then looking at bnode subclasses explicitly
    #   3.2 if node = bnode: directly looking at bnode subclasses

        
    # 1. getting bnodes and their sub- and superclasses + the information if a bnode is a root
    # at first, the "FILTER(isBlank(?bnode))" was included in the UNION parts, but VIRUTOSO returned
    # an error with that query, so I put the Filter in the end  
    # this query only returns bnode roots which really have at least one subclass
    # --> so when comparing to local version, it can be the case that the local one finds more bnode roots,
    # but the paths with those roots will have path length = 0 !
    query_bnode_subclasses_root = """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>

        SELECT DISTINCT ?bnode ?relation ?other ?isRoot
        WHERE {
            {
                # superclasses: bnode rdfs:subClassOf other
                ?bnode rdfs:subClassOf ?other .
                
                BIND("SubclassOf" AS ?relation)
            }
            UNION
            {
                # subclasses: other rdfs:subClassOf bnode
                ?other rdfs:subClassOf ?bnode .
                
                BIND("SuperclassOf" AS ?relation)
            }

            # if node is a root node -> "yes; otherwise "no"
            BIND(
                IF(
                EXISTS {
                    { ?bnode rdf:type owl:Class } UNION { ?bnode rdf:type rdfs:Class }
                    FILTER NOT EXISTS { ?bnode rdfs:subClassOf ?anyClass }
                },
                "yes",
                "no"
                ) AS ?isRoot
            )

            FILTER(isBlank(?bnode))
        }
        ORDER BY ?bnode ?relation ?other
    """

    results = _send_query(query_bnode_subclasses_root, sparql, JSON)

    # stores superclasses of bnodes
    bnode_subclassOf = set()
    # stores subclasses of bnodes
    bnode_superclassOf = set()
    # stores roots of the tree
    tree_roots = set()

    for binding in results["results"]["bindings"]:
        bnode = binding["bnode"]
        bnode_value = bnode["value"]
        bnode_type = bnode["type"]
        
        relation = binding["relation"]["value"]
        
        node = binding["other"]
        node_value = node["value"]
        node_type = node["type"]

        bnode_isRoot = binding["isRoot"]["value"]

        if relation == "SuperclassOf":
            bnode_superclassOf.add(((bnode_value, bnode_type), (node_value, node_type)))
        else:
            bnode_subclassOf.add(((bnode_value, bnode_type), (node_value, node_type)))
        
        if bnode_isRoot == "yes":
            tree_roots.add((bnode_value, bnode_type))

    # 2. getting all nodes which are root nodes without looking at bnodes
    # ?root a owl:Class -> to ensure root is a class
    # FILTER NOT EXISTS {?root rdfs:subClassOf ?anyClass .} -> to get root which has no superclass
    query_root = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?root
        WHERE {
            {
                ?root rdf:type owl:Class .
            }
            UNION
            {
                ?root rdf:type rdfs:Class .
            }
            
            FILTER NOT EXISTS {
                ?root rdfs:subClassOf ?anyClass .
            }

            FILTER(!isBlank(?root)) # bnodes have already been covered
        }
    """

    results = _send_query(query_root, sparql, JSON)

    for binding in results["results"]["bindings"]:
        root = binding["root"]
        root_value = root["value"]
        root_type = root["type"]
        tree_roots.add((root_value, root_type))

    def get_subclasses(node):
        """
        Retrieves all direct subclasses for a given node from the SPARQL endpoint.

        This function finds all direct subclasses (`rdfs:subClassOf`) for a given `node`.
        It handles URI nodes and blank nodes differently:
        - For URI nodes, it executes a SPARQL query to find URI subclasses and
            consults a pre-computed set (`bnode_subclassOf`) for finding blank node subclasses.
        - For blank nodes, it looks up their subclasses in the pre-computed set
            (`bnode_superclassOf`).
        Results are cached to avoid redundant queries for the same node.

        Args:
            node (tuple): A tuple `(node_value, node_type)` representing the class.

        Returns:
            set: A set of tuples, where each tuple `(subclass_value, subclass_type)`
                    represents a direct subclass.
        """
        (node_value, node_type) = node

        if node in subclasses_cache:
            return subclasses_cache[node]
        
        subclasses = set()

        # 3.1 if node = uri: first looking at subclasses != bnodes, 
        # then looking at bnode subclasses explicitly
        if node_type == "uri":
            query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT DISTINCT ?child
                WHERE 
                {{
                    ?child rdfs:subClassOf <{str(node_value)}> .
                    FILTER(!isBlank(?child))
                }}
            """

            results = _send_query(query, sparql, JSON)

            for binding in results["results"]["bindings"]:
                child = binding["child"]
                child_value = child["value"]
                child_type = child["type"]

                subclasses.add((child_value, child_type)) 

            for (bnode_child, bnode_child_type), (father, _) in bnode_subclassOf:
                if father == node_value:
                    subclasses.add((bnode_child, bnode_child_type))

        # 3.2 if node = bnode: directly looking at bnode subclasses 
        else:
            for (bnode_father, _), (child, child_type) in bnode_superclassOf:
                if bnode_father == node_value:
                    subclasses.add((child, child_type))

        subclasses_cache[node] = subclasses

        return subclasses
    
    def dfs(path, node):
        """
        Performs a depth-first search (DFS) to find all subclass paths from a node.

        This recursive function explores the inheritance hierarchy starting from the
        given `node`. It avoids cycles by checking if a node's value has already
        been visited in the current `path`. When a path terminates (i.e., a class
        with no subclasses is reached), it updates the global metrics
        `num_paths_inh_tree` and `max_depth_inh_tree`.

        Args:
            path (list): The list of node values representing the current path from
                            the root to the parent of the current `node`.
            node (tuple): The current node `(node_value, node_type)` to visit in
                            the DFS traversal.
        """

        nonlocal max_depth_inh_tree

        node_value, _ = node

        # skip node if it already is in path (avoiding cycles)
        if node_value in path:
            return
        
        # add current node to path
        path.append(node_value)    #path.append(str(node_value))

        subclasses = list(get_subclasses(node)) 

        if not subclasses:
            max_depth_inh_tree = max(max_depth_inh_tree, len(path) - 1)

        else:
            for subclass in subclasses:
                #print("\node: " + str(node) + " - neighbor: " + str(neighbor))
                
                dfs(path, subclass)

        # remove node from the path to find next path
        path.pop()

    for root in tree_roots:
        dfs([], root)

    return pd.DataFrame([
        {"Metric": "Depth of Inheritance Tree", "Value": int(max_depth_inh_tree)},
    ])


def tbox_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates T-Box (Terminological Box) metrics for a graph via a SPARQL endpoint 
    (Property Class Ratio, Class Property Ratio, Inheritance Richness, Attribute Richness).

    The metrics are calculated as follows:
    - **Property Class Ratio**: The ratio of T-Box properties to the total number of classes.
    - **Class Property Ratio**: The inverse of the Property Class Ratio.
    - **Inheritance Richness**: The ratio of `rdfs:subClassOf` relationships to the total number of classes.
    - **Attribute Richness**: The ratio of T-Box datatype properties to the total number of classes.

    This function handles division-by-zero cases by returning 0 for a ratio if the
    denominator is zero.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph for queries. Defaults to None.
        dec_places (int, optional): The number of decimal places to round the results to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the names and calculated values of the T-Box metrics.
    """

    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    # Inheritance Richness = average number of subclasses per class (source 227 - page 9) 
    # Getting number of all subclasses
    query_subclasses = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(*) AS ?numInheritanceRelations)
        WHERE {
            ?subclass rdfs:subClassOf ?superclass .
        }
    """

    num_subclasses = 0

    results = _send_query(query_subclasses, sparql, JSON)

    for binding in results["results"]["bindings"]:
        num_subclasses = int(binding["numInheritanceRelations"]["value"])

    # Getting number of datatype properties in T-Box
    query_datatype_properties_t = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(DISTINCT ?property) AS ?numDatatypeProperties)
        WHERE {
            ?property rdf:type owl:DatatypeProperty .
        }
    """

    num_datatype_properties = 0

    results = _send_query(query_datatype_properties_t, sparql, JSON)

    for binding in results["results"]["bindings"]:
        num_datatype_properties_t = int(binding["numDatatypeProperties"]["value"])

    # this function calculates metrics regarding T-Box --> we are only interested in T-Box properties
    num_datatype_properties = num_datatype_properties_t

    # num_classes = _get_num_classes_ep(sparql)

    classes = _get_classes_ep(sparql)

    num_classes = len(classes)

    (num_properties_t, _) = _get_num_properties_ep(sparql, True, False)

    # this function calculates metrics regarding T-Box --> we are only interested in T-Box properties
    num_properties = num_properties_t

    # Property Class Ratio - Inheritance Richness - Attribute Richness
    if num_classes > 0:
        prop_class_ratio = round(num_properties / num_classes, dec_places) 
        inheritance_richness = round(num_subclasses / num_classes, dec_places) 
        attr_richness = round(num_datatype_properties / num_classes, dec_places)

    else:
        # source 172 - page: assumes that classes must exist for properties to exist (Number of Properties, Number of CLasses > 1)
        # I assume: no classes -> ratio = 0
        prop_class_ratio = 0.0
        inheritance_richness = 0.0
        attr_richness = 0.0

    # Class Property Ratio
    if num_properties > 0:
        class_prop_ratio = round(num_classes / num_properties, dec_places) 

    else:
    # metric is not defined for num_properties = 0
    # I assume: no properties -> ratio = 0
        class_prop_ratio = 0

    return pd.DataFrame([
        {"Metric": "Property Class Ratio", "Value": float(prop_class_ratio)},
        {"Metric": "Class Property Ratio", "Value": float(class_prop_ratio)},
        {"Metric": "Inheritance Richness", "Value": float(inheritance_richness)},
        {"Metric": "Attribute Richness", "Value": float(attr_richness)},
    ])

def abox_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates some A-Box metrics for a graph via a SPARQL endpoint 
    (Average Class Connectivity, Average Population).

    The metrics are calculated as follows:
    - **Average Class Connectivity**: Measures the average number of relationships
      that instances of a class have with instances of other classes. It executes
      a SPARQL query to find relationships between instances of different classes,
      ignoring `rdf:type` properties. The total connectivity is then averaged
      across all classes.
    - **Average Population**: The average number of instances per class. It is
      calculated by dividing the total number of individuals from the `_get_num_instances_ep` function
      by the total number of classes (from `_get_num_classes_ep`).

    The function handles division-by-zero cases by returning 0 if no classes are present.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph for queries. Defaults to None.
        dec_places (int, optional): The number of decimal places to round the results to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the names and calculated values of the A-Box metrics.
    """

    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    # num_classes = _get_num_classes_ep(sparql)

    classes = _get_classes_ep(sparql)

    num_classes = len(classes)

    # if num_classes == 0 --> Average Class Connectivity = Average Population = 0
    if num_classes == 0:
        return pd.DataFrame([
            {"Metric": "Average Class Connectivity", "Value":0},
            {"Metric": "Average Population", "Value": 0},
        ])

    num_instances = _get_num_instances_ep(sparql)

    # Average Class Connectivity
    # Connectivity of a class is defined as the total number of relationships instances of 
    # the class have with instances of other classes (source 227 - page 10)

    # looking for number of triples (c1, p, c2) or (c3, p, c1) for each class with instances c1
    # c1, c2 are instances of classes 
    # c1 != c2,c3
    # property != rdf:type because we are not interested in the class relationships 
    query_class_connectivity = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT ?class (COUNT(*) AS ?connectivity)
        WHERE {
            {
                ?instance ?property ?target .

                ?instance rdf:type ?class .
                ?target rdf:type ?targetClass .

                FILTER(?property != rdf:type)
                FILTER(?class != ?targetClass)
            }
            UNION
            {
                ?instance ?property ?target .

                ?target rdf:type ?class .
                ?instance rdf:type ?targetClass .

                FILTER(?property != rdf:type)
                FILTER(?class != ?targetClass)
            }
        }
        GROUP BY ?class
    """
    results = _send_query(query_class_connectivity, sparql, JSON)

    # for storing connectivvity of each class
    class_connectivity_list = []
    sum_connectivities = 0

    for binding in results["results"]["bindings"]:
        class_name = binding["class"]["value"]
        connectivity = int(binding["connectivity"]["value"])
        class_connectivity_list.append((class_name, connectivity))
        sum_connectivities += connectivity

    avg_class_connectivity = round(sum_connectivities / num_classes, dec_places)
    avg_population = round(num_instances / num_classes, dec_places)

    return pd.DataFrame([
        {"Metric": "Average Class Connectivity", "Value": float(avg_class_connectivity)},
        {"Metric": "Average Population", "Value": float(avg_population)},
    ])

def cohesion_endpoint(endpoint_url: str, default_graph: str | None = None, dec_places: int = 2) -> pd.DataFrame:
    """
    Calculates the cohesion metric of an RDF graph accessible via a SPARQL endpoint.

    This function determines how many disconnected subgraphs (connected components) 
    exist within the RDF graph available at the specified SPARQL endpoint.  
    Each connected component represents a subset of nodes that are mutually reachable 
    through RDF triples — either as subjects or objects.  
    The number of such components serves as a measure of the graph's cohesion:
    a higher number indicates a more fragmented (less cohesive) structure.

    Implementation Details:

    1. Connection Setup:
        Establishes a connection to the SPARQL endpoint using the provided `endpoint_url` and 
        optional `default_graph`.

    2. Node Extraction:
        Executes a SPARQL query to collect all nodes (both subjects and objects) 
        along with their direct neighbors.  
        This step ensures that all possible RDF nodes — including URIs, blank nodes, and literals — 
        are included in the analysis.
    
    3. Blank Node and Literal Handling:
        Since Blank Nodes are not globally identifiable and Literals can cause endpoint errors 
        due to their length or datatype complexity, their relationships are handled locally.
        From the results of the node query:
        - `bnode_neighbors` stores all connections where a blank node appears as subject or object.
        - `literal_neighbors` stores all connections between literals and their neighboring resources.
        These mappings are reused during traversal instead of querying the endpoint again.

    4. Connected Component Search: 
        The algorithm iteratively identifies all connected components:
        - Starts from an arbitrary unvisited node.  
        - Expands from this node by repeatedly retrieving unvisited neighbors using `get_new_neighbors`.  
        - Once no new nodes can be reached, the discovered set of nodes is marked as one component.  
        - The process repeats until all nodes have been visited.

    The total number of discovered components is returned as the cohesion metric.

    Args:
        endpoint_url (str): The URL of the SPARQL endpoint.
        default_graph (str | None, optional): The URI of the default graph for queries. Defaults to None.
        dec_places (int, optional): The number of decimal places to round the results to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing the name and calculated values of the cohesion metric.
    """

    sparql = get_sparql_from_endpoint(endpoint_url, default_graph)

    # for storing neighbors of nodes
    neighbors_cache = {}

    def get_new_neighbors(node, sparql, all_nodes, visited):
        """
        Retrieves all unvisited neighboring nodes of a given RDF node from a SPARQL endpoint.

        This function determines all nodes directly connected to the given `node` via RDF triples
        by querying the remote graph through the provided SPARQL connection. It handles different
        RDF node types — IRIs, blank nodes (bnodes), and literals — using dedicated logic for each case
        and employs a cache (`neighbors_cache`) to avoid redundant queries.

        - Literals / Typed Literals:
            Uses the precomputed `literal_neighbors` set to find all subjects that are connected
            to the given literal. Only neighbors that are part of `all_nodes` and not yet visited
            are included. If a discovered neighbor is not part of `all_nodes`, a `ValueError` is 
            raised, as this indicates an incomplete or inconsistent RDF graph.

        - Blank Nodes:
            Looks up all triples from the precomputed `bnode_neighbors` set where the given blank node
            appears as the subject. The connected objects are returned as neighbors if they are valid
            and unvisited. If a discovered neighbor is not part of `all_nodes`, a `ValueError` is raised,
            as this indicates an incomplete or inconsistent RDF graph.

        - URI Nodes:
            Performs a SPARQL query to find all resources connected to the node by any predicate.
            Blank nodes are excluded in this query (`FILTER(!isBlank(?neighbor))`).  
            The function then adds any associated blank nodes from `bnode_neighbors` that link back to
            the current node. If a discovered neighbor is not part of `all_nodes`, a `ValueError` is raised,
            as this indicates an incomplete or inconsistent RDF graph.

        All retrieved neighbors are cached in `neighbors_cache` for reuse in later lookups.

        Args:
            node (tuple): The RDF node for which to find unvisited neighbors, represented as 
                        (value, type, lang_tag, datatype).
            sparql: A SPARQLWrapper connection object used to send queries to the endpoint.
            all_nodes (set): A set of all nodes in the graph used to validate neighbor membership.
            visited (set): A set of all nodes that have already been visited during whole process.

        Returns:
            set: A set of unvisited neighboring nodes represented as tuples 
                (value, type, lang_tag, datatype). Returns an empty set if no valid neighbors exist.
        
        Raises:
            ValueError: If a discovered neighbor is not present in `all_nodes`,  indicating potential 
                        data inconsistency or incomplete graph retrieval.
        """
        nonlocal bnode_neighbors
        nonlocal literal_neighbors
        nonlocal neighbors_cache

        neighbors = set()

        (node_value, node_type, _, _) = node 

        # TODO: ich denke das nachschauen im cache is unnötig -> wenn cache eintrag existiert -> return set() !
        # Check if node is in cache
        if node in neighbors_cache:
            return neighbors_cache[node]
        
        
        if node_type in ["literal", "typed-literal"]:
            #node = (node_value, node_type, lang_tag, datatype)
            
            for literal, neighbor in literal_neighbors:
                # check if literal == literal we are searching neighbors for
                if literal == node:
                    # check if neighbor has not been visited yet
                    if ((neighbor in all_nodes) and (neighbor not in visited)):
                        neighbors.add(neighbor)

                    # in this case, graph provided by database is inconsistent
                    elif neighbor not in visited:
                    
                        raise ValueError(
                            f"Neighbor {neighbor} (from outgoing edge) not found in all_nodes. "
                            f"This may indicate an incomplete or inconsistent RDF graph."
                        )  

        elif node_type == "bnode":
            
            # we can directly look at the bnode_neighbors set
            for bnode, neighbor in bnode_neighbors:
                #(bnode_value, _, _, _) = bnode
                #(neighbor_val, _, _, _) = neighbor

                # check if bnode == bnode we are searching neighbors for
                if bnode == node:
                    #(neighbor_val, _, _, _) = neighbor

                    # check if neighbors has not been visited yet
                    if ((neighbor in all_nodes) and (neighbor not in visited)):
                        neighbors.add(neighbor)

                    elif neighbor not in visited:
                        raise ValueError(
                            f"Neighbor {neighbor} (from outgoing edge) not found in all_nodes. "
                            f"This may indicate an incomplete or inconsistent RDF graph."
                        )   
        
        # TODO: ich denke ich sollte das hier so umändern, dass literal_neighbors bei uri nodes verwendet wird!

        # in this case, node is of type uri 
        else:

            #node = (node_value, node_type, lang_tag, datatype)

            # looking for non-Bnode neighbors, Bnode neighbors will be handled afterwards
            query = f"""
            SELECT DISTINCT ?neighbor 
            WHERE {{
                {{ <{node_value}> ?p1 ?neighbor .  FILTER(!isBlank(?neighbor))}}
                UNION
                {{ ?neighbor ?p2 <{node_value}> .  FILTER(!isBlank(?neighbor))}}
            }}
            """

            pot_new_nghs = _send_query(query, sparql, JSON)
            
            for binding in pot_new_nghs["results"]["bindings"]:
                pot_new_ngh_obj = binding["neighbor"]
                pot_new_ngh_val = pot_new_ngh_obj["value"]
                pot_new_ngh_type = pot_new_ngh_obj["type"]
                # i know lang_tag & datatype will be None, but using get function is important for variable types!!
                pot_new_ngh = pot_new_ngh_val, pot_new_ngh_type, pot_new_ngh_obj.get("xml:lang"), pot_new_ngh_obj.get("datatype")
                #debug_print("Got neighbor: " + str(pot_new_ngh_val))
                
                if ((pot_new_ngh in all_nodes) and (pot_new_ngh not in visited)):
                    neighbors.add(pot_new_ngh)
                
                # if this is the case, graph provided by database is inconsistent
                elif pot_new_ngh not in visited:
                    raise ValueError(
                            f"Neighbor {pot_new_ngh} (from outgoing edge) not found in all_nodes. "
                            f"This may indicate an incomplete or inconsistent RDF graph."
                        ) 
                
            for bnode, neighbor in bnode_neighbors:
                #(bnode_value, _, _, _) = bnode
                #(neighbor_val, _, _, _) = neighbor

                # check if uri == uri we are searching neighbors for
                if neighbor == node:
                    # check if neighbors has not been visited yet
                    if ((bnode in all_nodes) and (bnode not in visited)):
                        neighbors.add(bnode)

                    # if this is the case, graph provided by database is inconsistent
                    elif neighbor not in visited:
                        raise ValueError(
                            f"Neighbor {neighbor} (from outgoing edge) not found in all_nodes. "
                            f"This may indicate an incomplete or inconsistent RDF graph."
                        ) 
        
        neighbors_cache[node] = neighbors
        
        return neighbors

    # Getting all nodes with their neighbors for:
    # 1. storing all nodes in graph and
    # 2. getting all neighbors of bnodes at once
    # 3. getting all neighbors of literals to store them
    # this way needed (look at function documentation for info)
    query_all_nodes = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?node ?neighbor
        WHERE {
            {
                ?node ?p ?o .
                BIND(?o AS ?neighbor)
            }
            UNION
            {
                ?s ?p ?node .
                BIND(?s AS ?neighbor)
            }
        }
    """

    results = _send_query(query_all_nodes, sparql, JSON)

    all_nodes = set()

    # storing all bnodes with their neighbors
    bnode_neighbors = set()

    # storing all literals with their neighbors
    literal_neighbors = set()

    for binding in results["results"]["bindings"]:
        #print("TEST " + node_obj)
        node_obj = binding["node"]
        #print("TEST " + node_obj)
        node_value = node_obj["value"]
        node_type = node_obj["type"]
        node_lang = node_obj.get("xml:lang") 
        node_dtype = node_obj.get("datatype") 
        
        ngh_obj = binding["neighbor"]
        ngh_value = ngh_obj["value"]
        ngh_type = ngh_obj["type"]
        ngh_lang = ngh_obj.get("xml:lang") 
        ngh_dtype = ngh_obj.get("datatype") 

        all_nodes.add((node_value, node_type, node_lang, node_dtype))    

        # if node is a blank node -> add (node, neighbor) to bnode_neighbors
        if node_type == "bnode":
            bnode = (node_value, node_type, node_lang, node_dtype)
            neighbor = (ngh_value, ngh_type, ngh_lang, ngh_dtype)
            bnode_neighbors.add((bnode, neighbor))
            #bnode_neighbors.append((bnode, neighbor))
        
        # if neighbor is a blank node -> add (neighbor, node) to bnode_neighbors
        elif ngh_type == "bnode":
            bnode = (ngh_value, ngh_type, ngh_lang, ngh_dtype)
            neighbor = (node_value, node_type, node_lang, node_dtype)
            
            bnode_neighbors.add((bnode, neighbor))
        
        # if node is a (typed) literal node -> add (node, neighbor) to literal_neighbors
        if node_type in ["literal", "typed-literal"]:
            lit = (node_value, node_type, node_lang, node_dtype)
            neighbor = (ngh_value, ngh_type, ngh_lang, ngh_dtype)
            
            literal_neighbors.add((lit, neighbor))
        
        # if neighbor is a (typed) literal node -> add (neighbor, node) to literal_neighbors
        elif ngh_type in ["literal", "typed-literal"]:
            lit = (ngh_value, ngh_type, ngh_lang, ngh_dtype)
            neighbor = (node_value, node_type, node_lang, node_dtype)
            
            # bnode_neighbors.add((lit, neighbor))
            literal_neighbors.add((lit, neighbor))

    # # had to do it that way because:
    # # 1) looking at neighbors of one literal directly can be complicated because of
    # # possible language tags or datatype tags
    # # 2) literal strings can be very long and full with new lines etc. -> some endpoints
    # # do not accept the query then
    # # -> so i have to ask for every literal so that the endpoint gives me the literals
    # query_literal_neighbors = """
    #     SELECT DISTINCT ?literal ?neighbor 
    #     WHERE {
    #             ?neighbor ?p ?literal .
    #             FILTER(isLiteral(?literal))
    #             FILTER(!isBlank(?neighbor))
    #     }
    # """
    # results = _send_query(query_literal_neighbors, sparql, JSON)

    # for binding in results["results"]["bindings"]:
    #     lit_obj = binding["literal"]
    #     ngh_obj = binding["neighbor"]

    #     lit = (lit_obj["value"], lit_obj["type"], lit_obj.get("xml:lang"), lit_obj.get("datatype"))
    #     neighbor = (ngh_obj["value"], ngh_obj["type"], ngh_obj.get("xml:lang"), ngh_obj.get("datatype"))

    #     literal_neighbors.add((lit, neighbor))

    # describes number of independent components of graph (like subgraphs which are connected internally)
    components = 0

    # describes visited / discovered nodes
    visited = set()

    # stores each component with its nodes
    components_nodes = []

    ## algorithm: searches for components of graph
    while visited != all_nodes:

        # Choose first unvisited node
        # iter creates iterator for set - next gives next element in set
        start_node = next(iter(all_nodes - visited))
        node_value, node_type, lang_tag, datatype = start_node

        frontier = {start_node}
        visited.add(start_node)

        # starting new component for graph
        current_component = set([start_node])

        while frontier:

            new_frontier = set()
            
            for node in frontier:
            
                new_visited_neighbors = set()
                # getting just all completely new neighbors
                new_visited_neighbors = get_new_neighbors(node, sparql, all_nodes, visited)
                
                # neighbors just consists of new neighbors 
                visited |= new_visited_neighbors

                current_component |= new_visited_neighbors

                if visited == all_nodes:
                    break

                new_frontier |= new_visited_neighbors
            

            if visited == all_nodes:
                break

            frontier = new_frontier

        components += 1
        components_nodes.append(current_component)

    return pd.DataFrame([
        {"Metric": "Cohesion", "Value": int(components)},
    ])

