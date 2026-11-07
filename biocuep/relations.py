"""
Stage 4: Semantic Relation Extraction from BKGs (Hetionet & PrimeKG)
Implements Section 3.4 of the paper:
Retrieves first-level semantic relations (h, r, t) from the BKG for every concept in LS and LQ,
forming local semantic neighborhoods RS and RQ.
"""

import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


def connect_to_neo4j(uri: str, username: str = "neo4j", password: str = ""):
    """
    Establishes connection to Neo4j graph database.
    """
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(username, password))
        return driver
    except Exception as e:
        logger.warning(f"Could not connect to Neo4j at {uri}: {e}")
        return None


def fetch_first_level_triples_neo4j(
    driver,
    node_name: str,
    node_name_attribute: str = "name",
    allowed_attributes: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Executes Cypher query to retrieve all directly connected 1st-level relations:
    MATCH (n {name: $node_name})-[r]-(m) RETURN type(r) as relation, collect(m.name) as nodes
    """
    if driver is None or not node_name:
        return []

    triples: Set[tuple] = set()
    try:
        with driver.session() as session:
            query = (
                f"MATCH (n {{{node_name_attribute}: $node_name}})-[r]-(m) "
                f"RETURN type(r) as relation, collect(m.{node_name_attribute}) as nodes"
            )
            result = session.run(query, node_name=node_name)
            for record in result:
                rel = record["relation"]
                for neighbor in record["nodes"]:
                    if neighbor:
                        triples.add((node_name, rel, neighbor))
    except Exception as ex:
        logger.debug(f"Neo4j query failed for node '{node_name}': {ex}")

    return [{"source": h, "relation": r, "target_nodes": t} for h, r, t in triples]


def extract_relations_for_concepts(
    concept_map: Dict[str, str],
    driver=None,
    node_name_attribute: str = "name",
    offline_graph: Optional[Dict[str, List[Dict[str, str]]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts 1st-level relations for all linked concepts in a concept map {mention: node_name}.
    Returns:
      {concept: {"triples": [{"source": ..., "relation": ..., "target_nodes": ...}, ...]}}
    """
    result: Dict[str, Dict[str, Any]] = {}

    for mention, node_name in concept_map.items():
        if not node_name:
            continue

        triples = []
        # Check offline graph / cache first
        if offline_graph and node_name in offline_graph:
            triples = offline_graph[node_name]
        elif driver is not None:
            triples = fetch_first_level_triples_neo4j(driver, node_name, node_name_attribute)

        result[mention] = {"triples": triples}

    return result


def extract_relations_for_sample(
    sample: Dict[str, Any],
    kg_name: str,
    driver=None,
    node_name_attribute: str = "name",
    offline_graph: Optional[Dict[str, List[Dict[str, str]]]] = None
) -> Dict[str, Any]:
    """
    Extracts relations for both context and question concepts for a single dataset sample.
    """
    q_concepts = sample.get(f"{kg_name}_concepts_question", {})
    ctx_concepts = sample.get(f"{kg_name}_concepts_context", {})

    q_kg_data = extract_relations_for_concepts(q_concepts, driver, node_name_attribute, offline_graph)
    ctx_kg_data = extract_relations_for_concepts(ctx_concepts, driver, node_name_attribute, offline_graph)

    sample[f"{kg_name}_concepts_question_with_kg_data"] = q_kg_data
    sample[f"{kg_name}_concepts_context_with_kg_data"] = ctx_kg_data
    return sample
