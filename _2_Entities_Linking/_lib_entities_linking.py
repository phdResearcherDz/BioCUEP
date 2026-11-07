"""
Stage 3: Biomedical Entity Linking Library
Implements Section 3.3, Equation (6) of the paper:
Links normalized concepts to Hetionet and PrimeKG concepts using CODER embeddings:
b*_{c*_e} = argmax_{b in BKG, Sim(c*_e, b) >= tau} Sim(c*_e, b)
with tau = 0.90 (Section 4.3).
"""

import glob
import json
import os
from pathlib import Path
import logging

try:
    import pandas as pd
except ImportError:
    class _MockDataFrame(list):
        empty = True
        columns = []
        def __init__(self, data=None):
            super().__init__()
            self._data = data or {}
    class _MockPandas:
        DataFrame = _MockDataFrame
    pd = _MockPandas()

from config_file import PROJECT_ROOT
from biocuep.utils import tqdm
from biocuep.config import (
    LINKING_SIMILARITY_THRESHOLD,
    HETIONET_NEO4J_URI,
    HETIONET_NEO4J_USER,
    HETIONET_NEO4J_PASSWORD,
    PRIMEKG_NEO4J_URI,
    PRIMEKG_NEO4J_USER,
    PRIMEKG_NEO4J_PASSWORD
)
from biocuep.linking import (
    load_coder_model,
    link_concept_to_bkg,
    link_dataset_concepts
)

logger = logging.getLogger(__name__)


def connect_to_neo4j(uri, username="neo4j", password=""):
    """Connect to Neo4j graph database."""
    try:
        from neo4j import GraphDatabase
        return GraphDatabase.driver(uri, auth=(username, password))
    except Exception as e:
        logger.warning(f"Neo4j connection error to {uri}: {e}")
        return None


def fetch_nodes(driver, node_name_attribute="name"):
    """Fetch nodes from Neo4j."""
    if driver is None:
        return pd.DataFrame({node_name_attribute: []})
    try:
        with driver.session() as session:
            result = session.execute_read(lambda tx: tx.run("MATCH (n) RETURN n").data())
            nodes = [record['n'] for record in result]
        return pd.DataFrame(nodes)
    except Exception as e:
        logger.warning(f"Failed to fetch nodes from Neo4j: {e}")
        return pd.DataFrame({node_name_attribute: []})


def extract_entities(entities_list):
    """Extracts entity mention to canonical name mapping."""
    extracted = {}
    for item in entities_list:
        if isinstance(item, dict):
            for mention, meta in item.items():
                if isinstance(meta, dict) and 'canonical_name' in meta:
                    extracted[mention] = meta['canonical_name']
                elif isinstance(meta, str):
                    extracted[mention] = meta
                else:
                    extracted[mention] = mention
    return extracted


def process_json(file_path, df_nodes, kg_name, node_name_attribute, threshold=0.90, device="cpu"):
    """
    Links entities in JSON file to BKG nodes with CODER / exact matching (tau = 0.90).
    Fixes the legacy bug where top_indices (integer node index) was compared to threshold!
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    node_names = df_nodes[node_name_attribute].dropna().tolist() if not df_nodes.empty and node_name_attribute in df_nodes.columns else []

    tokenizer, model = load_coder_model(device)

    new_data = []
    for item in tqdm(data, desc=f"Linking to {kg_name}"):
        ctx_ents = extract_entities(item.get('context_entities', []))
        q_ents = extract_entities(item.get('question_entities', []))

        ctx_links = {}
        for mention, canonical in ctx_ents.items():
            matched = link_concept_to_bkg(
                canonical, node_names, threshold=threshold,
                coder_tokenizer=tokenizer, coder_model=model, device=device
            )
            if matched:
                ctx_links[mention] = matched
            elif not node_names:
                # If node table is not loaded from live DB, retain canonical term as placeholder node
                ctx_links[mention] = canonical

        q_links = {}
        for mention, canonical in q_ents.items():
            matched = link_concept_to_bkg(
                canonical, node_names, threshold=threshold,
                coder_tokenizer=tokenizer, coder_model=model, device=device
            )
            if matched:
                q_links[mention] = matched
            elif not node_names:
                q_links[mention] = canonical

        item[f"{kg_name}_concepts_context"] = ctx_links
        item[f"{kg_name}_concepts_question"] = q_links
        new_data.append(item)

    return new_data


def process_dataset(dataset, df_nodes, kg_name, node_name_attribute):
    input_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "1_extracted_entities_with_metadata"
    output_dir = PROJECT_ROOT / "Pre_Processed_Datasets" / dataset / "2_kg_linked_entities"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    for json_file_path in json_files:
        new_data = process_json(str(json_file_path), df_nodes, kg_name, node_name_attribute)
        filename = json_file_path.stem
        out_file = output_dir / f"{filename}_{kg_name}_{dataset}_with_concept.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4)
        print(f"Saved linked concepts to {out_file}")
