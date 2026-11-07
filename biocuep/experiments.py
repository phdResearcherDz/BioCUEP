"""
BioCUEP Ablation Study Experiments
Implements Table 2 and Section 5 of the paper:
Defines the 14 experimental configurations toggling definitions, relation filtering (Sim vs QC),
and knowledge graph sources (Hetionet, PrimeKG, or both).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from biocuep.formatting import (
    assemble_enriched_input,
    extract_definitions_from_sample,
)

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    exp_id: str
    name: str
    include_definitions: bool
    filter_type: Optional[str]  # 'sim', 'qc', or None
    kgs: List[str]             # ['hetionet'], ['primekg'], or ['hetionet', 'primekg']
    description: str = ""


# Table 2: 14 Experimental Configurations
EXPERIMENT_CONFIGS: Dict[str, AblationConfig] = {
    "baseline": AblationConfig(
        exp_id="baseline",
        name="Baseline (Context-only)",
        include_definitions=False,
        filter_type=None,
        kgs=[],
        description="Standard QA context without definitions or KG relations."
    ),
    "exp1": AblationConfig(
        exp_id="exp1",
        name="Exp 1 (PBCA: UMLS Definitions Only)",
        include_definitions=True,
        filter_type=None,
        kgs=[],
        description="Enriches context with UMLS concept definitions only."
    ),
    "exp2": AblationConfig(
        exp_id="exp2",
        name="Exp 2 (Def + Sim + PrimeKG)",
        include_definitions=True,
        filter_type="sim",
        kgs=["primekg"],
        description="Definitions + TAS-B similarity filtering over PrimeKG."
    ),
    "exp3": AblationConfig(
        exp_id="exp3",
        name="Exp 3 (Def + Sim + Hetionet)",
        include_definitions=True,
        filter_type="sim",
        kgs=["hetionet"],
        description="Definitions + TAS-B similarity filtering over Hetionet."
    ),
    "exp4": AblationConfig(
        exp_id="exp4",
        name="Exp 4 (Def + Sim + Both KGs)",
        include_definitions=True,
        filter_type="sim",
        kgs=["hetionet", "primekg"],
        description="Definitions + TAS-B similarity filtering over Hetionet and PrimeKG."
    ),
    "exp5": AblationConfig(
        exp_id="exp5",
        name="Exp 5 (Def + QC + PrimeKG)",
        include_definitions=True,
        filter_type="qc",
        kgs=["primekg"],
        description="Definitions + Entity-link (QC) filtering over PrimeKG."
    ),
    "exp6": AblationConfig(
        exp_id="exp6",
        name="Exp 6 (Def + QC + Hetionet)",
        include_definitions=True,
        filter_type="qc",
        kgs=["hetionet"],
        description="Definitions + Entity-link (QC) filtering over Hetionet."
    ),
    "exp7": AblationConfig(
        exp_id="exp7",
        name="Exp 7 (Def + QC + Both KGs)",
        include_definitions=True,
        filter_type="qc",
        kgs=["hetionet", "primekg"],
        description="Definitions + Entity-link (QC) filtering over Hetionet and PrimeKG."
    ),
    "exp8": AblationConfig(
        exp_id="exp8",
        name="Exp 8 (Sim + PrimeKG, No Def)",
        include_definitions=False,
        filter_type="sim",
        kgs=["primekg"],
        description="TAS-B similarity filtered PrimeKG relations without definitions."
    ),
    "exp9": AblationConfig(
        exp_id="exp9",
        name="Exp 9 (Sim + Hetionet, No Def)",
        include_definitions=False,
        filter_type="sim",
        kgs=["hetionet"],
        description="TAS-B similarity filtered Hetionet relations without definitions."
    ),
    "exp10": AblationConfig(
        exp_id="exp10",
        name="Exp 10 (Sim + Both KGs, No Def)",
        include_definitions=False,
        filter_type="sim",
        kgs=["hetionet", "primekg"],
        description="TAS-B similarity filtered relations from both KGs without definitions."
    ),
    "exp11": AblationConfig(
        exp_id="exp11",
        name="Exp 11 (QC + PrimeKG, No Def)",
        include_definitions=False,
        filter_type="qc",
        kgs=["primekg"],
        description="QC filtered PrimeKG relations without definitions."
    ),
    "exp12": AblationConfig(
        exp_id="exp12",
        name="Exp 12 (QC + Hetionet, No Def)",
        include_definitions=False,
        filter_type="qc",
        kgs=["hetionet"],
        description="QC filtered Hetionet relations without definitions."
    ),
    "exp13": AblationConfig(
        exp_id="exp13",
        name="Exp 13 (QC + Both KGs, No Def)",
        include_definitions=False,
        filter_type="qc",
        kgs=["hetionet", "primekg"],
        description="QC filtered relations from both Hetionet and PrimeKG without definitions."
    ),
}


def get_experiment_config(exp_key: str) -> AblationConfig:
    """Retrieve config by key (e.g. 'exp4', 'baseline', '4')."""
    key = str(exp_key).strip().lower()
    if not key.startswith("exp") and key.isdigit():
        key = f"exp{key}"
    if key not in EXPERIMENT_CONFIGS:
        raise KeyError(f"Unknown experiment '{exp_key}'. Choose from: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[key]


def format_sample_for_experiment(
    sample: Dict[str, Any],
    config: AblationConfig,
    tokenizer=None,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Assembles a sample's enriched input according to the chosen AblationConfig (Table 2).
    """
    question = sample.get("question", "")
    context = sample.get("context", "")

    # Gather definitions if enabled
    definitions = {}
    if config.include_definitions:
        definitions = extract_definitions_from_sample(sample)

    # Gather relations from requested KGs
    relations: List[Dict[str, str]] = []
    if config.filter_type:
        filter_suffix = "sm" if config.filter_type == "sim" else "qc"
        for kg in config.kgs:
            kg_rels = sample.get(f"{kg}_relevant_relations_{filter_suffix}", [])
            if not kg_rels and f"{kg}_relevant_relations" in sample:
                kg_rels = sample[f"{kg}_relevant_relations"]
            relations.extend(kg_rels)

    # Deduplicate relations
    seen = set()
    unique_relations = []
    for r in relations:
        key = (r.get("source", "").lower(), r.get("relation", "").lower(), r.get("target_nodes", "").lower())
        if key not in seen and key[0] and key[2]:
            seen.add(key)
            unique_relations.append(r)

    # Assemble enriched input adhering to atomic truncation and token budget
    assembly = assemble_enriched_input(
        question=question,
        context=context,
        relations=unique_relations,
        definitions=definitions,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        include_relations=bool(config.filter_type and unique_relations),
        include_definitions=bool(config.include_definitions and definitions)
    )

    result = dict(sample)
    result["experiment_id"] = config.exp_id
    result["experiment_name"] = config.name
    result["enriched_input"] = assembly["enriched_input"]
    result["metadata"] = assembly["metadata"]
    result["token_count"] = assembly["total_tokens"]
    return result
