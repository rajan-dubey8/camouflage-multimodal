"""
Neo4j Camouflage Knowledge Graph Ingestion Script - V2
Restructured: Camouflage properties belong to Environment, not Organism
Colors and Textures are shared nodes between Organism and Environment

Graph Structure:
    Organism ─hasColor→ Color ←hasColor─ Environment
    Organism ─hasTexture→ Texture ←hasTexture─ Environment  
    Organism ─hasPattern→ Pattern
    Environment ─hasCamouflageAssessment→ CamouflageAssessment
    CamouflageAssessment ─hasSimilarity→ SimilarityMetric
    ObservationContext ─hasOrganism→ Organism
    ObservationContext ─observedIn→ Environment
"""

import os
import json
from typing import Dict, List, Any, Set
from neo4j import GraphDatabase
from tqdm import tqdm
import logging

# Configuration
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
TARGET_DB = os.getenv("TARGET_DB")

ANNOTATION_DIR = "./annotations"
PROCESSED_LOG = "./processed_files.txt"
BATCH_SIZE = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
#           NORMALIZATION 
# ============================================================

COLOR_MAPPING = {
    "olive green": "yellow-green", "olive-green": "yellow-green",
    "light yellowish-beige": "beige", "yellowish-beige": "beige",
    "pale blue-grey": "blue-gray", "blue-grey": "blue-gray",
    "light orange": "orange", "light pinkish-white": "pink-white",
    "sandy brown": "sandy-brown", "dark green": "green-dark",
    "light green": "green-light", "dark water": "water-dark",
    "murky blue": "blue-murky", "deep blue": "blue-deep",
    "brownish-green": "brown-green", "translucent": "transparent",
}

TEXTURE_MAPPING = {
    "gravel": "pebbled", "rocky": "rough", "smooth": "smooth",
    "scaly": "scaled", "tentacled": "tentacle-textured",
    "root-like": "fibrous", "vegetation": "leafy", "coral": "coral-textured",
}

PATTERN_MAPPING = {
    "disruptive pattern": "disruptive", "shape disruption": "disruptive",
    "spotted": "spotted", "striped": "striped", "uniform": "uniform",
    "mottled": "mottled", "banded": "banded",
}

ENVIRONMENT_KEYWORDS = {
    "underwater": "aquatic", "ocean": "marine", "water": "aquatic",
    "forest": "terrestrial-forest", "desert": "terrestrial-desert",
    "grassland": "terrestrial-grassland", "reef": "marine-reef",
    "coral": "marine-coral", "seabed": "marine-seabed",
    "sandy": "marine-sandy", "rocky": "marine-rocky",
}

SIMILARITY_MAPPING = {
    "high": 0.8, "medium": 0.5, "low": 0.2,
    "very high": 0.9, "very low": 0.1,
}

def normalize_color(color_text: str) -> str:
    color_lower = color_text.lower().strip()
    return COLOR_MAPPING.get(color_lower, color_lower)

def normalize_texture(texture_text: str) -> str:
    texture_lower = texture_text.lower().strip()
    return TEXTURE_MAPPING.get(texture_lower, texture_lower)

def normalize_pattern(pattern_text: str) -> str:
    pattern_lower = pattern_text.lower().strip()
    return PATTERN_MAPPING.get(pattern_lower, pattern_lower)

def extract_colors_from_text(text: str) -> List[str]:
    colors = set()
    text_lower = text.lower()
    all_colors = set(COLOR_MAPPING.keys()) | set(COLOR_MAPPING.values())
    all_colors.update(["orange", "pink", "white", "black", "brown", "green", 
                       "blue", "yellow", "red", "gray", "grey", "beige", "purple"])
    for color in all_colors:
        if color in text_lower:
            colors.add(normalize_color(color))
    return list(colors) if colors else ["unknown"]

def extract_textures_from_text(text: str) -> List[str]:
    textures = set()
    text_lower = text.lower()
    all_textures = set(TEXTURE_MAPPING.keys()) | set(TEXTURE_MAPPING.values())
    for texture in all_textures:
        if texture in text_lower:
            textures.add(normalize_texture(texture))
    return list(textures) if textures else ["smooth"]

def determine_environment_type(background_desc: str) -> str:
    desc_lower = background_desc.lower()
    for keyword, env_type in ENVIRONMENT_KEYWORDS.items():
        if keyword in desc_lower:
            return env_type
    return "unknown"

def text_similarity_to_numeric(text: str) -> float:
    text_lower = text.lower().strip()
    return SIMILARITY_MAPPING.get(text_lower, 0.5)

def extract_structured(json_obj: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """Extract structured data - V2 structure"""
    
    organism_name = json_obj.get("object_name", "Unknown")
    category = json_obj.get("object_category", "Unknown")
    background_desc = json_obj.get("background_description", "")
    explanation = json_obj.get("explanation", "")
    
    environment_type = determine_environment_type(background_desc)
    
    organism_colors = extract_colors_from_text(explanation)
    background_colors = extract_colors_from_text(background_desc)
    
    organism_textures = extract_textures_from_text(explanation)
    background_textures = extract_textures_from_text(background_desc)
    
    pattern_raw = json_obj.get("camouflage_type", "None")
    pattern = normalize_pattern(pattern_raw) if pattern_raw.lower() != "none" else "uniform"
    
    color_similarity = text_similarity_to_numeric(json_obj.get("color_similarity", "medium"))
    texture_similarity = text_similarity_to_numeric(json_obj.get("texture_similarity", "medium"))
    contrast_difference = text_similarity_to_numeric(json_obj.get("contrast_difference", "medium"))
    
    camouflage_score = float(json_obj.get("camouflage_score", 0.0))
    confidence = float(json_obj.get("confidence", 0.0))
    
    camo_presence = json_obj.get("camouflage_presence", "Unknown")
    is_camouflaged = camo_presence.lower() == "camouflage"
    
    lighting_condition = "bright"
    if "dark" in background_desc.lower() or "dim" in background_desc.lower():
        lighting_condition = "dim"
    elif "shadow" in background_desc.lower():
        lighting_condition = "shadowed"
    
    return {
        "organism_name": organism_name,
        "category": category,
        "environment_type": environment_type,
        "environment_description": background_desc,
        "organism_colors": organism_colors,
        "background_colors": background_colors,
        "pattern": pattern,
        "organism_textures": organism_textures,
        "background_textures": background_textures,
        "lighting_condition": lighting_condition,
        "color_similarity": color_similarity,
        "texture_similarity": texture_similarity,
        "contrast_difference": contrast_difference,
        "camouflage_score": camouflage_score,
        "confidence": confidence,
        "is_camouflaged": is_camouflaged,
        "camouflage_type": pattern,
        "source_file": source_file,
        "explanation": explanation,
    }

# ============================================================
#           NEO4J STRUCTURE
# ============================================================

class CamouflageKnowledgeGraphV2:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
    def close(self):
        self.driver.close()
    
    def check_connection(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS ok")
                return result.single()["ok"] == 1
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def setup_schema(self):
        """Create constraints for V2 structure"""
        constraints = [
            # Organism uniqueness by name only (can appear in multiple environments)
            "CREATE CONSTRAINT organism_name IF NOT EXISTS FOR (o:Organism) REQUIRE o.name IS UNIQUE",
            
            # Shared Color nodes
            "CREATE CONSTRAINT color_name IF NOT EXISTS FOR (c:Color) REQUIRE c.name IS UNIQUE",
            
            # Shared Texture nodes
            "CREATE CONSTRAINT texture_name IF NOT EXISTS FOR (t:Texture) REQUIRE t.name IS UNIQUE",
            
            # Pattern nodes
            "CREATE CONSTRAINT pattern_type IF NOT EXISTS FOR (p:Pattern) REQUIRE p.type IS UNIQUE",
            
            # Environment uniqueness by type + description + source_file
            "CREATE CONSTRAINT environment_unique IF NOT EXISTS FOR (e:Environment) REQUIRE (e.type, e.description, e.source_file) IS UNIQUE",
            
            # Observation context uniqueness
            "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (oc:ObservationContext) REQUIRE oc.id IS UNIQUE",
            
            # Camouflage assessment uniqueness
            "CREATE CONSTRAINT assessment_id IF NOT EXISTS FOR (ca:CamouflageAssessment) REQUIRE ca.id IS UNIQUE",
            
            # Similarity metric uniqueness
            "CREATE CONSTRAINT similarity_id IF NOT EXISTS FOR (sm:SimilarityMetric) REQUIRE sm.id IS UNIQUE",
            
            # Lighting condition
            "CREATE CONSTRAINT lighting_condition IF NOT EXISTS FOR (lc:LightingCondition) REQUIRE lc.condition IS UNIQUE",
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint warning: {e}")
        
        logger.info("✅ Schema V2 setup complete")
    
    def ingest_batch(self, batch_data: List[Dict[str, Any]]):
        """Ingest batch with V2 structure"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._create_batch_graph_v2, batch_data)
    
    @staticmethod
    def _create_batch_graph_v2(tx, batch_data: List[Dict[str, Any]]):
        """Create V2 graph structure"""
        for data in batch_data:
            
            # 1. Create Organism node (shared across environments)
            tx.run("""
                MERGE (o:Organism {name: $name})
                SET o.category = $category
                """,
                name=data["organism_name"],
                category=data["category"]
            )
            
            # 2. Create organism colors (shared nodes)
            for color in data["organism_colors"]:
                tx.run("""
                    MERGE (c:Color {name: $color_name})
                    WITH c
                    MATCH (o:Organism {name: $organism_name})
                    MERGE (o)-[:HAS_COLOR]->(c)
                    """,
                    color_name=color,
                    organism_name=data["organism_name"]
                )
            
            # 3. Create organism textures (shared nodes)
            for texture in data["organism_textures"]:
                tx.run("""
                    MERGE (t:Texture {name: $texture_name})
                    WITH t
                    MATCH (o:Organism {name: $organism_name})
                    MERGE (o)-[:HAS_TEXTURE]->(t)
                    """,
                    texture_name=texture,
                    organism_name=data["organism_name"]
                )
            
            # 4. Create organism pattern
            tx.run("""
                MERGE (p:Pattern {type: $pattern_type})
                WITH p
                MATCH (o:Organism {name: $organism_name})
                MERGE (o)-[:HAS_PATTERN]->(p)
                """,
                pattern_type=data["pattern"],
                organism_name=data["organism_name"]
            )
            
            # 5. Create Environment node (unique per observation)
            env_desc_short = data["environment_description"][:200]
            tx.run("""
                MERGE (e:Environment {type: $env_type, description: $description, source_file: $source_file})
                SET e.lighting_condition = $lighting
                """,
                env_type=data["environment_type"],
                description=env_desc_short,
                source_file=data["source_file"],
                lighting=data["lighting_condition"]
            )
            
            # 6. Create environment colors (shared nodes)
            for color in data["background_colors"]:
                tx.run("""
                    MERGE (c:Color {name: $color_name})
                    WITH c
                    MATCH (e:Environment {type: $env_type, description: $description, source_file: $source_file})
                    MERGE (e)-[:HAS_COLOR]->(c)
                    """,
                    color_name=color,
                    env_type=data["environment_type"],
                    description=env_desc_short,
                    source_file=data["source_file"]
                )
            
            # 7. Create environment textures (shared nodes)
            for texture in data["background_textures"]:
                tx.run("""
                    MERGE (t:Texture {name: $texture_name})
                    WITH t
                    MATCH (e:Environment {type: $env_type, description: $description, source_file: $source_file})
                    MERGE (e)-[:HAS_TEXTURE]->(t)
                    """,
                    texture_name=texture,
                    env_type=data["environment_type"],
                    description=env_desc_short,
                    source_file=data["source_file"]
                )
            
            # 8. Create LightingCondition
            tx.run("""
                MERGE (lc:LightingCondition {condition: $condition})
                WITH lc
                MATCH (e:Environment {type: $env_type, description: $description, source_file: $source_file})
                MERGE (e)-[:HAS_LIGHTING_CONDITION]->(lc)
                """,
                condition=data["lighting_condition"],
                env_type=data["environment_type"],
                description=env_desc_short,
                source_file=data["source_file"]
            )
            
            # 9. Create CamouflageAssessment (connected to Environment)
            assessment_id = f"assess_{data['source_file']}"
            tx.run("""
                MERGE (ca:CamouflageAssessment {id: $assessment_id})
                SET ca.camouflage_score = $score,
                    ca.confidence = $confidence,
                    ca.is_camouflaged = $is_camouflaged,
                    ca.camouflage_type = $camo_type
                WITH ca
                MATCH (e:Environment {type: $env_type, description: $description, source_file: $source_file})
                MERGE (e)-[:HAS_CAMOUFLAGE_ASSESSMENT]->(ca)
                """,
                assessment_id=assessment_id,
                score=data["camouflage_score"],
                confidence=data["confidence"],
                is_camouflaged=data["is_camouflaged"],
                camo_type=data["camouflage_type"],
                env_type=data["environment_type"],
                description=env_desc_short,
                source_file=data["source_file"]
            )
            
            # 10. Create SimilarityMetric
            metric_id = f"sim_{data['source_file']}"
            tx.run("""
                MERGE (sm:SimilarityMetric {id: $metric_id})
                SET sm.color_similarity = $color_sim,
                    sm.texture_similarity = $texture_sim,
                    sm.contrast_difference = $contrast_diff
                WITH sm
                MATCH (ca:CamouflageAssessment {id: $assessment_id})
                MERGE (ca)-[:HAS_SIMILARITY]->(sm)
                """,
                metric_id=metric_id,
                color_sim=data["color_similarity"],
                texture_sim=data["texture_similarity"],
                contrast_diff=data["contrast_difference"],
                assessment_id=assessment_id
            )
            
            # 11. Create ObservationContext linking Organism to Environment
            observation_id = f"obs_{data['organism_name']}_{data['source_file']}"
            tx.run("""
                MERGE (oc:ObservationContext {id: $observation_id})
                SET oc.source_file = $source_file
                WITH oc
                MATCH (o:Organism {name: $organism_name})
                MATCH (e:Environment {type: $env_type, description: $description, source_file: $source_file})
                MERGE (oc)-[:HAS_ORGANISM]->(o)
                MERGE (oc)-[:OBSERVED_IN]->(e)
                """,
                observation_id=observation_id,
                source_file=data["source_file"],
                organism_name=data["organism_name"],
                env_type=data["environment_type"],
                description=env_desc_short
            )

# ============================================================
#           PROCESSING
# ============================================================

def get_processed_files() -> Set[str]:
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r") as f:
        return set(f.read().splitlines())

def mark_as_processed(filename: str):
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")

def main():
    logger.info("🚀 Starting KG V2 Ingestion")
    
    kg = CamouflageKnowledgeGraphV2(NEO4J_URI, NEO4J_USER, NEO4J_PASS, TARGET_DB)
    
    if not kg.check_connection():
        logger.error("❌ Neo4j connection failed")
        return
    
    logger.info("✅ Connected to Neo4j")
    kg.setup_schema()
    
    processed_files = get_processed_files()
    all_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json")]
    files_to_process = [f for f in all_files if f not in processed_files]
    
    logger.info(f"📊 To process: {len(files_to_process)} files")
    
    if not files_to_process:
        logger.info("✅ All files processed!")
        kg.close()
        return
    
    batch = []
    success_count = 0
    failed_count = 0
    
    for filename in tqdm(files_to_process, desc="Processing"):
        filepath = os.path.join(ANNOTATION_DIR, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_obj = json.load(f)
            
            structured_data = extract_structured(json_obj, filename)
            batch.append(structured_data)
            
            if len(batch) >= BATCH_SIZE:
                kg.ingest_batch(batch)
                for item in batch:
                    mark_as_processed(item["source_file"])
                success_count += len(batch)
                batch = []
        
        except Exception as e:
            logger.error(f"❌ Error: {filename}: {e}")
            failed_count += 1
    
    if batch:
        kg.ingest_batch(batch)
        for item in batch:
            mark_as_processed(item["source_file"])
        success_count += len(batch)
    
    kg.close()
    
    logger.info(f"✅ Complete! Success: {success_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()