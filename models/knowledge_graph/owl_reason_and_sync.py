"""
OWL Reasoning and Synchronization Script
Uses owlready2 to perform reasoning and sync inferred triples to Neo4j.

Requirements:
    pip install owlready2 neo4j
"""

import os
from owlready2 import *
from neo4j import GraphDatabase
import logging

# Configuration
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"
TARGET_DB = "neo4j"

OWL_FILE = "./camouflage_ontology.owl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
#           SWRL RULES & REASONING THRESHOLDS
# ============================================================

"""
SWRL RULES (Conceptual - owlready2 uses Python rules):

Rule 1: Infer camouflagedIn when similarities are high
    Organism(?o) ∧ Environment(?e) ∧ observedIn(?o, ?e) ∧ 
    SimilarityMetric(?sm) ∧ hasSimilarity(?o, ?sm) ∧
    colorSimilarityValue(?sm, ?cs) ∧ textureSimilarityValue(?sm, ?ts) ∧
    swrlb:greaterThan(?cs, 0.7) ∧ swrlb:greaterThan(?ts, 0.5)
    → camouflagedIn(?o, ?e)

Rule 2: Infer high camouflage score
    Organism(?o) ∧ SimilarityMetric(?sm) ∧ hasSimilarity(?o, ?sm) ∧
    colorSimilarityValue(?sm, ?cs) ∧ textureSimilarityValue(?sm, ?ts) ∧
    camouflageScore(?o, ?score) ∧ swrlb:equal(?score, (?cs + ?ts) / 2)

Thresholds:
    high: ≥ 0.75
    medium: 0.4-0.75
    low: < 0.4
"""

CAMOUFLAGE_COLOR_THRESHOLD = 0.7
CAMOUFLAGE_TEXTURE_THRESHOLD = 0.5

# ============================================================
#           OWL REASONING WITH OWLREADY2
# ============================================================

def load_ontology(owl_path: str):
    """Load OWL ontology file."""
    if not os.path.exists(owl_path):
        logger.error(f"❌ OWL file not found: {owl_path}")
        return None
    
    logger.info(f"📖 Loading ontology from {owl_path}")
    onto = get_ontology(f"file://{os.path.abspath(owl_path)}").load()
    logger.info(f"✅ Loaded ontology: {onto.base_iri}")
    return onto

def populate_ontology_from_neo4j(onto, driver, database):
    """
    Load sample data from Neo4j into OWL ontology for reasoning.
    This is a simplified example - in production you'd load all relevant data.
    """
    logger.info("📥 Loading sample data from Neo4j into ontology...")
    
    with driver.session(database=database) as session:
        # Load organisms and their properties
        result = session.run("""
            MATCH (o:Organism)-[:OBSERVED_AS]->(ce:CamouflageEvent)-[:HAS_SIMILARITY]->(sm:SimilarityMetric)
            MATCH (o)-[:OBSERVED_IN]->(e:Environment)
            RETURN o.name AS organism_name, 
                   o.category AS category,
                   e.type AS env_type,
                   sm.colorSimilarityValue AS color_sim,
                   sm.textureSimilarityValue AS texture_sim,
                   ce.camouflage_score AS score
            LIMIT 100
        """)
        
        # Access ontology classes
        Organism = onto.Organism
        Environment = onto.Environment
        SimilarityMetric = onto.SimilarityMetric
        
        individuals_created = 0
        seen_organisms = set()
        
        for record in result:
            try:
                # Create unique organism identifier (sanitize name)
                org_name_raw = record["organism_name"]
                # Remove spaces, special chars, make unique
                org_name = f"Org_{org_name_raw.replace(' ', '_').replace('-', '_')}_{individuals_created}"
                
                # Skip if already created (shouldn't happen with unique suffix, but safety check)
                if org_name in seen_organisms:
                    continue
                
                seen_organisms.add(org_name)
                
                # Create organism individual with unique name
                org = Organism(org_name)
                
                # Set data properties (use lists for owlready2)
                if record["organism_name"]:
                    # Store original name in label or comment, not as individual name
                    org.label = [record["organism_name"]]
                
                # Create environment individual with unique name
                env_name = f"Env_{record['env_type']}_{individuals_created}"
                env = Environment(env_name)
                
                # Create similarity metric with unique name
                sim_name = f"Sim_{individuals_created}"
                sim = SimilarityMetric(sim_name)
                
                # Set similarity values (must be lists in owlready2)
                if record["color_sim"] is not None:
                    sim.colorSimilarityValue = [float(record["color_sim"])]
                if record["texture_sim"] is not None:
                    sim.textureSimilarityValue = [float(record["texture_sim"])]
                
                # Set relationships (must be lists)
                org.observedIn = [env]
                
                # Note: hasSimilarity relationship needs to be defined in ontology
                # For now, we'll skip it or you can add it to the OWL file
                
                individuals_created += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Error creating individual {individuals_created}: {e}")
                continue
        
        logger.info(f"✅ Created {individuals_created} individuals in ontology")
def apply_reasoning_rules(onto):
    """
    Apply custom reasoning rules using owlready2's rule system.
    These simulate SWRL rules.
    """
    logger.info("🧠 Applying reasoning rules...")
    
    with onto:
        # Rule: Infer camouflagedIn when color and texture similarity are high
        class camouflagedIn_inferred_rule(Thing >> Thing):
            """Infers camouflaged relationship based on similarity thresholds."""
            pass
        
        # Get all organisms
        for org in onto.Organism.instances():
            if hasattr(org, 'hasSimilarity') and org.hasSimilarity:
                for sim_metric in org.hasSimilarity:
                    color_sim = sim_metric.colorSimilarityValue[0] if sim_metric.colorSimilarityValue else 0
                    texture_sim = sim_metric.textureSimilarityValue[0] if sim_metric.textureSimilarityValue else 0
                    
                    # Apply threshold rule
                    if color_sim >= CAMOUFLAGE_COLOR_THRESHOLD and texture_sim >= CAMOUFLAGE_TEXTURE_THRESHOLD:
                        # Infer camouflagedIn relationship
                        if hasattr(org, 'observedIn') and org.observedIn:
                            for env in org.observedIn:
                                if not hasattr(org, 'camouflagedIn'):
                                    org.camouflagedIn = []
                                if env not in org.camouflagedIn:
                                    org.camouflagedIn.append(env)
                                    logger.info(f"✨ Inferred: {org.name[0]} camouflagedIn {env}")
    
    logger.info("✅ Reasoning rules applied")

def run_pellet_reasoner(onto):
    """
    Run Pellet reasoner for full OWL DL reasoning.
    Note: Requires Java and Pellet to be installed.
    """
    try:
        logger.info("🔬 Running Pellet reasoner...")
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
        logger.info("✅ Pellet reasoning complete")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Pellet reasoner failed (may not be installed): {e}")
        logger.info("💡 Continuing with custom rule-based reasoning...")
        return False

def extract_inferred_triples(onto):
    """Extract inferred relationships from ontology."""
    logger.info("📤 Extracting inferred triples...")
    
    inferred_triples = []
    
    # Extract camouflagedIn relationships
    for org in onto.Organism.instances():
        if hasattr(org, 'camouflagedIn') and org.camouflagedIn:
            for env in org.camouflagedIn:
                org_name = org.name[0] if org.name else str(org)
                triple = {
                    "subject": org_name,
                    "predicate": "CAMOUFLAGED_IN",
                    "object": str(env),
                    "inferred": True
                }
                inferred_triples.append(triple)
                logger.info(f"📋 Extracted triple: {org_name} -[CAMOUFLAGED_IN]-> {env}")
    
    logger.info(f"✅ Extracted {len(inferred_triples)} inferred triples")
    return inferred_triples

def sync_inferred_to_neo4j(inferred_triples, driver, database):
    """Push inferred triples back into Neo4j."""
    logger.info("🔄 Syncing inferred triples to Neo4j...")
    
    with driver.session(database=database) as session:
        for triple in inferred_triples:
            try:
                # Create inferred CAMOUFLAGED_IN relationship
                session.run("""
                    MATCH (o:Organism)
                    WHERE o.name = $org_name
                    MATCH (e:Environment)
                    MERGE (o)-[r:CAMOUFLAGED_IN_INFERRED]->(e)
                    SET r.inferred = true,
                        r.source = 'OWL_Reasoning'
                    """,
                    org_name=triple["subject"]
                )
                logger.info(f"✅ Synced: {triple['subject']} -[CAMOUFLAGED_IN_INFERRED]-> Environment")
            except Exception as e:
                logger.error(f"❌ Error syncing triple: {e}")
    
    logger.info("✅ Inference sync complete")

# ============================================================
#           NEOSEMANTICS (n10s) IMPORT EXAMPLE
# ============================================================

def import_owl_with_n10s(driver, database, owl_path):
    """
    Import OWL ontology into Neo4j using Neosemantics (n10s) plugin.
    
    Prerequisites:
        - Install n10s plugin in Neo4j (via Neo4j Desktop or manually)
        - Restart Neo4j after installation
    
    Commands to run in Neo4j Browser:
        CALL n10s.graphconfig.init();
        CALL n10s.onto.import.fetch("file:///path/to/camouflage_ontology.owl", "RDF/XML");
    """
    logger.info("📥 Importing OWL with Neosemantics (n10s)...")
    
    with driver.session(database=database) as session:
        try:
            # Initialize n10s
            session.run("CALL n10s.graphconfig.init()")
            logger.info("✅ n10s initialized")
            
            # Import ontology
            abs_path = f"file:///{os.path.abspath(owl_path)}"
            session.run(
                "CALL n10s.onto.import.fetch($owl_path, 'RDF/XML')",
                owl_path=abs_path
            )
            logger.info(f"✅ Imported OWL ontology from {abs_path}")
            
            # Query imported classes
            result = session.run("MATCH (c:Class) RETURN c.name LIMIT 10")
            logger.info("📋 Imported OWL classes:")
            for record in result:
                logger.info(f"  - {record['c.name']}")
        
        except Exception as e:
            logger.error(f"❌ n10s import failed: {e}")
            logger.info("💡 Make sure n10s plugin is installed in Neo4j")
            logger.info("   Install via: Neo4j Desktop -> Plugins -> Neosemantics")

# ============================================================
#           MAIN REASONING PIPELINE
# ============================================================

def main():
    """Main OWL reasoning and sync pipeline."""
    logger.info("🚀 Starting OWL Reasoning Pipeline")
    
    # Load ontology
    onto = load_ontology(OWL_FILE)
    if not onto:
        return
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    try:
        # Method A: Reason in OWL then sync to Neo4j
        logger.info("\n" + "="*60)
        logger.info("METHOD A: OWL Reasoning → Neo4j Sync")
        logger.info("="*60)
        
        # Populate ontology with Neo4j data
        populate_ontology_from_neo4j(onto, driver, TARGET_DB)
        
        # Try Pellet reasoner first, fallback to custom rules
        reasoner_success = run_pellet_reasoner(onto)
        
        if not reasoner_success:
            # Apply custom reasoning rules
            apply_reasoning_rules(onto)
        
        # Extract inferred triples
        inferred_triples = extract_inferred_triples(onto)
        
        # Sync back to Neo4j
        if inferred_triples:
            sync_inferred_to_neo4j(inferred_triples, driver, TARGET_DB)
        
        # Method B: Import OWL structure with n10s (optional)
        logger.info("\n" + "="*60)
        logger.info("METHOD B: Import OWL with Neosemantics (n10s)")
        logger.info("="*60)
        logger.info("⚠️  Requires n10s plugin installed in Neo4j")
        logger.info("💡 Uncomment the line below to try n10s import:")
        logger.info("   import_owl_with_n10s(driver, TARGET_DB, OWL_FILE)")
        
        # Uncomment to try n10s:
        # import_owl_with_n10s(driver, TARGET_DB, OWL_FILE)
        
    finally:
        driver.close()
    
    logger.info("✅ OWL Reasoning Pipeline Complete!")

if __name__ == "__main__":
    main()