from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")   # aura: neo4j+s://<host>:7687
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ---------- Driver ----------
def get_driver(uri=None, user=None, password=None):
    uri = uri or NEO4J_URI
    user = user or NEO4J_USER
    password = password or NEO4J_PASSWORD
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def test_connection(uri=None, user=None, password=None) -> bool:
    try:
        driver = get_driver(uri, user, password)
        driver.verify_connectivity()
        with driver.session() as s:
            return s.run("RETURN 1 AS ok").single().get("ok") == 1
    except Exception:
        return False
    finally:
        try:
            driver.close()
        except Exception:
            pass

# ---------- Write helpers ----------
def upsert_disease(name: str, condition: str = None):
    """Create or update a Disease node by name."""
    query = """
    MERGE (d:Disease {name: $name})
    ON CREATE SET d.condition = $condition
    ON MATCH SET  d.condition = coalesce($condition, d.condition)
    RETURN d.name AS name
    """
    with get_driver() as driver, driver.session() as s:
        return s.run(query, name=name, condition=condition).single()["name"]

def upsert_organ(name: str):
    with get_driver() as driver, driver.session() as s:
        s.run("MERGE (:Organ {name:$name})", name=name)

def upsert_effect(name: str):
    with get_driver() as driver, driver.session() as s:
        s.run("MERGE (:Effect {name:$name})", name=name)

def relate_affects(disease: str, organ: str):
    """(Disease)-[:AFFECTS]->(Organ)"""
    query = """
    MATCH (d:Disease {name:$disease})
    MERGE (o:Organ {name:$organ})
    MERGE (d)-[:AFFECTS]->(o)
    """
    with get_driver() as driver, driver.session() as s:
        s.run(query, disease=disease, organ=organ)

def relate_results_in(source_disease: str, target_disease: str):
    """(Disease)-[:RESULTS_IN]->(Disease)"""
    query = """
    MERGE (src:Disease {name:$src})
    MERGE (dst:Disease {name:$dst})
    MERGE (src)-[:RESULTS_IN]->(dst)
    """
    with get_driver() as driver, driver.session() as s:
        s.run(query, src=source_disease, dst=target_disease)

def relate_causes(cause_disease: str, effect_disease: str):
    """(Disease)-[:CAUSES]->(Disease)"""
    query = """
    MERGE (c:Disease {name:$cause})
    MERGE (e:Disease {name:$effect})
    MERGE (c)-[:CAUSES]->(e)
    """
    with get_driver() as driver, driver.session() as s:
        s.run(query, cause=cause_disease, effect=effect_disease)

def relate_can_have(disease: str, effect: str):
    """(Disease)-[:CAN_HAVE]->(Effect)"""
    query = """
    MATCH (d:Disease {name:$disease})
    MERGE (e:Effect {name:$effect})
    MERGE (d)-[:CAN_HAVE]->(e)
    """
    with get_driver() as driver, driver.session() as s:
        s.run(query, disease=disease, effect=effect)

# ---------- Read helper ----------
def query_related_organs(disease_name: str):
    """
    Return a list of organs affected by a disease (case-insensitive match on name).
    """
    query = """
    MATCH (d:Disease)-[:AFFECTS]->(o:Organ)
    WHERE toLower(d.name) = toLower($name)
    RETURN o.name AS organ
    ORDER BY organ
    """
    try:
        with get_driver() as driver, driver.session() as s:
            return [r["organ"] for r in s.run(query, name=disease_name)]
    except (ServiceUnavailable, AuthError, Exception):
        return []

# ---------- One-time setup/example for Cardiomegaly ----------
if __name__ == "__main__":
    assert test_connection(), "Neo4j connection failed. Check URI/user/password."

    # 1) Upsert Cardiomegaly and some commonly linked nodes
    upsert_disease("Cardiomegaly", "Enlarged heart due to chronic pressure/volume overload")
    upsert_disease("Heart Failure")  # already exists in your graph, but MERGE is safe
    upsert_organ("Heart")
    upsert_organ("Lungs")

    # 2) Relationships (adjust to your domain rules)
    relate_affects("Cardiomegaly", "Heart")
    relate_affects("Cardiomegaly", "Lungs")
    relate_results_in("Cardiomegaly", "Heart Failure")      # Cardiomegaly -> Heart Failure
    relate_causes("Coronary Artery Disease", "Cardiomegaly") # if CAD exists; MERGE will create if missing
    relate_can_have("Cardiomegaly", "Shortness of Breath")

    # 3) Query from the frontend input (e.g., "cardiomegaly")
    print(query_related_organs("cardiomegaly"))  # -> ['Heart', 'Lungs'] once relationships exist
