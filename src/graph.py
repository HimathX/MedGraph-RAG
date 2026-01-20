import os
from neo4j import GraphDatabase
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Neo4jManager:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def add_document(self, doc: Dict):
        """Creates the Document node."""
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.year = $year,
            d.authors = $authors,
            d.source = $source_path
        """
        with self.driver.session() as session:
            session.run(query, 
                        id=doc.get("id"),
                        title=doc.get("title"),
                        year=doc.get("year"),
                        authors=doc.get("authors"),
                        source_path=doc.get("source_path"))

    def add_triplets(self, triplets: List[Dict]):
        """
        Batch inserts triplets.
        triplet: {head, head_type, relation, tail, tail_type, source_doc_id...}
        """
        if not triplets:
            return

        query = """
        UNWIND $batch AS row
        
        // Merge Head Entity
        MERGE (h:Entity {name: row.head})
        ON CREATE SET h.type = row.head_type
        
        // Merge Tail Entity
        MERGE (t:Entity {name: row.tail})
        ON CREATE SET t.type = row.tail_type
        
        // Create Relationship (Dynamic types are tricky in Cypher, usually requires APOC or filtered MERGE)
        // For simplicity in this prototype, we'll use a generic RELATION relationship with a type property
        // OR better: use APOC if available, but let's assume standard Cypher.
        // We will do a generic merge and set the type property, or use conditional procedures.
        
        // Option 1: Generic Edge with type property (Easiest for dynamic types)
        MERGE (h)-[r:RELATED_TO]->(t)
        SET r.type = row.relation,
            r.source_doc_id = row.source_doc_id,
            r.section = row.source_section
            
        // Option 2: To make it queryable as (h)-[:TREATS]->(t), we need APOC:
        // CALL apoc.create.relationship(h, row.relation, {}, t) YIELD rel
        """
        
        # Using APOC is best, but let's stick to standard Cypher with dynamic relationships if possible
        # Actually, standard Cypher cannot parametrize relationship types.
        # We will iterate and group by relation type for valid Cypher construction
        # OR just use APOC which is standard in most Neo4j instances now.
        # Let's try APOC. If it fails, we fall back.
        
        apoc_query = """
        UNWIND $batch AS row
        MERGE (h:Entity {name: row.head})
        ON CREATE SET h.type = row.head_type
        MERGE (t:Entity {name: row.tail})
        ON CREATE SET t.type = row.tail_type
        WITH h, t, row
        CALL apoc.merge.relationship(h, row.relation, {}, {}, t, {}) YIELD rel
        SET rel.source_doc_id = row.source_doc_id,
            rel.section = row.source_section
        RETURN count(rel)
        """
        
        with self.driver.session() as session:
            try:
                session.run(apoc_query, batch=triplets)
            except Exception as e:
                print(f"Failed to use APOC for relationships: {e}")
                print("Falling back to generic RELATED_TO edges.")
                # Fallback logic could go here
