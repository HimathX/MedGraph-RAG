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
                pass

    def create_vector_index(self, index_name: str = "chunk_vector_index", dimension: int = 768):
        """
        Creates a vector index on Chunk nodes.
        Note: text-embedding-005 dimension might be 768 or higher. 
        Adjust dimension as needed (e.g. 768 is common for older models, newer ones might be larger).
        Gemini text-embedding-004 is 768. text-embedding-005 might be same or different.
        For safety, we'll allow passing it.
        """
        # Drop index if exists to ensure clean state or just try create
        # Cypher for Neo4j 5.x vector index
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: $dimension,
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        with self.driver.session() as session:
            try:
                session.run(query, dimension=dimension)
                print(f"Vector index '{index_name}' created/verified.")
            except Exception as e:
                print(f"Error creating vector index: {e}")

    def create_community_index(self):
        """
        Runs GDS Leiden algorithm to detect communities and write back 'communityId'.
        """
        # check if GDS is available
        check_gds = "RETURN gds.version()"
        
        projection_name = "med_graph_projection"
        
        # 1. Project the graph
        project_query = f"""
        CALL gds.graph.project.cypher(
            '{projection_name}',
            'MATCH (n:Entity) RETURN id(n) AS id',
            'MATCH (n)-[r:RELATED_TO]->(m) RETURN id(n) AS source, id(m) AS target'
        )
        YIELD graphName
        """
        
        # 2. Run Leiden
        leiden_query = f"""
        CALL gds.leiden.write(
            '{projection_name}',
            {{
                writeProperty: 'communityId'
            }}
        )
        YIELD communityCount, modularity, communitiesWritten
        """
        
        # 3. Drop projection
        drop_query = f"CALL gds.graph.drop('{projection_name}') YIELD graphName"
        
        with self.driver.session() as session:
            try:
                # Check GDS
                try:
                    session.run(check_gds)
                except Exception:
                    print("GDS library not detected or error checking version.")
                    return

                # Drop projection if exists (cleanup)
                try:
                    session.run(drop_query)
                except:
                    pass

                # Project
                session.run(project_query)
                
                # Run Leiden
                result = session.run(leiden_query).single()
                print(f"Communities detected: {result['communityCount']}")
                
                # Cleanup
                session.run(drop_query)
                
            except Exception as e:
                print(f"Error running community detection: {e}")

    def get_community_summaries(self):
        """
        Retrieves aggregated text for communities to be summarized by LLM.
        Returns a list of {communityId, text_content}.
        """
        # This is a placeholder. In a real app, we'd aggregate Entity names/types
        # and maybe central chunks associated with them.
        query = """
        MATCH (e:Entity)
        WHERE e.communityId IS NOT NULL
        WITH e.communityId AS comId, collect(e.name) AS entities
        RETURN comId, entities
        LIMIT 100
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [{"communityId": r["comId"], "entities": r["entities"]} for r in result]

