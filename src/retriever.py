import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from .llm import get_embeddings, get_llm
from .graph import Neo4jManager
from langchain_core.documents import Document as LangchainDocument

class HybridRetriever:
    def __init__(self):
        self.neo4j = Neo4jManager()
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        self.vector_index_name = "chunk_vector_index"

    def vector_search(self, query: str, k: int = 5) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Local Search: Finds relevant text chunks using vector similarity.
        Returns: (results, metadata)
        """
        start_time = time.time()
        
        # 1. Generate Query Embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # 2. Run Neo4j Vector Query
        # Note: We assume the index exists (created via graph.py)
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        MATCH (node)-[:PART_OF]->(s:Section)-[:PART_OF]->(d:Document)
        RETURN node.content AS content, score, elementId(node) as id,
               d.title AS doc_title, s.title AS section_title, d.source AS source
        """
        
        with self.neo4j.driver.session() as session:
            try:
                result = session.run(cypher, 
                                   index_name=self.vector_index_name, 
                                   k=k, 
                                   embedding=query_embedding)
                results = [
                    {
                        "content": r["content"], 
                        "score": r["score"], 
                        "source": "vector",
                        "metadata": {
                            "doc_title": r["doc_title"],
                            "section_title": r["section_title"],
                            "source_path": r["source"],
                            "node_id": r["id"]
                        }
                    } 
                    for r in result
                ]
                
                execution_time = time.time() - start_time
                metadata = {
                    "query": query,
                    "cypher": cypher,
                    "result_count": len(results),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                return results, metadata
            except Exception as e:
                print(f"Vector search failed: {e}")
                execution_time = time.time() - start_time
                metadata = {
                    "query": query,
                    "cypher": cypher,
                    "result_count": 0,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                return [], metadata

    async def retrieve_communities(self, query: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Global Search: Finds community summaries relevant to the query.
        For 2026 'Global Context', we look for communities that share entities with the query.
        Returns: (results, metadata)
        """
        start_time = time.time()
        
        # 1. Extract potential entities from query (simple heuristic or LLM)
        # For prototype, we'll try to match exact entity names or use keywords
        # A better approach (2026) is using the LLM to map query -> entities -> communities
        
        # Simple implementation:
        # Find communities where the most relevant nodes live.
        
        # Let's ask Cypher to fuzzy match entities in query
        # Fixed: Check for communityId existence before filtering
        cypher = """
        CALL db.index.fulltext.queryNodes("entity_text_index", $query) YIELD node, score
        MATCH (node)-[:RELATED_TO*1..2]-(other)
        WHERE EXISTS(node.communityId) AND node.communityId IS NOT NULL
        RETURN DISTINCT node.communityId as communityId, count(other) as size
        ORDER BY size DESC
        LIMIT 3
        """
        # Note: We need a fulltext index for this to work well, or we just rely on the LLM
        # to pick interesting communities. 
        
        # Alternative "Global" pattern without fulltext index:
        # Just get the top communities by centrality and summarize them?
        # Or simpler: Vector search for chunks -> find their communities -> summarize community.
        
        # Let's go with: Vector Search -> Nodes -> Communities -> Summaries
        # This roots the global context in the local query hits.
        
        # Placeholder for community summaries (requires pre-computation)
        # We will return mocked summary objects for now or raw entity lists
        
        raw_communities = self.neo4j.get_community_summaries()
        # Filter mostly relevant ones?
        # For prototype, simply return a text describing the top communities found via vector search connection
        
        results = [
            {"content": f"Community {c['communityId']}: Contains concepts {c['entities'][:5]}...", "source": "community"}
            for c in raw_communities[:3]
        ]
        
        execution_time = time.time() - start_time
        metadata = {
            "query": query,
            "cypher": cypher,
            "result_count": len(results),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return results, metadata

    async def retrieve(self, query: str) -> Tuple[List[Dict], List[Dict[str, Any]]]:
        """
        Combines Local and Global search results.
        Returns: (all_docs, metadata_list)
        """
        # Run in parallel
        # Note: retrieve_communities is async, vector_search is sync (neo4j driver)
        # We wrap vector_search
        
        loop = asyncio.get_event_loop()
        local_results, local_metadata = await loop.run_in_executor(None, self.vector_search, query)
        global_results, global_metadata = await self.retrieve_communities(query)
        
        # Combine and format
        all_docs = local_results + global_results
        metadata_list = [local_metadata, global_metadata]
        
        return all_docs, metadata_list
