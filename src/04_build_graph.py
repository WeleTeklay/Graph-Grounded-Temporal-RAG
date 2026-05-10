#src/04_build_graph.py
"""
Phase 4: Production-Grade Graph Builder
Builds Neo4j graph with SUPERSEDES relationships and temporal constraints.
FIXED: Document-level relationships instead of clause-level cartesian product.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from tqdm import tqdm

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, safe_json_load, safe_json_dump, Timer

logger = get_logger(__name__)


@dataclass
class GraphStats:
    """Statistics from graph building."""
    nodes_created: int
    relationships_created: int
    documents_processed: int
    chunks_processed: int
    build_time_seconds: float
    timestamp: str


class Neo4jGraphBuilder:
    """
    Production-grade Neo4j graph builder with validation and error recovery.
    """
    
    def __init__(self):
        self.uri = config.neo4j.uri
        self.user = config.neo4j.user
        self.password = config.neo4j.password
        self.driver = None
        self._connect()
    
    def _connect(self) -> bool:
        """Establish Neo4j connection with retry."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=10
                )
                self.driver.verify_connectivity()
                logger.info(f"Connected to Neo4j: {self.uri}")
                return True
                
            except AuthError as e:
                logger.error(f"Authentication failed: {e}")
                return False
            except ServiceUnavailable as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}")
                return False
        
        logger.error("Failed to connect to Neo4j after retries")
        return False
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.debug("Neo4j connection closed")
    
    @handle_errors(default_return=False)
    def clear_database(self) -> bool:
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
            return True
    
    @handle_errors(default_return=False)
    def create_indexes(self) -> bool:
        """Create indexes for performance."""
        with self.driver.session() as session:
            # Index on Clause.id
            session.run("CREATE INDEX clause_id IF NOT EXISTS FOR (c:Clause) ON (c.id)")
            # Index on Clause.doc_id
            session.run("CREATE INDEX clause_doc_id IF NOT EXISTS FOR (c:Clause) ON (c.doc_id)")
            # Index on Clause.effective_date
            session.run("CREATE INDEX clause_date IF NOT EXISTS FOR (c:Clause) ON (c.effective_date)")
            # Index on Document.doc_id
            session.run("CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.doc_id)")
            logger.info("Indexes created")
            return True
    
    @handle_errors(default_return=None)
    def create_clause_node(self, chunk: Dict) -> Optional[str]:
        """
        Create a single Clause node.
        """
        chunk_id = chunk.get('chunk_id')
        doc_id = chunk.get('doc_id')
        effective_date = chunk.get('effective_date')
        text = chunk.get('text', '')
        
        text_preview = text[:1000] if text else ''
        
        with self.driver.session() as session:
            result = session.run("""
                MERGE (c:Clause {id: $chunk_id})
                SET c.doc_id = $doc_id,
                    c.effective_date = date($effective_date),
                    c.text = $text,
                    c.char_count = $char_count,
                    c.created_at = datetime()
                RETURN c.id as id
            """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                effective_date=effective_date,
                text=text_preview,
                char_count=len(text)
            )
            record = result.single()
            return record['id'] if record else None
    
    @handle_errors(default_return=0)
    def create_clause_nodes_batch(self, chunks: List[Dict]) -> int:
        """Create Clause nodes in batch."""
        created = 0
        
        with self.driver.session() as session:
            for chunk in tqdm(chunks, desc="Creating nodes", leave=False):
                try:
                    node_id = self.create_clause_node(chunk)
                    if node_id:
                        created += 1
                except Exception as e:
                    logger.error(f"Failed to create node {chunk.get('chunk_id')}: {e}")
        
        return created
    
    @handle_errors(default_return=0)
    def create_document_nodes(self, manifest: pd.DataFrame, chunks: List[Dict]) -> int:
        """
        Create Document nodes (one per document) and link to Clause nodes.
        NEW FUNCTION: Creates document-level nodes for efficient SUPERSEDES.
        """
        created = 0
        
        # Get unique documents with their effective dates
        doc_info = manifest[['doc_id', 'effective_date', 'doc_title']].drop_duplicates()
        
        with self.driver.session() as session:
            for _, row in tqdm(doc_info.iterrows(), desc="Creating document nodes", total=len(doc_info)):
                doc_id = row['doc_id']
                effective_date = row['effective_date']
                doc_title = row.get('doc_title', doc_id)
                
                try:
                    # Create Document node
                    session.run("""
                        MERGE (d:Document {doc_id: $doc_id})
                        SET d.effective_date = date($effective_date),
                            d.title = $title,
                            d.created_at = datetime()
                    """, doc_id=doc_id, effective_date=effective_date, title=doc_title)
                    
                    # Link all Clause nodes to this Document
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (c:Clause {doc_id: $doc_id})
                        MERGE (c)-[:BELONGS_TO]->(d)
                    """, doc_id=doc_id)
                    
                    created += 1
                except Exception as e:
                    logger.error(f"Failed to create document node {doc_id}: {e}")
        
        logger.info(f"Created {created} Document nodes")
        return created
    
    @handle_errors(default_return=0)
    def create_supersedes_relationships(self, manifest: pd.DataFrame) -> int:
        """
        Create SUPERSEDES relationships BETWEEN DOCUMENTS (not clauses).
        FIXED: Only 1 relationship per document pair, not N×M.
        """
        created = 0
        
        # Filter rows with valid supersedes
        valid_rows = manifest[manifest['supersedes_doc_id'].notna()]
        valid_rows = valid_rows[valid_rows['supersedes_doc_id'] != 'None']
        valid_rows = valid_rows[valid_rows['supersedes_doc_id'] != '']
        
        with self.driver.session() as session:
            for _, row in tqdm(valid_rows.iterrows(), 
                              desc="Creating document relationships", 
                              total=len(valid_rows)):
                doc_id = row['doc_id']
                supersedes = row['supersedes_doc_id']
                
                try:
                    # FIXED: Create ONE relationship between Document nodes
                    result = session.run("""
                        MATCH (new_doc:Document {doc_id: $doc_id})
                        MATCH (old_doc:Document {doc_id: $supersedes})
                        WHERE new_doc.effective_date > old_doc.effective_date
                        MERGE (new_doc)-[:SUPERSEDES]->(old_doc)
                        RETURN count(*) as count
                    """,
                        doc_id=doc_id,
                        supersedes=supersedes
                    )
                    record = result.single()
                    if record and record['count'] > 0:
                        created += 1
                        logger.debug(f"SUPERSEDES: {doc_id} -> {supersedes}")
                        
                except Exception as e:
                    logger.error(f"Failed to create SUPERSEDES for {doc_id}: {e}")
        
        return created
    
    def get_graph_stats(self) -> Dict:
        """Get current graph statistics."""
        with self.driver.session() as session:
            clause_count = session.run("MATCH (n:Clause) RETURN count(n) as c").single()['c']
            doc_count = session.run("MATCH (n:Document) RETURN count(n) as c").single()['c']
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
            
            supersedes_count = session.run(
                "MATCH ()-[r:SUPERSEDES]->() RETURN count(r) as c"
            ).single()['c']
            
            belongs_to_count = session.run(
                "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as c"
            ).single()['c']
            
            return {
                'clause_nodes': clause_count,
                'document_nodes': doc_count,
                'total_relationships': rel_count,
                'supersedes_relationships': supersedes_count,
                'belongs_to_relationships': belongs_to_count
            }
    
    def build(self, chunks_path: Path = None, manifest_path: Path = None) -> Optional[GraphStats]:
        """
        Build complete graph from chunks and manifest.
        """
        if not self.driver:
            logger.error("No Neo4j connection")
            return None
        
        start_time = datetime.now()
        
        # Load data
        chunks_path = chunks_path or config.paths.chunks_dir / "clauses.json"
        manifest_path = manifest_path or config.paths.project_root / "document_manifest.csv"
        
        if not chunks_path.exists():
            logger.error(f"Chunks not found: {chunks_path}")
            return None
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return None
        
        chunks = safe_json_load(chunks_path, default=[])
        manifest = pd.read_csv(manifest_path)
        
        logger.info(f"Loaded {len(chunks)} chunks from {len(manifest)} documents")
        
        with Timer("Graph building"):
            # Clear existing data
            self.clear_database()
            
            # Create indexes
            self.create_indexes()
            
            # Create Clause nodes
            nodes_created = self.create_clause_nodes_batch(chunks)
            logger.info(f"Created {nodes_created} Clause nodes")
            
            # Create Document nodes and relationships
            docs_created = self.create_document_nodes(manifest, chunks)
            logger.info(f"Created {docs_created} Document nodes")
            
            # Create SUPERSEDES relationships (document-level)
            supersedes_created = self.create_supersedes_relationships(manifest)
            logger.info(f"Created {supersedes_created} SUPERSEDES relationships")
        
        # Get final stats
        stats = self.get_graph_stats()
        build_time = (datetime.now() - start_time).total_seconds()
        
        # Save build report
        report = {
            'build_timestamp': datetime.now().isoformat(),
            'clause_nodes_created': nodes_created,
            'document_nodes_created': docs_created,
            'supersedes_created': supersedes_created,
            'documents_processed': len(manifest),
            'chunks_processed': len(chunks),
            'build_time_seconds': build_time,
            'final_stats': stats
        }
        
        report_path = config.paths.processed_texts_dir / "graph_build_report.json"
        safe_json_dump(report, report_path)
        
        logger.info(f"Graph build complete in {build_time:.1f}s")
        logger.info(f"Clause nodes: {stats['clause_nodes']}")
        logger.info(f"Document nodes: {stats['document_nodes']}")
        logger.info(f"SUPERSEDES relationships: {stats['supersedes_relationships']}")
        logger.info(f"Report: {report_path}")
        
        return GraphStats(
            nodes_created=nodes_created,
            relationships_created=supersedes_created,
            documents_processed=len(manifest),
            chunks_processed=len(chunks),
            build_time_seconds=build_time,
            timestamp=datetime.now().isoformat()
        )


def main():
    """Main entry point for graph building phase."""
    logger.info("=" * 60)
    logger.info("Phase 4: Production Graph Builder (FIXED)")
    logger.info("=" * 60)
    
    builder = Neo4jGraphBuilder()
    
    try:
        stats = builder.build()
        
        if stats:
            logger.info(f"Successfully built graph with {stats.nodes_created} nodes")
            return 0
        else:
            logger.error("Graph build failed")
            return 1
            
    finally:
        builder.close()


if __name__ == "__main__":
    exit(main())