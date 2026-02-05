import os
from neo4j import GraphDatabase
from tqdm import tqdm

# Neo4j configuration - You may need to change these
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

class KnowledgeGraphImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_triples(self, file_path):
        """
        Reads triples from a text file and imports them into Neo4j.
        File format expected: Entity1,Relationship,Entity2
        """
        print(f"Reading triples from {file_path}...")
        
        triples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    # Some entities might contain commas, so we handle split carefully
                    # Assuming format is always: Head,Relation,Tail
                    # But if commas are in entities, this simple split might fail.
                    # Given the sample: "肺泡蛋白质沉积症,disease_need_check,胸部CT检查"
                    # It seems safe to take first as head, second as relation, rest as tail?
                    # Or simple 3 parts. Let's assume standard 3 parts for now.
                    # If lines have more than 3 parts, we might need to adjust.
                    
                    if len(parts) == 3:
                        head, relation, tail = parts
                    else:
                        # Fallback: assume relation is the second element
                        head = parts[0]
                        relation = parts[1]
                        tail = ",".join(parts[2:])
                    
                    triples.append((head, relation, tail))

        print(f"Found {len(triples)} triples. Starting import...")

        # Batch processing
        batch_size = 1000
        for i in tqdm(range(0, len(triples), batch_size)):
            batch = triples[i:i + batch_size]
            self._write_batch(batch)
            
        print("Import completed.")

    def _write_batch(self, batch):
        with self.driver.session() as session:
            session.execute_write(self._create_relationships, batch)

    @staticmethod
    def _create_relationships(tx, batch):
        # Cypher query to merge nodes and create relationship
        # Using UNWIND for batch processing
        query = """
        UNWIND $batch AS row
        MERGE (h:Entity {name: row[0]})
        MERGE (t:Entity {name: row[2]})
        WITH h, t, row
        CALL apoc.create.relationship(h, row[1], {}, t) YIELD rel
        RETURN count(*)
        """
        
        # Standard Cypher without APOC (Dynamic relationships in pure Cypher are tricky)
        # Since relationship types are dynamic (disease_has_symptom, etc.), 
        # we can't easily parameterize the relationship type in pure Cypher like (:Entity)-[:$REL]->(:Entity).
        # We have two options:
        # 1. Use APOC (as above).
        # 2. Group by relationship type in Python and execute separate queries.
        
        # Let's use Option 2 (Python grouping) to avoid dependency on APOC plugin being installed.
        pass

    def import_triples_safe(self, file_path):
        """
        Imports triples grouping by relationship type to avoid APOC dependency.
        """
        print(f"Reading triples from {file_path}...")
        
        # Group by relationship type
        rel_groups = {}
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                
                # Handle potential commas in entities
                # Assuming the relation is always the second item is risky if the first entity has comma.
                # But based on typical KG dumps, usually Entity,Relation,Entity.
                # Let's stick to the assumption: Head, Relation, Tail
                
                head = parts[0]
                relation = parts[1]
                tail = ",".join(parts[2:]) # Join rest in case tail has comma
                
                if relation not in rel_groups:
                    rel_groups[relation] = []
                rel_groups[relation].append({'head': head, 'tail': tail})
                count += 1
        
        print(f"Found {count} triples with {len(rel_groups)} relationship types.")
        
        for relation, items in rel_groups.items():
            print(f"Importing {len(items)} relationships of type '{relation}'...")
            self._batch_insert(relation, items)
            
    def _batch_insert(self, relation, items, batch_size=1000):
        # Sanitize relationship type (remove special chars if any)
        # Neo4j relationship types should be alphanumeric
        sanitized_rel = "".join(c for c in relation if c.isalnum() or c == '_')
        if not sanitized_rel:
            sanitized_rel = "RELATED_TO"
            
        query = f"""
        UNWIND $batch AS row
        MERGE (h:Entity {{name: row.head}})
        MERGE (t:Entity {{name: row.tail}})
        MERGE (h)-[:{sanitized_rel}]->(t)
        """
        
        with self.driver.session() as session:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                session.run(query, batch=batch)

if __name__ == "__main__":
    # Path to triples file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    triples_file = os.path.join(base_dir, "data", "OpenCMKG", "triples.txt")
    
    if not os.path.exists(triples_file):
        print(f"Error: File not found at {triples_file}")
        exit(1)
        
    importer = KnowledgeGraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        importer.import_triples_safe(triples_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Neo4j is running and credentials are correct.")
    finally:
        importer.close()

'''
Reading triples from d:\projects\classes\week10\week10\data\OpenCMKG\triples.txt...
Found 354755 triples with 43 relationship types.
Importing 59562 relationships of type 'disease_has_symptom'...
Importing 25082 relationships of type 'disease_acompany_disease'...
Importing 37 relationships of type 'department_belong_department'...
Importing 8792 relationships of type 'disease_belong_department'...
Importing 39315 relationships of type 'disease_need_check'...
Importing 14649 relationships of type 'disease_common_drug'...
Importing 60428 relationships of type 'disease_recommand_drug'...
Importing 27556 relationships of type 'disease_noteat_food'...
Importing 22191 relationships of type 'disease_eat_food'...
Importing 40087 relationships of type 'disease_recommand_food'...
Importing 17530 relationships of type 'drug_relate_producer'...
'''