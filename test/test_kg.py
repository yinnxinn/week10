import os
from neo4j import GraphDatabase

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")


print(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD )

class Neo4jDemo:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_entity_relationships(self, entity_name):
        """
        Query all relationships and connected entities for a given entity name.
        """
        print(f"\n--- Querying relationships for: {entity_name} ---")
        query = """
        MATCH (h:Entity {name: $name})-[r]->(t:Entity)
        RETURN h.name AS head, type(r) AS relation, t.name AS tail
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, name=entity_name)
            records = list(result)
            if not records:
                print("No relationships found.")
            for record in records:
                print(f"{record['head']} --[{record['relation']}]--> {record['tail']}")

    def query_by_relation(self, relation_type):
        """
        Query sample triples with a specific relationship type.
        """
        print(f"\n--- Querying samples for relation: {relation_type} ---")
        query = f"""
        MATCH (h:Entity)-[r:{relation_type}]->(t:Entity)
        RETURN h.name AS head, t.name AS tail
        LIMIT 5
        """
        with self.driver.session() as session:
            try:
                result = session.run(query)
                for record in result:
                    print(f"{record['head']} --[{relation_type}]--> {record['tail']}")
            except Exception as e:
                print(f"Query failed: {e}")

    def run_demo(self):
        # 1. Test Connection
        print("Connecting to Neo4j...")
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Connection successful!")
        except Exception as e:
            print(f"Connection failed: {e}")
            return

        # 2. Example Queries based on typical medical data
        # "肺泡蛋白质沉积症" is from the user's snippet
        self.query_entity_relationships("肺泡蛋白质沉积症")
        
        # Query typical medical relation
        self.query_by_relation("disease_has_symptom")
        self.query_by_relation("disease_need_check")

if __name__ == "__main__":
    demo = Neo4jDemo(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        demo.run_demo()
    finally:
        demo.close()
