import os
import sys
from neo4j import GraphDatabase
from pyexpat.errors import messages

# Add project root to sys.path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.services.llm import LLMService

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USEd   v,mvkkkR", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

class KGRAGDemo:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.llm = LLMService()

    def close(self):
        self.driver.close()

    def retrieve_knowledge(self, question: str) -> str:
        """
        Retrieves relevant knowledge from Neo4j based on the user's question.
        Strategy: Find entities in the graph that appear in the question, then fetch their relationships.
        """
        print(f"Searching KG for entities in question: '{question}'...")
        
        # 1. Identify entities mentioned in the question
        # We search for any Entity node whose name is contained in the question.
        # Note: This is a simple substring match. For production, use NER or Embedding retrieval.
        find_entities_query = """
        MATCH (n:Entity)
        WHERE $question CONTAINS n.name
        RETURN n.name AS entity_name
        LIMIT 5
        """
        
        entities = []
        with self.driver.session() as session:
            result = session.run(find_entities_query, question=question)
            entities = [record["entity_name"] for record in result]
            
        if not entities:
            print("No matching entities found in KG.")
            return ""

        print(f"Found entities: {entities}")
        
        # 2. Fetch 1-hop neighborhood for these entities
        context_triples = []
        for entity in entities:
            # Get outgoing relationships
            query_out = """
            MATCH (h:Entity {name: $name})-[r]->(t:Entity)
            RETURN h.name AS head, type(r) AS relation, t.name AS tail
            LIMIT 10
            """
            # Get incoming relationships (optional, but good for context)
            query_in = """
            MATCH (h:Entity)-[r]->(t:Entity {name: $name})
            RETURN h.name AS head, type(r) AS relation, t.name AS tail
            LIMIT 10
            """
            
            with self.driver.session() as session:
                res_out = session.run(query_out, name=entity)
                for record in res_out:
                    context_triples.append(f"{record['head']} -[{record['relation']}]-> {record['tail']}")
                
                res_in = session.run(query_in, name=entity)
                for record in res_in:
                    context_triples.append(f"{record['head']} -[{record['relation']}]-> {record['tail']}")

        # Deduplicate and format
        print(f' searched result: {context_triples}')
        unique_context = list(set(context_triples))
        context_str = "\n".join(unique_context)
        print(f"Retrieved {len(unique_context)} facts.")
        return context_str

    def answer_question(self, question: str):
        """
        Main RAG flow: Question -> KG Search -> Context -> LLM -> Answer
        """
        # Step 1: Retrieve Knowledge
        kg_context = self.retrieve_knowledge(question)

        
        if not kg_context:
            print("Insufficient knowledge found in KG to answer specifically.")
            kg_context = "无相关知识库信息。"

        # Step 2: Generate Answer using LLM
        print("Generating answer with LLM...")
        
        system_prompt = """你是一个基于知识图谱的智能问答助手。请利用提供的[知识图谱上下文]来回答用户的问题。
        
要求：
1. 答案必须基于提供的上下文事实。
2. 结合给你的知识库检索信息，以医学专家的身份给出靠谱的回答，帮助缓解用户的情绪
3. 按照逻辑整理答案，使其通顺易读。"""



        user_prompt = f"""用户问题：{question}

知识图谱上下文：
{kg_context}

请回答："""

        print(user_prompt)

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
                stream=True

            )

            for item in response:
                print(item)


            # answer = response.choices[0].message.content
            # print("\n" + "="*20 + " 最终回答 " + "="*20)
            # print(answer)
            # print("="*50)
            # return answer
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Error generating answer."

if __name__ == "__main__":
    demo = KGRAGDemo()
    try:
        # Example Question 1
        # q1 = "5个月宝宝体重超标严重，怎么办"
        # print(f"\nExample 1: {q1}")
        # demo.answer_question(q1)

        # # Example Question 2
        q2 = "肺泡蛋白质沉积症应该去哪个科室？需要做什么检查？"
        print(f"\nExample 2: {q2}")
        demo.answer_question(q2)
        
    finally:
        demo.close()

'''

==================== 最终回答 ====================
根据提供的知识图谱上下文，回答如下：

**就诊科室：**
肺泡蛋白质沉积症建议前往**呼吸内科**就诊。

**确诊检查项目：**
为了确认该疾病，建议进行以下检查：
1.  **胸部CT检查**
2.  **支气管镜检查**
3.  **肺活检**
==================================================
'''