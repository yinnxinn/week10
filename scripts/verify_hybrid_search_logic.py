
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.knowledge_base import KnowledgeBase
from app.core.config import settings

class TestHybridSearch(unittest.TestCase):
    @patch("app.services.knowledge_base.MilvusClient")
    def test_connect_and_schema(self, mock_milvus_cls):
        # Setup mock
        mock_client = MagicMock()
        mock_milvus_cls.return_value = mock_client
        mock_client.has_collection.return_value = False # Force creation
        
        mock_schema = MagicMock()
        mock_client.create_schema.return_value = mock_schema
        
        # Initialize KB
        kb = KnowledgeBase()
        
        # Verify connection
        mock_milvus_cls.assert_called_with(settings.mivlus_host)
        
        # Verify schema creation
        mock_client.create_schema.assert_called()
        
        # Verify fields added
        # We can't easily check exact arguments order without complex matching, 
        # but we can check if add_field was called multiple times.
        self.assertTrue(mock_schema.add_field.call_count >= 5) 
        
        # Check specific fields in calls
        field_names = [call.kwargs.get('field_name') for call in mock_schema.add_field.call_args_list]
        self.assertIn('id', field_names)
        self.assertIn('text', field_names)
        self.assertIn('dense_vector', field_names)
        self.assertIn('sparse_vector', field_names)
        self.assertIn('department', field_names)
        
        # Verify function added (BM25)
        mock_schema.add_function.assert_called()
        
        # Verify indices
        mock_client.prepare_index_params.assert_called()
        
        # Verify create_collection
        mock_client.create_collection.assert_called()

    @patch("app.services.knowledge_base.MilvusClient")
    def test_extract_tags(self, mock_milvus_cls):
        kb = KnowledgeBase()
        tags = kb.extract_tags("请问儿科在哪里")
        self.assertIn("儿科", tags)
        
        tags = kb.extract_tags("没有科室信息")
        self.assertEqual(len(tags), 0)

    @patch("app.services.knowledge_base.MilvusClient")
    @patch("app.services.knowledge_base.RRFRanker")
    @patch("app.services.knowledge_base.AnnSearchRequest")
    def test_hybrid_search(self, mock_ann_req, mock_rrf, mock_milvus_cls):
        mock_client = MagicMock()
        mock_milvus_cls.return_value = mock_client
        
        # Mock search result
        mock_client.search.return_value = [[
            {"entity": {"text": "doc1", "department": "儿科"}, "score": 0.9},
            {"entity": {"text": "doc2", "department": "内科"}, "score": 0.8}
        ]]
        
        kb = KnowledgeBase()
        
        # Mock rerank to avoid loading model
        kb.rerank = MagicMock(return_value=[0.95, 0.85])
        
        import numpy as np
        query_vec = np.random.rand(512).astype(np.float32)
        
        results = kb.hybrid_search("儿科 咳嗽", query_vec, top_k=2)
        
        # Verify AnnSearchRequest created (called twice: dense and sparse)
        self.assertEqual(mock_ann_req.call_count, 2)
        
        # Verify filter expression in both calls
        # The query contains "儿科", so extract_tags should find it
        # and create filter "department in ['儿科']"
        for call in mock_ann_req.call_args_list:
            self.assertIn("department in ['儿科']", call.kwargs.get('expr', ''))
        
        # Verify search called with reqs
        mock_client.search.assert_called()
        
        # Verify RRFReranker used
        mock_rrf.assert_called()
        
        # Verify results returned
        self.assertEqual(len(results[0]), 2)
        self.assertIn("rerank_score", results[0][0])

if __name__ == "__main__":
    unittest.main()
