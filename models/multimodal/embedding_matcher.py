"""
Embedding Matcher
Matches RG (image-level) embeddings with KG (category-level) embeddings

Strategy:
- Load RG embeddings from rg_embeddings/all_rg_embeddings.pt
- Load KG embeddings from kg_embeddings_v2/all_embeddings.pt
- Match based on organism category in image filename or metadata
"""

import torch
import os
import json
import numpy as np
from collections import defaultdict

class EmbeddingMatcher:
    """
    Match Region Graph embeddings to Knowledge Graph embeddings
    """
    def __init__(self, rg_embeddings_path, kg_embeddings_path, category_mapping=None):
        """
        Args:
            rg_embeddings_path: path to all_rg_embeddings.pt
            kg_embeddings_path: path to kg_embeddings_v2/all_embeddings.pt
            category_mapping: dict mapping image names to organism categories
                             If None, will try to extract from filenames
        """
        print(f"\n📂 Loading embeddings...")
        
        # Load RG embeddings
        print(f"   RG: {rg_embeddings_path}")
        self.rg_embeddings = torch.load(rg_embeddings_path)
        print(f"   ✓ Loaded {len(self.rg_embeddings)} RG embeddings")
        
        # Load KG embeddings
        print(f"   KG: {kg_embeddings_path}")
        self.kg_embeddings = torch.load(kg_embeddings_path)
        print(f"   ✓ Loaded {len(self.kg_embeddings)} KG category embeddings")
        
        # Category mapping
        self.category_mapping = category_mapping
        
        # Create category to ID mapping
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.kg_embeddings.keys())}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        
        print(f"\n📊 Categories: {len(self.category_to_id)}")
        print(f"   {list(self.category_to_id.keys())[:5]}...")
    
    def extract_category_from_filename(self, filename):
        """
        Extract organism category from COD10K filename
        
        Example: 'COD10K-CAM-1-Aquatic-1-Batfish-2.jpg' → 'Fish' or 'Batfish'
        
        Note: This is a heuristic. Adjust based on your dataset structure.
        """
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # COD10K naming: COD10K-CAM-{cam_id}-{environment}-{seq}-{organism}-{id}
        parts = name.split('-')
        
        if len(parts) >= 6:
            organism_name = parts[5]  # e.g., 'Batfish', 'Octopus', 'Crab'
            
            # Try to match to KG categories
            # First try exact match
            if organism_name in self.kg_embeddings:
                return organism_name
            
            # Try partial match
            for category in self.kg_embeddings.keys():
                if organism_name.lower() in category.lower() or category.lower() in organism_name.lower():
                    return category
        
        # Default: return None (will use all categories)
        return None
    
    def get_kg_embedding_for_image(self, image_name, use_all_categories=False):
        """
        Get KG embedding(s) for an image
        
        Args:
            image_name: name of the image
            use_all_categories: if True, return all KG embeddings (for attention)
        
        Returns:
            kg_emb: [num_categories, kg_dim] or [1, kg_dim]
            category_ids: list of category IDs
        """
        if use_all_categories:
            # Return all KG embeddings (for cross-attention)
            kg_emb_list = [emb for emb in self.kg_embeddings.values()]
            kg_emb = torch.stack(kg_emb_list)  # [num_categories, kg_dim]
            category_ids = list(range(len(self.kg_embeddings)))
            return kg_emb, category_ids
        else:
            # Try to find specific category
            if self.category_mapping and image_name in self.category_mapping:
                category = self.category_mapping[image_name]
            else:
                category = self.extract_category_from_filename(image_name)
            
            if category and category in self.kg_embeddings:
                kg_emb = self.kg_embeddings[category].unsqueeze(0)  # [1, kg_dim]
                category_ids = [self.category_to_id[category]]
            else:
                # Use all categories (average)
                kg_emb_list = [emb for emb in self.kg_embeddings.values()]
                kg_emb = torch.stack(kg_emb_list).mean(dim=0, keepdim=True)  # [1, kg_dim]
                category_ids = [0]  # Placeholder
            
            return kg_emb, category_ids
    
    def create_matched_dataset(self, use_all_kg_categories=True):
        """
        Create matched RG-KG dataset
        
        Args:
            use_all_kg_categories: if True, each image gets all KG categories (for cross-attention)
                                  if False, each image gets matched category only (for late fusion)
        
        Returns:
            matched_data: list of dicts with keys:
                - 'image_name': str
                - 'rg_node_embeddings': [num_nodes, 128]
                - 'rg_graph_embedding': [1, 128]
                - 'kg_embeddings': [num_kg, 128] or [1, 128]
                - 'category_ids': list of ints
        """
        matched_data = []
        
        print(f"\n🔗 Creating matched dataset...")
        print(f"   Strategy: {'All KG categories' if use_all_kg_categories else 'Matched categories'}")
        
        for image_name, rg_data in self.rg_embeddings.items():
            # Get RG embeddings
            rg_node_emb = rg_data['node_embeddings']  # [num_nodes, 128]
            rg_graph_emb = rg_data['graph_embedding']  # [1, 128]
            
            # Get KG embeddings
            kg_emb, category_ids = self.get_kg_embedding_for_image(
                image_name, use_all_categories=use_all_kg_categories
            )
            
            matched_data.append({
                'image_name': image_name,
                'rg_node_embeddings': rg_node_emb,
                'rg_graph_embedding': rg_graph_emb,
                'kg_embeddings': kg_emb,
                'category_ids': category_ids,
                'num_rg_nodes': rg_node_emb.shape[0],
                'num_kg_categories': kg_emb.shape[0]
            })
        
        print(f"   ✓ Matched {len(matched_data)} samples")
        
        # Statistics
        avg_rg_nodes = np.mean([d['num_rg_nodes'] for d in matched_data])
        avg_kg_cats = np.mean([d['num_kg_categories'] for d in matched_data])
        
        print(f"\n📊 Dataset statistics:")
        print(f"   Avg RG nodes: {avg_rg_nodes:.1f}")
        print(f"   Avg KG categories: {avg_kg_cats:.1f}")
        
        return matched_data
    
    def save_matched_dataset(self, output_path, use_all_kg_categories=True):
        """
        Create and save matched dataset
        """
        matched_data = self.create_matched_dataset(use_all_kg_categories)
        
        torch.save(matched_data, output_path)
        print(f"\n💾 Saved matched dataset: {output_path}")
        
        return matched_data

# ==================== TESTING ====================
if __name__ == "__main__":
    print("="*70)
    print("TESTING EMBEDDING MATCHER")
    print("="*70)
    
    # Paths (adjust to your structure)
    rg_path = "../region_graph/rg_embeddings/all_rg_embeddings.pt"
    kg_path = "../knowledge_graph/kg_embeddings_v2/all_embeddings.pt"
    
    if os.path.exists(rg_path) and os.path.exists(kg_path):
        matcher = EmbeddingMatcher(rg_path, kg_path)
        
        # Test matching
        matched_data = matcher.create_matched_dataset(use_all_kg_categories=True)
        
        # Show sample
        print(f"\n📋 Sample matched data:")
        sample = matched_data[0]
        print(f"   Image: {sample['image_name']}")
        print(f"   RG nodes: {sample['rg_node_embeddings'].shape}")
        print(f"   RG graph: {sample['rg_graph_embedding'].shape}")
        print(f"   KG: {sample['kg_embeddings'].shape}")
        print(f"   Categories: {sample['category_ids'][:5]}...")
        
        # Save
        matcher.save_matched_dataset("matched_embeddings.pt")
        
        print("\n✅ TEST PASSED!")
    else:
        print(f"\n⚠️  Embedding files not found:")
        print(f"   RG: {rg_path}")
        print(f"   KG: {kg_path}")
        print(f"\n   Please run extract_rg_embeddings.py and extract_kg_embeddings.py first!")