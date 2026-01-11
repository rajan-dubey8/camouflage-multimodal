"""
Extract KG embeddings for fusion 
Works with restructured graph where camouflage is on Environment
"""

import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from train_model import KnowledgeGraphGNN, Neo4jGraphExtractorV2

from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
TARGET_DB = os.getenv("TARGET_DB")

# ============================================================
#           EMBEDDING EXTRACTION
# ============================================================

def extract_category_embedding(model, extractor, category, device):
    """Extract embedding for organism category"""
    print(f"\n📊 Extracting: {category}")
    
    subgraphs = extractor.extract_category_subgraphs(category, limit=10)
    
    if not subgraphs:
        print(f"  ⚠️  No data for {category}")
        return None
    
    print(f"  ✓ Found {len(subgraphs)} samples")
    
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for subgraph in subgraphs:
            subgraph = subgraph.to(device)
            embedding = model.get_embedding(subgraph)
            embeddings.append(embedding.cpu())
    
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    
    print(f"  ✓ Shape: {avg_embedding.shape}")
    print(f"  ✓ Norm: {avg_embedding.norm().item():.4f}")
    
    return avg_embedding

def batch_extract_embeddings(model, extractor, output_dir='kg_embeddings_v2', device='cpu'):
    """Extract embeddings for all categories"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("BATCH EMBEDDING EXTRACTION V2")
    print("="*70)
    
    with extractor.driver.session(database=extractor.database) as session:
        result = session.run("""
            MATCH (o:Organism)
            WHERE o.category IS NOT NULL
            RETURN DISTINCT o.category AS category, COUNT(o) AS count
            ORDER BY count DESC
        """)
        categories = [(r['category'], r['count']) for r in result]
    
    print(f"\nFound {len(categories)} categories")
    
    embeddings = {}
    stats = {}
    
    for category, count in tqdm(categories, desc="Extracting"):
        try:
            embedding = extract_category_embedding(model, extractor, category, device)
            
            if embedding is not None:
                embeddings[category] = embedding
                stats[category] = {
                    'organism_count': count,
                    'embedding_norm': embedding.norm().item(),
                    'embedding_mean': embedding.mean().item(),
                    'embedding_std': embedding.std().item()
                }
                
                filename = f"{category.lower().replace(' ', '_')}_embedding.pt"
                filepath = os.path.join(output_dir, filename)
                torch.save({
                    'embedding': embedding,
                    'category': category,
                    'organism_count': count,
                    'embedding_dim': embedding.shape[-1]
                }, filepath)
        
        except Exception as e:
            print(f"\n  ❌ Error: {category}: {e}")
    
    # Save combined
    combined_path = os.path.join(output_dir, 'all_embeddings.pt')
    torch.save(embeddings, combined_path)
    print(f"\n💾 Saved: {combined_path}")
    
    # Save stats
    stats_path = os.path.join(output_dir, 'embedding_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Saved: {stats_path}")
    
    # Summary
    summary = {
        'num_categories': len(embeddings),
        'embedding_dim': 128,
        'categories': list(embeddings.keys()),
        'model_path': 'kg_gnn_model_v2.pth',
        'graph_version': 'v2'
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved: {summary_path}")
    
    return embeddings, stats

# ============================================================
#           TESTING
# ============================================================

def test_model_predictions(model, extractor, device, num_categories=5):
    """Test model predictions"""
    print("\n" + "="*70)
    print("MODEL PREDICTION TESTING")
    print("="*70)
    
    with extractor.driver.session(database=extractor.database) as session:
        result = session.run("""
            MATCH (o:Organism)
            RETURN DISTINCT o.category AS category
            LIMIT 5
        """)
        test_categories = [r['category'] for r in result]
    
    model.eval()
    
    for category in test_categories:
        print(f"\n📊 Testing: {category}")
        
        subgraphs = extractor.extract_category_subgraphs(category, limit=5)
        
        if not subgraphs:
            print("  ⚠️  No samples")
            continue
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for subgraph in subgraphs:
                subgraph = subgraph.to(device)
                pred = model(subgraph).item()
                gt = subgraph.y.item()
                
                predictions.append(pred)
                ground_truths.append(gt)
        
        import numpy as np
        mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))
        
        print(f"  ✓ Samples: {len(predictions)}")
        print(f"  ✓ MAE: {mae:.4f}")
        print(f"  ✓ Avg pred: {np.mean(predictions):.4f}")
        print(f"  ✓ Avg GT: {np.mean(ground_truths):.4f}")

def compare_embeddings(embeddings_dict):
    """Compare embedding similarity"""
    print("\n" + "="*70)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("="*70)
    
    categories = list(embeddings_dict.keys())
    print(f"\nComparing {len(categories)} categories")
    
    similarities = {}
    
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories[i+1:], start=i+1):
            emb1 = embeddings_dict[cat1]
            emb2 = embeddings_dict[cat2]
            sim = F.cosine_similarity(emb1, emb2, dim=-1).item()
            similarities[f"{cat1} vs {cat2}"] = sim
    
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Most similar:")
    for pair, sim in sorted_sims[:3]:
        print(f"  {pair}: {sim:.4f}")
    
    print("\n📊 Most different:")
    for pair, sim in sorted_sims[-3:]:
        print(f"  {pair}: {sim:.4f}")
    
    avg_sim = sum(similarities.values()) / len(similarities)
    print(f"\n📊 Average similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.9:
        print("  ⚠️  Embeddings very similar - consider longer training")
    elif avg_sim < 0.3:
        print("  ✅ Good separation")
    else:
        print("  ✓ Reasonable separation")

# ============================================================
#           MAIN
# ============================================================

def main_test(model_path='kg_gnn_model_v2.pth', output_dir='kg_embeddings_v2'):
    """Main testing and extraction"""
    print("="*70)
    print("KG-GNN TESTING & EMBEDDING EXTRACTION V2")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load model
    print(f"\n📂 Loading: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = KnowledgeGraphGNN(
        in_channels=32,
        hidden_channels=128,
        embedding_dim=checkpoint['embedding_dim'],
        out_channels=1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded (epoch {checkpoint['epoch']}, val loss: {checkpoint.get('val_loss', 'N/A')})")
    
    # Initialize extractor
    extractor = Neo4jGraphExtractorV2(NEO4J_URI, NEO4J_USER, NEO4J_PASS, TARGET_DB)
    
    # Test predictions
    test_model_predictions(model, extractor, device)
    
    # Extract embeddings
    embeddings, stats = batch_extract_embeddings(model, extractor, output_dir, device)
    
    # Compare
    compare_embeddings(embeddings)
    
    extractor.close()
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)
    print(f"\n📁 Saved to: {output_dir}/")
    print("   Files:")
    print("   - all_embeddings.pt")
    print("   - embedding_stats.json")
    print("   - summary.json")
    print("   - <category>_embedding.pt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kg_gnn_model_v2.pth')
    parser.add_argument('--output', type=str, default='kg_embeddings_v2')
    parser.add_argument('--category', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.category:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.model, map_location=device)
        
        model = KnowledgeGraphGNN(in_channels=32, hidden_channels=128, embedding_dim=128, out_channels=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        extractor = Neo4jGraphExtractorV2(NEO4J_URI, NEO4J_USER, NEO4J_PASS, TARGET_DB)
        embedding = extract_category_embedding(model, extractor, args.category, device)
        
        if embedding is not None:
            os.makedirs(args.output, exist_ok=True)
            filename = f"{args.category.lower().replace(' ', '_')}_embedding.pt"
            filepath = os.path.join(args.output, filename)
            torch.save({
                'embedding': embedding,
                'category': args.category,
                'embedding_dim': embedding.shape[-1]
            }, filepath)
            print(f"\n💾 Saved: {filepath}")
        
        extractor.close()
    else:
        main_test(args.model, args.output)