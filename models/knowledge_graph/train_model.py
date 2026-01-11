"""
Knowledge Graph GNN Training 
Updated for new graph structure where camouflage belongs to Environment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from neo4j import GraphDatabase
import numpy as np
import os
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
TARGET_DB = os.getenv("TARGET_DB")

# ============================================================
#           MODEL 
# ============================================================

class KnowledgeGraphGNN(nn.Module):
    """GNN for encoding organism-environment relationships"""
    
    def __init__(self, in_channels=32, hidden_channels=128, embedding_dim=128, out_channels=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Embedding layer for fusion
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier for training
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_channels)
        )
    
    def forward(self, data, return_embedding=False):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x_graph = global_mean_pool(x, batch)
        embedding = self.embedding_layer(x_graph)
        
        if return_embedding:
            return embedding
        
        output = self.classifier(embedding)
        return output
    
    def get_embedding(self, data):
        return self.forward(data, return_embedding=True)

# ============================================================
#           DATA EXTRACTION
# ============================================================

class Neo4jGraphExtractorV2:
    """Extract subgraphs """
    
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.node_feature_dim = 32
        
        self.node_types = ['Organism', 'Color', 'Texture', 'Pattern', 'Environment', 
                          'CamouflageAssessment', 'SimilarityMetric', 'LightingCondition', 
                          'ObservationContext']
        
        self.color_vocab = ['green', 'brown', 'gray', 'grey', 'yellow', 'orange',
                           'blue', 'white', 'black', 'red', 'beige', 'sandy']
        
        self.texture_vocab = ['smooth', 'rough', 'scaly', 'scaled', 'bumpy', 'fuzzy', 
                             'slimy', 'hard', 'soft', 'pebbled']
    
    def close(self):
        self.driver.close()
    
    def extract_category_subgraphs(self, category, limit=50):
        """
        Extract subgraphs for organisms of given category
        
        New query follows:
        Organism -> ObservationContext -> Environment -> CamouflageAssessment
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (o:Organism {category: $category})
            MATCH (oc:ObservationContext)-[:HAS_ORGANISM]->(o)
            MATCH (oc)-[:OBSERVED_IN]->(e:Environment)
            MATCH (e)-[:HAS_CAMOUFLAGE_ASSESSMENT]->(ca:CamouflageAssessment)
            MATCH (ca)-[:HAS_SIMILARITY]->(sm:SimilarityMetric)
            OPTIONAL MATCH (o)-[:HAS_COLOR]->(oc_color:Color)
            OPTIONAL MATCH (o)-[:HAS_TEXTURE]->(ot:Texture)
            OPTIONAL MATCH (o)-[:HAS_PATTERN]->(op:Pattern)
            OPTIONAL MATCH (e)-[:HAS_COLOR]->(ec:Color)
            OPTIONAL MATCH (e)-[:HAS_TEXTURE]->(et:Texture)
            OPTIONAL MATCH (e)-[:HAS_LIGHTING_CONDITION]->(lc:LightingCondition)
            RETURN o, oc, e, ca, sm,
                   collect(DISTINCT oc_color) as org_colors,
                   collect(DISTINCT ot) as org_textures,
                   collect(DISTINCT op) as org_patterns,
                   collect(DISTINCT ec) as env_colors,
                   collect(DISTINCT et) as env_textures,
                   lc
            LIMIT $limit
            """
            
            result = session.run(query, category=category, limit=limit)
            
            subgraphs = []
            for record in result:
                subgraph = self._build_subgraph_from_record(record)
                if subgraph is not None:
                    subgraphs.append(subgraph)
            
            return subgraphs
    
    def _build_subgraph_from_record(self, record):
        """Convert Neo4j record to PyG Data"""
        nodes = {}
        edges = []
        node_id_counter = 0
        
        # Organism
        organism = record['o']
        org_id = node_id_counter
        nodes[org_id] = {
            'type': 'Organism',
            'category': organism.get('category', 'Unknown'),
            'name': organism.get('name', 'Unknown')
        }
        node_id_counter += 1
        
        # Observation Context
        oc = record['oc']
        oc_id = node_id_counter
        nodes[oc_id] = {'type': 'ObservationContext'}
        edges.append((oc_id, org_id, 'HAS_ORGANISM'))
        node_id_counter += 1
        
        # Environment
        env = record['e']
        env_id = node_id_counter
        nodes[env_id] = {
            'type': 'Environment',
            'env_type': env.get('type', 'unknown')
        }
        edges.append((oc_id, env_id, 'OBSERVED_IN'))
        node_id_counter += 1
        
        # Camouflage Assessment (TARGET NODE)
        ca = record['ca']
        ca_id = node_id_counter
        nodes[ca_id] = {
            'type': 'CamouflageAssessment',
            'camouflage_score': float(ca.get('camouflage_score', 0.5)),
            'confidence': float(ca.get('confidence', 0.5)),
            'is_camouflaged': bool(ca.get('is_camouflaged', False))
        }
        edges.append((env_id, ca_id, 'HAS_CAMOUFLAGE_ASSESSMENT'))
        node_id_counter += 1
        
        # Similarity Metric
        sm = record['sm']
        sm_id = node_id_counter
        nodes[sm_id] = {
            'type': 'SimilarityMetric',
            'color_sim': float(sm.get('color_similarity', 0.5)),
            'texture_sim': float(sm.get('texture_similarity', 0.5)),
            'contrast': float(sm.get('contrast_difference', 0.5))
        }
        edges.append((ca_id, sm_id, 'HAS_SIMILARITY'))
        node_id_counter += 1
        
        # Organism colors (shared nodes)
        for color_node in record['org_colors']:
            if color_node:
                color_id = node_id_counter
                nodes[color_id] = {
                    'type': 'Color',
                    'name': color_node.get('name', 'unknown'),
                    'from': 'organism'
                }
                edges.append((org_id, color_id, 'HAS_COLOR'))
                node_id_counter += 1
        
        # Organism textures
        for texture_node in record['org_textures']:
            if texture_node:
                texture_id = node_id_counter
                nodes[texture_id] = {
                    'type': 'Texture',
                    'name': texture_node.get('name', 'unknown'),
                    'from': 'organism'
                }
                edges.append((org_id, texture_id, 'HAS_TEXTURE'))
                node_id_counter += 1
        
        # Organism patterns
        for pattern_node in record['org_patterns']:
            if pattern_node:
                pattern_id = node_id_counter
                nodes[pattern_id] = {
                    'type': 'Pattern',
                    'name': pattern_node.get('type', 'unknown')
                }
                edges.append((org_id, pattern_id, 'HAS_PATTERN'))
                node_id_counter += 1
        
        # Environment colors (shared nodes)
        for color_node in record['env_colors']:
            if color_node:
                color_id = node_id_counter
                nodes[color_id] = {
                    'type': 'Color',
                    'name': color_node.get('name', 'unknown'),
                    'from': 'environment'
                }
                edges.append((env_id, color_id, 'HAS_COLOR'))
                node_id_counter += 1
        
        # Environment textures
        for texture_node in record['env_textures']:
            if texture_node:
                texture_id = node_id_counter
                nodes[texture_id] = {
                    'type': 'Texture',
                    'name': texture_node.get('name', 'unknown'),
                    'from': 'environment'
                }
                edges.append((env_id, texture_id, 'HAS_TEXTURE'))
                node_id_counter += 1
        
        # Lighting condition
        if record['lc']:
            lc_id = node_id_counter
            nodes[lc_id] = {
                'type': 'LightingCondition',
                'condition': record['lc'].get('condition', 'bright')
            }
            edges.append((env_id, lc_id, 'HAS_LIGHTING_CONDITION'))
            node_id_counter += 1
        
        if len(nodes) < 2:
            return None
        
        node_features = self._encode_nodes(nodes)
        edge_index = self._encode_edges(edges)
        
        # Target is from CamouflageAssessment
        target = torch.tensor([nodes[ca_id]['camouflage_score']], dtype=torch.float)
        
        data = Data(x=node_features, edge_index=edge_index, y=target)
        return data
    
    def _encode_nodes(self, nodes):
        """Encode nodes to feature vectors"""
        features = []
        
        for node_id in sorted(nodes.keys()):
            node = nodes[node_id]
            feat = np.zeros(self.node_feature_dim, dtype=np.float32)
            
            # Node type one-hot (0-8)
            if node['type'] in self.node_types:
                feat[self.node_types.index(node['type'])] = 1.0
            
            # Numeric features (9-11)
            if node['type'] == 'CamouflageAssessment':
                feat[9] = node.get('camouflage_score', 0.5)
                feat[10] = node.get('confidence', 0.5)
                feat[11] = 1.0 if node.get('is_camouflaged', False) else 0.0
            elif node['type'] == 'SimilarityMetric':
                feat[9] = node.get('color_sim', 0.5)
                feat[10] = node.get('texture_sim', 0.5)
                feat[11] = node.get('contrast', 0.5)
            
            # Color vocabulary (12-23)
            if node['type'] == 'Color':
                color_name = node.get('name', '').lower()
                for i, vocab_color in enumerate(self.color_vocab):
                    if vocab_color in color_name:
                        feat[12 + i] = 1.0
            
            # Texture vocabulary (24-31)
            if node['type'] == 'Texture':
                texture_name = node.get('name', '').lower()
                for i, vocab_texture in enumerate(self.texture_vocab[:8]):
                    if vocab_texture in texture_name:
                        feat[24 + i] = 1.0
            
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _encode_edges(self, edges):
        """Bidirectional edges"""
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_list = []
        for src, dst, rel_type in edges:
            edge_list.append([src, dst])
            edge_list.append([dst, src])
        
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# ============================================================
#           TRAINING
# ============================================================

def create_dataset_from_neo4j(extractor):
    """Create dataset from V2 graph"""
    print("\n📊 Creating dataset from Neo4j V2...")
    
    with extractor.driver.session(database=extractor.database) as session:
        result = session.run("""
            MATCH (o:Organism)
            WHERE o.category IS NOT NULL
            RETURN DISTINCT o.category AS category, COUNT(o) AS count
            ORDER BY count DESC
        """)
        categories = [(r['category'], r['count']) for r in result]
    
    print(f"   Found {len(categories)} categories")
    
    all_subgraphs = []
    for category, count in tqdm(categories, desc="Extracting"):
        subgraphs = extractor.extract_category_subgraphs(category, limit=50)
        all_subgraphs.extend(subgraphs)
    
    print(f"✅ Created {len(all_subgraphs)} samples")
    return all_subgraphs

def train_kg_gnn(model, train_loader, val_loader, epochs=50, lr=0.001, save_path='kg_gnn_model_v2.pth'):
    """Train model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    
    print(f"\n🏋️  Training on {device}")
    print(f"   Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            loss = criterion(output.squeeze(), batch.y.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output.squeeze(), batch.y.squeeze())
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'embedding_dim': model.embedding_dim,
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"  → Best model saved!")
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    return model

# ============================================================
#           MAIN
# ============================================================

def main_train():
    print("="*70)
    print("KG-GNN TRAINING V2")
    print("="*70)
    
    extractor = Neo4jGraphExtractorV2(NEO4J_URI, NEO4J_USER, NEO4J_PASS, TARGET_DB)
    
    dataset = create_dataset_from_neo4j(extractor)
    extractor.close()
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = KnowledgeGraphGNN(
        in_channels=32,
        hidden_channels=128,
        embedding_dim=128,
        out_channels=1
    )
    
    print(f"\n📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model = train_kg_gnn(model, train_loader, val_loader, epochs=50, lr=0.001)
    
    print("\n✅ Model saved to: kg_gnn_model_v2.pth")

if __name__ == "__main__":
    main_train()