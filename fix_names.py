import os

folder = "models/region_graph/rg_embeddings"

for filename in os.listdir(folder):
    if filename.endswith("_embedding.pt"):
        new_name = filename.replace("_embedding", "")
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
        print(f"✅ Renamed: {filename} → {new_name}")

print("\n🎉 All RG embeddings renamed successfully!")
