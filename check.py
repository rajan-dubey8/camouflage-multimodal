import os

rg_path = "models/region_graph/rg_embeddings"
mask_path = "data/COD10K/gt_object"

rg_files = [f.replace('.pt', '') for f in os.listdir(rg_path) if f.endswith('.pt')]
mask_files = [f.replace('.png', '') for f in os.listdir(mask_path) if f.endswith('.png')]

intersection = set(rg_files).intersection(mask_files)

print(f"Total RG: {len(rg_files)}")
print(f"Total masks: {len(mask_files)}")
print(f"Matched: {len(intersection)}")

if len(intersection) == 0:
    print("⚠️ No matches found! Dataset loader will see 0 valid samples.")
else:
    print(f"✅ Found {len(intersection)} matching RG–GT pairs.")
