import time
import numpy as np
from pymilvus import (
    MilvusClient,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")
milvus_client.drop_collection(collection_name)
milvus_client.create_collection(collection_name, dim, consistency_level="Bounded", metric_type="L2")

print("collections:", milvus_client.list_collections())
print(f"{collection_name} :", milvus_client.describe_collection(collection_name))
rng = np.random.default_rng(seed=19530)

rows = [
        {"id": 1, "vector": rng.random((1, dim))[0], "a": 1},
        {"id": 2, "vector": rng.random((1, dim))[0], "b": 1},
        {"id": 3, "vector": rng.random((1, dim))[0], "c": 1},
        {"id": 4, "vector": rng.random((1, dim))[0], "d": 1},
        {"id": 5, "vector": rng.random((1, dim))[0], "e": 1},
        {"id": 6, "vector": rng.random((1, dim))[0], "f": 1},
]
print(fmt.format("Start inserting entities"))
pks = milvus_client.insert(collection_name, rows, progress_bar=True)

print("len of pks:", len(pks), "first pk is :", pks[0])

print(f"get primary key {pks[0]} from {collection_name}")
first_pk_data = milvus_client.get(collection_name, pks[0:1], consistency_level="Strong")
print(f"data of primary key {pks[0]} is", first_pk_data)

pks2 = milvus_client.upsert(collection_name, {"id": 1, "vector": rng.random((1, dim))[0], "g": 1})

print("upsert pk:%d 's data"% pks2[0])

print(f"get primary key {pks[0]} from {collection_name}")
#first_pk_data = milvus_client.get(collection_name, pks[0:1])
first_pk_data = milvus_client.get(collection_name, pks[0:1], consistency_level="Strong")
print(f"data of primary key {pks[0]} is", first_pk_data)

milvus_client.drop_collection(collection_name)
