import json
import os
import numpy as np
from numpy import asarray
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

DATABASE_PATH = "database.json"

def classify_objects(feature_vectors, eps=0.5):
    if os.path.exists(DATABASE_PATH) and os.path.getsize(DATABASE_PATH) > 0:
        with open(DATABASE_PATH, 'r') as file:
            database = {int(k): np.array(v) for k, v in json.load(file).items()}
    else:
        database = {}

    if not feature_vectors:
        return []

    assigned_groups = []
    next_group_index = max(database.keys()) + 1 if database else 0

    for query_vector in feature_vectors:
        query_vector = query_vector.flatten()
        query_normal = np.linalg.norm(query_vector)

        if query_normal == 0:
            assigned_groups.append(-1)
            continue

        query_unit = query_vector / query_normal

        best_match_group = None
        best_similarity = -1.0

        for group_index, database_vector in database.items():
            database_normal = np.linalg.norm(database_vector)
            if database_normal == 0:
                continue
            database_unit = database_vector / database_normal
            similarity = np.dot(query_unit, database_unit)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_group = group_index

        if best_similarity >= eps:
            assigned_groups.append(best_match_group)
        else:
            assigned_groups.append(next_group_index)
            database[next_group_index] = query_vector
            next_group_index += 1

    json_ready_database = {str(k): v.tolist() for k, v in database.items()}
    with open(DATABASE_PATH, 'w') as file:
        json.dump(json_ready_database, file, indent=4)

    return assigned_groups

def classify_from_reference(feature_vectors, reference_vector, eps):
    if os.path.exists(DATABASE_PATH) and os.path.getsize(DATABASE_PATH) > 0:
        with open(DATABASE_PATH, 'r') as file:
            database = {int(k): np.array(v) for k, v in json.load(file).items()}
    else:
        raise ValueError("Database is empty or does not exist.")

    if not reference_vector:
        raise ValueError("Reference vector is empty or None.")
    
    if not feature_vectors:
            raise ValueError("Feature vectors list is empty or None.")

    similarity_results = []
    reference_vector = reference_vector.flatten()
    reference_normal = np.linalg.norm(reference_vector)

    if reference_normal == 0:
        similarity_results = [-1] * len(feature_vectors)
        return similarity_results

    reference_unit = reference_vector / reference_normal

    for query_vector in feature_vectors:
        query_vector = query_vector.flatten()
        query_normal = np.linalg.norm(query_vector)

        if query_normal == 0:
            similarity_results.append(-1)
            continue

        query_unit = query_vector / query_normal
        similarity = np.dot(query_unit, reference_unit)

        if similarity >= eps:
            similarity_results.append(True)
        else:
            similarity_results.append(False)