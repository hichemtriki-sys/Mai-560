
import pandas as pd

from search_methods import (
    linear_search_by_id,
    binary_search_by_id,
    hash_search_by_id,
    linear_search_multi,
)

print("Loading dataset...")

df = pd.read_excel(
    "AI_Search_500k_Dataset.xlsx",
    sheet_name="Data"
)

records = df.to_dict("records")

record_lookup = {
    record["Record_ID"]: record
    for record in records
}

target_id = "STU-425000"

print("\nEXACT-ID SEARCH TEST")

linear_result = linear_search_by_id(records, target_id)
binary_result = binary_search_by_id(records, target_id)
hash_result = hash_search_by_id(record_lookup, target_id)

print("Linear:", len(linear_result))
print("Binary:", len(binary_result))
print("Hash:", len(hash_result))

assert linear_result == binary_result == hash_result

print("\nMULTI-CONDITION SEARCH TEST")

multi_result = linear_search_multi(records)

print("Matching records:", len(multi_result))

assert len(multi_result) == 1711

print("\nAll tests passed successfully!")
