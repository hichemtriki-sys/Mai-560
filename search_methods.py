
from bisect import bisect_left


def linear_search_by_id(records, target_id):
    """
    Sequentially inspect records until the target ID is found.

    Time complexity: O(n)
    """
    for record in records:
        if record["Record_ID"] == target_id:
            return [record]

    return []


def binary_search_by_id(records, target_id):
    """
    Search sorted records by repeatedly halving
    the remaining search interval.

    Time complexity: O(log n)
    """
    left = 0
    right = len(records) - 1

    while left <= right:
        mid = (left + right) // 2
        current_id = records[mid]["Record_ID"]

        if current_id == target_id:
            return [records[mid]]

        if current_id < target_id:
            left = mid + 1
        else:
            right = mid - 1

    return []


def hash_search_by_id(record_lookup, target_id):
    """
    Retrieve a record using its unique ID.

    Average lookup complexity: O(1)
    """
    result = record_lookup.get(target_id)
    return [result] if result is not None else []


def linear_search_multi(records):
    """
    Sequentially inspect every record for
    the four required conditions.

    Time complexity: O(n)
    """
    matches = []

    for record in records:
        if (
            record["Country"] == "Colombia"
            and record["Program"] == "MS AI"
            and record["GPA"] >= 3.50
            and record["Attendance_Pct"] < 80
        ):
            matches.append(record)

    return matches
