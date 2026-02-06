# src/custom_algorithms.py

def merge_sort(arr, key):
    """Custom merge sort for a list of dictionaries by a given key."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key)
    right = merge_sort(arr[mid:], key)

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][key] <= right[j][key]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


def binary_search_threshold(arr, key, threshold):
    """
    Custom binary search to find all items in a sorted list of dicts
    where dict[key] >= threshold.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid][key] < threshold:
            low = mid + 1
        else:
            high = mid - 1
    return arr[low:]  # all items >= threshold
