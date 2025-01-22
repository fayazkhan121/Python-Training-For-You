def binary_search(arr, target):
    """
    Performs binary search to find target element in a sorted array.
    
    Args:
        arr (list): Sorted list of numbers
        target (int): Element to find
        
    Returns:
        int: Index of target if found, -1 otherwise
    """
    # Initialize pointers for the start and end of search range
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        # Calculate middle index
        # Using (left + right) // 2 could cause integer overflow
        # This is a safer way to find the middle
        mid = left + (right - left) // 2
        
        # If element is found at middle, return its index
        if arr[mid] == target:
            return mid
        
        # If element is smaller than mid, search in left subarray
        elif arr[mid] > target:
            right = mid - 1
        
        # If element is larger than mid, search in right subarray
        else:
            left = mid + 1
    
    # Element not found
    return -1

# Example usage
def test_binary_search():
    # Test case 1: Element exists in array
    arr1 = [1, 3, 5, 7, 9, 11, 13, 15]
    target1 = 7
    result1 = binary_search(arr1, target1)
    print(f"Test 1: Finding {target1} in {arr1}")
    print(f"Result: Found at index {result1}\n")
    
    # Test case 2: Element doesn't exist
    arr2 = [2, 4, 6, 8, 10]
    target2 = 5
    result2 = binary_search(arr2, target2)
    print(f"Test 2: Finding {target2} in {arr2}")
    print(f"Result: {result2} (Element not found)\n")
    
    # Test case 3: Empty array
    arr3 = []
    target3 = 1
    result3 = binary_search(arr3, target3)
    print(f"Test 3: Finding {target3} in {arr3}")
    print(f"Result: {result3} (Empty array)\n")

if __name__ == "__main__":
    test_binary_search()
