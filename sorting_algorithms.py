def bubble_sort(arr):
    """
    Bubble Sort: Repeatedly steps through the list, compares adjacent elements,
    and swaps them if they're in the wrong order.
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(arr)
    for i in range(n):
        # Flag to optimize if array is already sorted
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Compare adjacent elements
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        
        # If no swapping occurred, array is already sorted
        if not swapped:
            break
    return arr

def selection_sort(arr):
    """
    Selection Sort: Divides array into sorted and unsorted portions,
    repeatedly selects minimum element from unsorted portion.
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(arr)
    for i in range(n):
        # Find minimum element in unsorted portion
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap found minimum with first element of unsorted part
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """
    Insertion Sort: Builds final sorted array one item at a time,
    by repeatedly inserting a new element into the sorted portion.
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

def test_sorting_algorithms():
    """Test function to demonstrate all sorting algorithms."""
    # Test arrays
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 12, 1, 3],
        [],
        [1],
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    ]
    
    for i, arr in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {arr}")
        
        # Test each sorting algorithm with a copy of the array
        bubble_result = bubble_sort(arr.copy())
        selection_result = selection_sort(arr.copy())
        insertion_result = insertion_sort(arr.copy())
        
        print(f"Bubble Sort:    {bubble_result}")
        print(f"Selection Sort: {selection_result}")
        print(f"Insertion Sort: {insertion_result}")

if __name__ == "__main__":
    test_sorting_algorithms()


"""
SORTING ALGORITHMS EXPLANATION:

1. Bubble Sort:
    * Repeatedly compares adjacent elements and swaps them if they're in wrong order
    * Optimization: Stops if no swaps needed (array already sorted)
    * Best for: Small arrays or nearly sorted arrays
    * Disadvantage: Poor performance on large arrays

2. Selection Sort:
    * Divides array into sorted and unsorted portions
    * Finds minimum element in unsorted portion and puts it at the end of sorted portion
    * Best for: Small arrays or when memory space is limited
    * Advantage: Makes minimum number of swaps (O(n))

3. Insertion Sort:
    * Builds sorted array one element at a time
    * Takes each element and inserts it in its correct position in sorted portion
    * Best for: Small arrays or nearly sorted arrays
    * Advantage: Adaptive (runs faster on nearly sorted arrays)

Test Cases Include:
    * Random arrays
    * Small arrays
    * Empty array
    * Single-element array
    * Array with duplicates

Algorithm Strengths:
    * Bubble Sort: Simple to understand and implement
    * Selection Sort: Minimal memory writes
    * Insertion Sort: Efficient for small and nearly sorted arrays

Usage:
    1. Save as 'sorting_algorithms.py'
    2. Open terminal/command prompt
    3. Run: python sorting_algorithms.py
"""
