import numpy
# --- GENERATED CODE ---
import numpy as np

def square_all_values(input_array):
    squared_list = [x ** 2 for x in input_array]
    squared_array = np.array(squared_list)
    return squared_list, squared_array

# --- UNCOMPILED CODE ---
def main():
    test_array = [1, 2, 3, 4, 5]
    list_result, np_result = square_all_values(test_array)
    print("Original array:", test_array)
    print("List result:", list_result)
    print("NumPy array result:", np_result)

if __name__ == '__main__':
    main()