import pandas as pd

files = ['CombinedSets.csv', 'TestingSet.csv', 'TrainingSet.csv']

def check(filepath):
    df = pd.read_csv(filepath)
    type_counts = df['type'].value_counts()
    print(type_counts)



import csv

def find_unique_states(csv_file_path):
    unique_states = set()
    
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = row.get("State")
            if state:
                unique_states.add(state.strip())
    
    return unique_states

# Example usage
if __name__ == "__main__":
    path_to_csv = "CombinedSets.csv"  # Replace with the path to your CSV file
    unique_states = find_unique_states(path_to_csv)
    
    print("Unique states found:")
    for state in sorted(unique_states):
        print(state)



exit()
for i in files:
    print("============= ", i, " ================")
    check(i)


