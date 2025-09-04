import csv
import math


def parse_body_values():
    # Path to your CSV file     
    filename = 'TAB_Driveline_data.csv'

    # Initialize an empty dictionary
    inertia_dict = {}

    # Open and read the CSV file
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        
        # Skip the header
        next(reader)
        
        # Read each row and populate the dictionary
        for row in reader:
            if len(row) == 2:
                body = row[0].strip()
                try:
                    inertia = float(row[1].strip())
                    inertia_dict[body] = inertia
                except ValueError:
                    print(f"Warning: Could not convert inertia value for '{body}'")

    # Output the dictionary
    print(inertia_dict)
    return

def parse_node_values():
    # Path to your CSV file
    filename = 'TAB_Driveline_Springs.csv'

    # Initialize an empty dictionary
    spring_data = {}

    # Open and read the CSV file
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        
        # Skip the header
        next(reader)
        
        # Read each row and populate the dictionary
        for row in reader:
            if len(row) == 4:
                spring_name = row[0].strip()
                try:
                    stiffness = float(row[1].strip()) # puttin index 2 yields damping, 3 yields the ratio
                    damping = float(row[2].strip())
                    ratio = float(row[3].strip())
                    #spring_dict[spring_name] = stiffness

                    # store all values in a nested dictionary
                    spring_data[spring_name] = {
                        'stiffness': stiffness,
                        'damping': damping,
                        'ratio': ratio
                    }
        
                    
                except ValueError:
                    print(f"Warning: Could not convert stiffness value for '{spring_name}'")

    # Output the dictionary
    print(spring_data)
    return