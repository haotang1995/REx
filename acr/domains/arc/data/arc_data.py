import json
import os
def get_arc_data():
    current_folder = os.path.abspath('.')
    path = os.path.join(current_folder, 'acr/domains/arc/data/arc_subset/arc_subset.json')
    with open(path, 'r') as f:
        arc_dataset = json.load(f)
    return arc_dataset