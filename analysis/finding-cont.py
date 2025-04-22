import pandas as pd
import ast

file_path = "papers_final-asian-withRace.csv"
df = pd.read_csv(file_path)
race_mapping = {
    "W_NL": 0.4655,
    "A":  0.6749,
    "HL": 0.9163,
    "B_NL":  0.9434,
}

def map_race(race_str):
    try:
        race_list = ast.literal_eval(race_str) if isinstance(race_str, str) else [race_str]
        mapped_values = [race_mapping[race] for race in race_list if race in race_mapping]
        return max(mapped_values) if mapped_values else None
    except:
        return None

df["Race_Continuous"] = df["Race"].apply(map_race)

output_path = "mapped_papers.csv"
df.to_csv(output_path, index=False)
