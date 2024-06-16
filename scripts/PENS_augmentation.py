import typer
import random
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

app = typer.Typer()

def generate_summ_id(counter):
    return 'S{}'.format(counter)

## Input: ref_df = pd.read_csv('../PENS/personalized_test.tsv', delimiter='\t', header=0, engine='python', index_col=0)
def prepare_one_to_one_reference_summ(ref_df):

    data = {
        "NewsID": [],
        "Headline": []
    }
    
    for index, row in ref_df.iterrows():
        news = row['posnewID'].split(',')
        titles = row['rewrite_titles'].split(';;')
        
        # Ensure we pair each title with a posnewsID (handling cases where they don't match in count)
        for i in range(len(news)):
            data["NewsID"].append(news[i])
            data["Headline"].append(titles[i])
    
    oto_ref_df = pd.DataFrame(data)
    oto_ref_df.to_csv('../datasets/oto_ref.csv', index=False)

@app.command()
def prepare_synthetic_data(
    input_file: str, 
    output_synthetic: str,
    output_summary: str, 
    start: int = -1, 
    end: int = -1, 
    summ_counter: int = 0
):
    """
    Prepare synthetic data by processing user interaction logs and generating summaries.

    Args:
        input_file (str): Path to the input TSV file containing user interaction data.
        output_synthetic (str): Path to save the processed synthetic data CSV file.
        output_summary (str): Path to save the summary data CSV file.
        start (int, optional): Starting index for processing the dataset. Defaults to -1, meaning start from the beginning.
        end (int, optional): Ending index for processing the dataset. Defaults to -1, meaning process till the end.
        summ_counter (int, optional): Initial counter value for generating summary IDs. Defaults to 0.

    Returns:
        None
    """
    print("Loading...")
    # Load the dataset
    user_df = pd.read_csv(input_file, delimiter='\t', engine='python')

    ref_df = pd.read_csv('../PENS/personalized_test.tsv', delimiter='\t', header=0, engine='python', index_col=0)
    prepare_one_to_one_reference_summ(ref_df)
    oto_ref_df = pd.read_csv('../datasets/oto_ref.csv')

    assert (start == -1 and end == -1) or ((start != -1 and end != -1))
    # Set the range to the whole dataset if start and end are -1
    if start == -1 and end == -1:
        start = 0
        end = user_df.shape[0]
        
        
    synthetic_df = pd.DataFrame(columns=['UserID', 'Docs', 'Action', 'Summaries'])
    summ_df = pd.DataFrame(columns=['SummID', 'NewsID', 'UserID', 'Summary'])

    for i, row in tqdm(user_df[start:end].iterrows(), total=user_df[start:end].shape[0]):
        h = np.array(row["ClicknewsID"].split(" "))  # history of clicked news IDs
        p = np.array(row["pos"].split(" "))  # positively clicked news IDs
        n = np.array(row["neg"].split(" "))  # negatively clicked or skipped news IDs

        t_h = np.zeros(len(h), dtype=int)  # tagged history, 0
        t_p = np.ones(len(p), dtype=int)  # tagged pos, 1
        t_n = np.full(len(n), 2, dtype=int)  # tagged neg, 2
        t_c = np.concatenate((t_h, t_p, t_n))  # combined

        np.random.shuffle(t_c)
        
        ns = np.array([], dtype=object)  # nodes
        es = np.array([], dtype=object)  # edges
        
        idxs = np.zeros(3, dtype=int)  # indices

        num_summ = 0
        
        for typ in t_c:
            if typ == 0:
                if idxs[0] < len(h):
                    ref = oto_ref_df[oto_ref_df['NewsID'] == h[idxs[0]]]
                    if ref.empty:
                        es = np.append(es, 'click')
                        ns = np.append(ns, h[idxs[0]])
                    else:
                        es = np.append(es, 'gen_summ')
                        ns = np.append(ns, h[idxs[0]])
                        es = np.append(es, 'summ_gen')
                        sample = ref.sample(n=1)

                        summ_counter += 1
                        summ_id = generate_summ_id(summ_counter)
                        # Append the new row to the DataFrame
                        summ_df = pd.concat([summ_df, 
                                             pd.DataFrame([{
                                                 'SummID': summ_id,
                                                 'NewsID': h[idxs[0]],
                                                 'UserID': row['UserID'],
                                                 'Summary': sample['Headline'].iloc[0]
                                             }])], ignore_index=True)
                        ns = np.append(ns, summ_id)
                        num_summ += 1
                    idxs[0] += 1
            elif typ == 1:
                if idxs[1] < len(p):
                    ref = oto_ref_df[oto_ref_df['NewsID'] == p[idxs[1]]]
                    if ref.empty:
                        es = np.append(es, 'click')
                        ns = np.append(ns, p[idxs[1]])
                    else:
                        es = np.append(es, 'gen_summ')
                        ns = np.append(ns, p[idxs[1]])
                        es = np.append(es, 'summ_gen')
                        sample = ref.sample(n=1)

                        summ_counter += 1
                        summ_id = generate_summ_id(summ_counter)
                        # Append the new row to the DataFrame
                        summ_df = pd.concat([summ_df, 
                                             pd.DataFrame([{
                                                 'SummID': summ_id,
                                                 'NewsID': p[idxs[1]],
                                                 'UserID': row['UserID'],
                                                 'Summary': sample['Headline'].iloc[0]
                                             }])], ignore_index=True)
                        ns = np.append(ns, summ_id)
                        num_summ += 1
                    idxs[1] += 1
            else:
                es = np.append(es, 'skip')
                ns = np.append(ns, n[idxs[2]])
                idxs[2] += 1
        
        assert len(ns) == len(es)
        synthetic_df = pd.concat([synthetic_df, pd.DataFrame([{ 
                                             'UserID': row['UserID'],
                                             'Docs': ns.tolist(), 
                                             'Action': es.tolist(),
                                             'Summaries': num_summ
                                         }])], ignore_index=True)
    
    # Save the resulting dataframes
    synthetic_df.to_csv(output_synthetic, index=False)
    summ_df.to_csv(output_summary, index=False)

def safe_eval(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val
    
@app.command()
def augment_synthetic_data(
    input_csv: str, 
    output_csv: str, 
    num_users: int = 5, 
    num_users_var: int = 25, 
    gap: int = 2, 
    gap_var: int = 5, 
    impute_gap: int = 2, 
    impute_gap_var: int = 5, 
    length: int = 100, 
    length_var: int = 200, 
    iterations: int = 3
):
    """
    Process a synthetic dataset by modifying and updating rows based on given parameters.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the processed CSV file.
        num_users (int, optional): Base number of users to sample. Defaults to 5.
        num_users_var (int, optional): Variability in the number of users to sample. Defaults to 25.
        gap (int, optional): Initial gap value. Defaults to 2.
        gap_var (int, optional): Variability in the gap value. Defaults to 5.
        impute_gap (int, optional): Initial impute gap value. Defaults to 2.
        impute_gap_var (int, optional): Variability in the impute gap value. Defaults to 5.
        length (int, optional): Base length of sequences to impute. Defaults to 100.
        length_var (int, optional): Variability in the length of sequences to impute. Defaults to 200.
        iterations (int, optional): Number of iterations to process the dataset. Defaults to 3.

    Returns:
        None
    """
    print(f"Loading Data form the path `{input_csv}`")
    synthetic_init_df = pd.read_csv(input_csv)
    print(f"Data form the path `{input_csv}` Successfully Loaded to the memory")

    for _ in range(iterations):
        for i, row in tqdm(synthetic_init_df.iterrows(), total=synthetic_init_df.shape[0]):
            sample_df = synthetic_init_df.sample(n = num_users + np.random.randint(0, num_users_var + 1))
            
            for si, srow in sample_df.iterrows():
                current_gap = gap + np.random.randint(0, gap_var + 1)
                current_length = length + np.random.randint(0, length_var + 1)
                impute_seq_docs = safe_eval(srow['Docs'])[current_gap:current_gap + current_length]
                impute_seq_acts = safe_eval(srow['Action'])[current_gap:current_gap + current_length]

                summ_gen_count = impute_seq_acts.count('summ_gen')
                if summ_gen_count == impute_seq_acts.count('gen_summ'):
                    current_impute_gap = impute_gap + np.random.randint(0, impute_gap_var + 1)
                    replace_seq_docs = (
                        safe_eval(row['Docs'])[:current_impute_gap] + 
                        impute_seq_docs + 
                        safe_eval(row['Docs'])[current_impute_gap:]
                    )
                    replace_seq_acts = (
                        safe_eval(row['Action'])[:current_impute_gap] + 
                        impute_seq_acts + 
                        safe_eval(row['Action'])[current_impute_gap:]
                    )
                    synthetic_init_df.at[i, 'Docs'] = replace_seq_docs
                    synthetic_init_df.at[i, 'Action'] = replace_seq_acts
                    assert len(synthetic_init_df.at[i, 'Docs']) == len(synthetic_init_df.at[i, 'Action'])
                    synthetic_init_df.at[i, 'Summaries'] =  replace_seq_acts.count('summ_gen')

    synthetic_init_df.to_csv(output_csv, index=False)

@app.command()
def extract_behaviors(
    input_csv: str,
    output_pkl: str,
):
    print(f"Loading Data from the path `{input_csv}`")
    synthetic_init_df = pd.read_csv(input_csv)
    print(f"Data from the path `{input_csv}` Successfully Loaded to the memory")

    # Initialize an empty list to hold the rows
    edges_list = []
    
    # Generate edges and assign edge IDs
    edge_id = 1
    for index, row in tqdm(synthetic_init_df.iterrows(), total=synthetic_init_df.shape[0]):
        user_id = row['UserID']
        docs = safe_eval(row['Docs'])
        actions = safe_eval(row['Action'])
    
        # Create an edge from the user to the first document
        if docs:
            edges_list.append({
                'EdgeID': f'B{edge_id}',
                'Head': user_id,
                'Relation': actions[0],
                'Tail': docs[0],
                'User': user_id
            })
            edge_id += 1
    
        # Create edges between the documents
        for i in range(len(docs) - 1):
            edges_list.append({
                'EdgeID': f'B{edge_id}',
                'Head': docs[i],
                'Relation': actions[i + 1],
                'Tail': docs[i + 1],
                'User': user_id
            })
            edge_id += 1
    
    # # Convert the list to a DataFrame
    # edges_df = pd.DataFrame(edges_list)
    
    # # Save the DataFrame to a CSV file
    # print(f"Saving to `{output_csv}`")
    # edges_df.to_csv(output_csv, index=False)
    
    
    with open(output_pkl, 'wb') as file:
        pickle.dump(edges_list, file)
    
    print(f"List successfully stored in {filename}")

@app.command()
def extract_runs(
    input_csv: str,
    output_csv: str,
):
    print(f"Loading Data from the path `{input_csv}`")
    synthetic_init_df = pd.read_csv(input_csv)
    print(f"Data from the path `{input_csv}` Successfully Loaded to the memory")

        # Initialize an empty DataFrame with the required columns
    edges_df = pd.DataFrame(columns=['EdgeID', 'Head', 'Relation', 'Tail', 'User'])
    
        # Initialize an empty list to hold the rows
    run_id = 1
    run_list = []
    for index, row in tqdm(synthetic_init_df.iterrows(), total=synthetic_init_df.shape[0]):
        user_id = row['UserID']
        docs = safe_eval(row['Docs'])
        actions = safe_eval(row['Action'])

        local_docs = []
        local_actions = []
        local_docs.append(user_id)

        for i in range(len(docs)):
            if actions[i] != "summ_gen":
                local_docs.append(docs[i])
                local_actions.append(actions[i])
            else:
                local_docs.append(docs[i])
                local_actions.append(actions[i])
                run_list.append({
                    'RunID': f'R{run_id}',
                    'UserID': user_id,
                    'Docs': local_docs,
                    'Actions': local_actions,
                })
                run_id += 1
                local_docs = []
                local_docs.append(docs[i])
                local_actions = []
                assert len(local_docs) == len(local_actions) + 1

    # Convert run_list to DataFrame
    runs_df = pd.DataFrame(run_list)

    runs_df.to_csv("synthetic-original-augmented-runs.csv", index=False)

    # Save the list to a pickle file
    with open(output_csv, 'wb') as file:
        pickle.dump(edges_list, file)
    
    print(f"List successfully stored in {output_pkl}")


if __name__ == "__main__":
    app()