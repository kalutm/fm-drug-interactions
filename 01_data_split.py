import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits():
    # 1. Load data
    df = pd.read_csv('db_drug_interactions_severity.csv')

    # 2. Create a canonical pair ID to force (A,B) and (B,A) into the same group
    df['canonical'] = df.apply(lambda x: tuple(sorted([x['Drug 1'], x['Drug 2']])), axis=1)

    # 3. Group by canonical pair to determine the group label for stratification
    # For the 14 reversed rows with conflicting labels, taking the first label is mathematically 
    # sufficient just for determining the split distribution.
    group_labels = df.groupby('canonical')['Label'].first().reset_index()

    # 4. Split the groups (80% Train, 10% Val, 10% Test) using SEED=42
    train_groups, temp_groups = train_test_split(
        group_labels['canonical'], 
        test_size=0.2, 
        stratify=group_labels['Label'], 
        random_state=42
    )

    temp_labels = group_labels[group_labels['canonical'].isin(temp_groups)]
    val_groups, test_groups = train_test_split(
        temp_groups, 
        test_size=0.5, 
        stratify=temp_labels['Label'], 
        random_state=42
    )

    # 5. Map the grouped splits back to the original rows
    train_df = df[df['canonical'].isin(train_groups)].drop(columns=['canonical'])
    val_df = df[df['canonical'].isin(val_groups)].drop(columns=['canonical'])
    test_df = df[df['canonical'].isin(test_groups)].drop(columns=['canonical'])

    # 6. Save the official splits
    train_df.to_csv('split_train.csv', index=False)
    val_df.to_csv('split_val.csv', index=False)
    test_df.to_csv('split_test.csv', index=False)
    
    print(f"✅ Split successful. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    create_splits()