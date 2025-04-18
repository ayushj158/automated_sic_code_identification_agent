import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def load_data(year):
    # STEP 1: Define the path to your local SIC Excel file
    file_path = f"embeddings/file_data/sic_codes_{year}.xls"

    # STEP 2: Load the Excel file
    df = pd.read_excel(file_path)
    # STEP 3: Clean and standardize columns
    df.columns = [col.strip() for col in df.columns]

    # STEP 4: Rename to standard column names
    standardized_columns = {
        df.columns[0]: "SIC Code",
        df.columns[1]: "Description",
        df.columns[2]: "Section Name",
        df.columns[3]: "Section Description"
    }

    df = df.rename(columns=standardized_columns)

    # STEP 5: Display preview
    print("SIC dataset preview:")
    print(df.head(10))

    return df

def fetch_sic_codes_df(year):
    df = load_data(year=year)
    df.to_csv("./embeddings/file_data/cleaned_sic_codes.csv", index=False)

    cleaned_df = pd.read_csv("./embeddings/file_data/cleaned_sic_codes.csv") 
    return cleaned_df