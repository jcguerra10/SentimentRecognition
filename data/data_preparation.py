import pandas as pd

# Define constants for saved path and mapping
DATA_PATH = "cleaned/clean.csv"
# Map labels to 0 and 1 that represents low and high priority
label_map = {
    'sadness': 1,
    'empty': 0,
    'joy': 0,
    'happiness': 0,
    'fun': 0,
    'enthusiasm': 0,
    'relief': 0,
    'love': 0,
    'anger': 1,
    'hate': 1,
    'fear': 1,
    'worry': 1,
    'surprise': 0,
    'neutral': 0,
    'boredom': 0,
    'positive': 0,
    'negative': 1
}

def process_data(data: pd.DataFrame):
    """
    Process the data and save the processed data into a file, before use make sure label_map has
    the correct mapping for the labels inside the data pass, add other if needed.

    :param data: The input data to be processed wich has a column called 'text' with the text to be process
    and a column called 'label' with the emotion to be mapped.
    :type data: pandas.DataFrame

    Example usage:
    >>> process_data(data)
    """
    first_time = False
    # check if saved data exists
    try:
        # load saved data
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        first_time = True

    # load text and label columns
    processed_data = data[['text', 'label']]
    # rename column label to priority
    processed_data = processed_data.rename(columns={'label': 'priority'})

    # map labels to 0 and 1
    processed_data['priority'] = processed_data['priority'].replace(label_map)
    # Drop missing values
    processed_data = processed_data.dropna()

    if first_time:
        # save processed data
        processed_data.to_csv(DATA_PATH, index=False)
        return
    # merge  saved data with processed data
    data = pd.concat([data, processed_data], ignore_index=True)

    # Save data
    data.to_csv(DATA_PATH, index=False)

